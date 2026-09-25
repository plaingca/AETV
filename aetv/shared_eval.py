"""Shared scoring protocol for the release codecs.

Every model is scored on the same clips: the first 64 of the seed-2026 shuffle
of the OpenVid cache (192x108, 12 frames). Rows are the clean latent, unit-RMS
latent AWGN, the model's own modem on a clean waveform, and the model's own
modem through ``mpp12``. The fade seed for GOP ``g`` of clip ``i`` is
``seed0 + i * G + g``, with ``G`` GOPs per clip.

Statistics are per clip, so two models scored in one run can be compared with
a paired standard error over clips. Face-region PSNR uses YuNet boxes found on
the source frames; clips without a face are left out of that column.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import AETV_MODES
from .hfchannel import emulate
from .modem import demodulate_gop_stream, modulate_gop_stream
from .narrowband_jscc import global_ssim, impair_wire

DEFAULT_CACHE = "data/openvid_aetv_cache/mode_ac6_192x108_12f"
DEFAULT_FACE_MODEL = "data/teachers/face_detection_yunet_2023mar.onnx"
EVAL_CLIPS = 64
CLIP_FRAMES = 12
SPREAD_THRESHOLD = 0.05


def shuffled_cache(cache: str | Path = DEFAULT_CACHE, seed: int = 2026) -> list[Path]:
    files = sorted(Path(cache).glob("*.pt"))
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    return [files[index] for index in order]


def load_clips(paths: list[Path]) -> torch.Tensor:
    clips = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    tensor = torch.stack(clips)
    if tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
    return tensor


def eval_split(cache: str | Path = DEFAULT_CACHE, seed: int = 2026, clips: int = EVAL_CLIPS) -> list[Path]:
    return shuffled_cache(cache, seed)[:clips]


def training_pool(cache: str | Path = DEFAULT_CACHE, seed: int = 2026) -> list[Path]:
    """Clips outside the eval split. Calibration must draw only from here."""
    return shuffled_cache(cache, seed)[EVAL_CLIPS:]


def scale_confidence(confidence: torch.Tensor, gain: float, threshold: float = SPREAD_THRESHOLD) -> torch.Tensor:
    """Scale modem confidence by ``gain`` only where it varies across the GOP.

    Flat weights (latent AWGN, or a clean loopback with a uniform estimate)
    are returned unchanged, so a gain never moves the AWGN rows.
    """
    if gain == 1.0:
        return confidence
    spread = confidence.amax(dim=-1, keepdim=True) - confidence.amin(dim=-1, keepdim=True)
    scaled = (confidence * gain).clamp(0, 1)
    return torch.where(spread > threshold, scaled, confidence)


def resize_clip(clip: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear resize of a (B, 3, T, H, W) clip."""
    if clip.shape[-2:] == (height, width):
        return clip
    b, c, t = clip.shape[:3]
    frames = clip.permute(0, 2, 1, 3, 4).reshape(b * t, c, *clip.shape[-2:])
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)
    return frames.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4)


def modem_exchange(wire: np.ndarray, mode_name: str, seed: int, profile: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """One GOP through a mode's own modem. A GOP that does not demodulate returns zeros."""
    mode = AETV_MODES[mode_name]
    audio = modulate_gop_stream([np.ascontiguousarray(wire, dtype=np.float32)], mode_name=mode_name, callsign="EVAL")
    impaired = audio if profile == "clean" else emulate(audio, profile, seed=seed, fs=mode.geometry.fs)
    try:
        demod = demodulate_gop_stream(
            impaired, band=mode.band, drift_track="off", expected_mode=mode if mode_name == "AC16" else None
        )
    except Exception:  # noqa: BLE001 - an acquisition failure is a lost GOP, not a crash
        demod = None
    if demod is None or not demod.gops_latents:
        zeros = np.zeros_like(wire, dtype=np.float32)
        return zeros, zeros, False
    received = np.asarray(demod.gops_latents[0], dtype=np.float32)
    weights = np.clip(np.asarray(demod.gops_weights[0], dtype=np.float32), 0.0, 1.0)
    return received, weights, True


class CodecAdapter:
    """Uniform encode/decode over a 12-frame 192x108 clip."""

    name: str
    mode: str
    gop: int
    height: int = 108
    width: int = 192

    def gops(self, clip: torch.Tensor) -> list[torch.Tensor]:
        clip = resize_clip(clip, self.height, self.width)
        return [clip[:, :, start : start + self.gop] for start in range(0, CLIP_FRAMES - self.gop + 1, self.gop)]

    def frame_index(self) -> list[int]:
        return [start + k for start in range(0, CLIP_FRAMES - self.gop + 1, self.gop) for k in range(self.gop)]

    def encode(self, gops: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError

    def decode(self, wires: list[torch.Tensor], confidences: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError


class StatefulAdapter(CodecAdapter):
    """AC2K / AC6: GOPs carry TX and RX state across the clip."""

    def __init__(self, name: str, model, gop: int):
        self.name, self.model, self.gop, self.mode = name, model.eval(), gop, "V8"

    def encode(self, gops):
        self.model.reset()
        return [self.model.encode_gop(gop, retain_state=True) for gop in gops]

    def decode(self, wires, confidences):
        self.model.reset()
        out = []
        for wire, confidence in zip(wires, confidences):
            recon, _ = self.model.decode_gop(wire, confidence=confidence, retain_state=True)
            out.append(recon.clamp(0, 1))
        return out


class AutoencoderAdapter(CodecAdapter):
    """V8 / V7 release autoencoders: one independent GOP per modem GOP."""

    def __init__(self, name: str, path: str | Path, mode_name: str, device):
        from .models import AETVAutoencoder

        payload = torch.load(path, map_location="cpu", weights_only=False)
        args = payload.get("args", {}) or {}
        spec = AETV_MODES[mode_name]
        self.model = AETVAutoencoder(
            mode=spec,
            width=int(args.get("model_width", 128)),
            latent_channels=int(args.get("latent_channels", 3)),
            compact=bool(args.get("compact", False)),
            causal=spec.causal,
        )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(device).eval()
        self.name, self.mode, self.spec = name, mode_name, spec
        self.gop, self.height, self.width = spec.gop_frames, spec.height, spec.width

    def encode(self, gops):
        return [self.model.encoder(gop) for gop in gops]

    def decode(self, wires, confidences):
        shape = (self.spec.gop_frames, self.spec.height, self.spec.width)
        return [self.model.decoder(w, c, output_shape=shape).clamp(0, 1) for w, c in zip(wires, confidences)]


class AC16Adapter(CodecAdapter):
    """AC16 on the first 10 frames, upscaled to 256x144."""

    def __init__(self, name: str, path: str | Path, device):
        from .ac16 import AsymmetricContextCodecV4

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model = AsymmetricContextCodecV4(**payload["model_config"])
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(device).eval()
        self.name, self.mode, self.gop, self.height, self.width = name, "AC16", 10, 144, 256

    def gops(self, clip):
        return [resize_clip(clip, self.height, self.width)[:, :, :10]]

    def frame_index(self):
        return list(range(10))

    def encode(self, gops):
        return [self.model.encode_gop(gop) for gop in gops]

    def decode(self, wires, confidences):
        return [self.model.decode_gop(w, c)[0].clamp(0, 1) for w, c in zip(wires, confidences)]


def build_adapter(name: str, device) -> CodecAdapter:
    """Build a named release/research codec, or ``label=path`` for any V8 checkpoint."""
    if "=" in name:
        label, path = name.split("=", 1)
        return AutoencoderAdapter(label, path, "V8", device)
    if name == "v8-face-gan":
        return AutoencoderAdapter(name, "models/v8-hf3k-face-gan.pt", "V8", device)
    if name == "v7-rxfix":
        return AutoencoderAdapter(name, "models/v8-flex8k-ota-rxfix.pt", "V7", device)
    if name == "ac16":
        return AC16Adapter(name, "models/ac16-best-inference.pt", device)
    if name == "ac2k-psnr":
        from .ac2k_psnr import load_psnr_checkpoint

        return StatefulAdapter(name, load_psnr_checkpoint("models/ac2k-psnr-2.2khz-best.pt", device)[0], 4)
    if name == "ac2k-v2-fidelity":
        from .ac2k_psnr import load_ac2k

        return StatefulAdapter(name, load_ac2k("models/ac2k-v2-fidelity-best-inference.pt").to(device), 4)
    if name == "ac6":
        from .ac6 import AC6Codec

        payload = torch.load("models/ac6-best-inference.pt", map_location="cpu", weights_only=False)
        config = payload["model_config"]
        model = AC6Codec(
            spatial_pooling=False,
            width=config.get("model_width", 48),
            features=config.get("features", 24),
            motion_width=config.get("motion_width", 48),
            refine_blocks=config.get("refine_blocks", 2),
            anchor_gate_max=config.get("anchor_gate_max", 0.35),
            anchor_style_max=config.get("anchor_style_max", 0.39),
            deep_tail=config.get("deep_tail", True),
            model_width=config.get("model_width"),
        )
        model.load_state_dict(payload["model_state_dict"])
        return StatefulAdapter(name, model.to(device), 6)
    raise KeyError(f"unknown model {name!r}")


class FaceMasks:
    """YuNet boxes on the source frames, cached per clip."""

    def __init__(self, model_path: str | Path = DEFAULT_FACE_MODEL, threshold: float = 0.72, upscale: int = 4):
        import cv2

        self.cv2, self.upscale = cv2, upscale
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (192 * upscale, 108 * upscale), threshold)
        self._cache: dict[str, torch.Tensor] = {}
        self._boxes: dict[str, torch.Tensor] = {}

    def boxes(self, key: str, clip_uint8: torch.Tensor) -> torch.Tensor:
        """(T, 4) largest face box per frame as x, y, w, h in 192x108 pixels; NaN where no face."""
        self.mask(key, clip_uint8)
        return self._boxes[key]

    def mask(self, key: str, clip_uint8: torch.Tensor) -> torch.Tensor:
        """(T, 108, 192) bool mask for a (3, T, 108, 192) uint8 clip."""
        if key in self._cache:
            return self._cache[key]
        _, frames, height, width = clip_uint8.shape
        out = torch.zeros(frames, height, width, dtype=torch.bool)
        largest = torch.full((frames, 4), float("nan"))
        for t in range(frames):
            rgb = clip_uint8[:, t].permute(1, 2, 0).numpy()
            bgr = self.cv2.resize(rgb[:, :, ::-1], (width * self.upscale, height * self.upscale),
                                  interpolation=self.cv2.INTER_CUBIC)
            _, faces = self.detector.detect(bgr)
            if faces is None:
                continue
            for face in faces:
                x, y, w, h = (float(v) / self.upscale for v in face[:4])
                if math.isnan(float(largest[t, 2])) or w * h > float(largest[t, 2] * largest[t, 3]):
                    largest[t] = torch.tensor([x, y, w, h])
                x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
                x1, y1 = min(width, int(math.ceil(x + w))), min(height, int(math.ceil(y + h)))
                if x1 > x0 and y1 > y0:
                    out[t, y0:y1, x0:x1] = True
        self._cache[key] = out
        self._boxes[key] = largest
        return out


def masked_psnr(reference: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor) -> float | None:
    """PSNR over masked pixels. ``mask`` broadcasts over channels: (T, H, W) against (3, T, H, W)."""
    count = int(mask.sum())
    if count == 0:
        return None
    error = (reference.float() - reconstruction.float()).square().mean(0)
    mse = float(error[mask].mean())
    return 100.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse)


def clip_psnr(gops: list[torch.Tensor], recons: list[torch.Tensor]) -> float:
    """Mean of per-GOP PSNR, the statistic used by every earlier report."""
    values = []
    for gop, recon in zip(gops, recons):
        mse = float((gop.float() - recon.float()).square().mean())
        values.append(100.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse))
    return float(np.mean(values))


def summarize(values: list[float]) -> dict:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else float("nan")
    return {"mean": float(array.mean()), "se": se, "n": int(array.size)}


def paired(candidate: list[float | None], reference: list[float | None]) -> dict:
    """Mean paired difference over clips where both values exist, with its standard error."""
    deltas = [c - r for c, r in zip(candidate, reference) if c is not None and r is not None]
    return summarize(deltas)


@dataclass
class ClipRecord:
    rows: dict[str, float] = field(default_factory=dict)
    face: dict[str, float | None] = field(default_factory=dict)
    lpips: dict[str, float] = field(default_factory=dict)
    ssim: dict[str, float] = field(default_factory=dict)
    nme: dict[str, float | None] = field(default_factory=dict)
    failures: int = 0


@torch.no_grad()
def score_clip(
    adapter: CodecAdapter,
    clip_uint8: torch.Tensor,
    clip_index: int,
    device,
    gains: dict[str, float],
    seed0: int = 2026,
    face_mask: torch.Tensor | None = None,
    face_boxes: torch.Tensor | None = None,
    face_geometry=None,
    lpips_metric: Callable | None = None,
    latent_rows: bool = True,
    keep: bool = False,
) -> dict[str, ClipRecord] | tuple[dict[str, ClipRecord], dict]:
    """Score one clip for every receiver setting in ``gains`` (label -> confidence gain).

    The encode, the AWGN draws and the modem waveforms are shared by all settings,
    so settings differ only in the receiver and their per-clip values are paired.
    """
    clip = clip_uint8.float().div(255).unsqueeze(0).to(device)
    gops = adapter.gops(clip)
    wires = adapter.encode(gops)
    frame_index = adapter.frame_index()
    mask = None
    if face_mask is not None:
        mask = face_mask[frame_index]
        if (adapter.height, adapter.width) != tuple(mask.shape[-2:]):
            mask = F.interpolate(mask[None].float(), size=(adapter.height, adapter.width), mode="nearest")[0].bool()
        mask = mask.to(device)
    boxes = None
    if face_boxes is not None and face_geometry is not None and torch.isfinite(face_boxes).any():
        boxes = face_boxes[frame_index].clone()
        boxes[:, 0::2] *= adapter.width / 192.0
        boxes[:, 1::2] *= adapter.height / 108.0
    received: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    failures = 0
    if latent_rows:
        torch.manual_seed(seed0 * 1000 + clip_index)
        received["clean"] = (wires, [torch.ones_like(w) for w in wires])
        for snr in (15, 6):
            draws = [impair_wire(w, float(snr)) for w in wires]
            received[f"awgn{snr}"] = ([d[0] for d in draws], [d[1] for d in draws])
    for profile, row in (("clean", "modem_clean"), ("mpp12", "mpp12")):
        rx, cf = [], []
        for g, wire in enumerate(wires):
            r, c, ok = modem_exchange(wire.squeeze(0).float().cpu().numpy(), adapter.mode, seed0 + clip_index * len(wires) + g, profile)
            failures += int(not ok)
            rx.append(torch.from_numpy(r).to(device).unsqueeze(0))
            cf.append(torch.from_numpy(c).to(device).unsqueeze(0))
        received[row] = (rx, cf)
    source = torch.cat(gops, 2)[0]
    records: dict[str, ClipRecord] = {}
    kept: dict = {}
    for label, gain in gains.items():
        record = ClipRecord(failures=failures)
        for row, (rx, cf) in received.items():
            recons = adapter.decode(rx, [scale_confidence(c, gain) for c in cf])
            record.rows[row] = clip_psnr(gops, recons)
            if row in ("clean", "mpp12"):
                joined = torch.cat(recons, 2)[0]
                record.ssim[row] = float(np.mean([
                    global_ssim(g[0].permute(1, 0, 2, 3), r[0].permute(1, 0, 2, 3)) for g, r in zip(gops, recons)
                ]))
                if lpips_metric is not None:
                    record.lpips[row] = float(np.mean([
                        float(lpips_metric(r[0].permute(1, 0, 2, 3) * 2 - 1, g[0].permute(1, 0, 2, 3) * 2 - 1).mean())
                        for g, r in zip(gops, recons)
                    ]))
                if mask is not None:
                    record.face[row] = masked_psnr(source, joined, mask)
                if boxes is not None:
                    record.nme[row] = face_geometry.nme(source, joined, boxes)
                if keep:
                    kept[(label, row)] = joined.cpu().half()
        records[label] = record
    if keep:
        kept["source"] = source.cpu().half()
        return records, kept
    return records
