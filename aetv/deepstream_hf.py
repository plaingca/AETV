"""Fresh DeepStream-style JSCC codec for the 2,816-real V8 wire.

Adapted from Chi et al., DeepStream (arXiv:2509.05971), to the existing HF
modem instead of a Wi-Fi OFDM prototype:

- Encoder features are clamped to gamma * [-1, 1] before they are normalized,
  the same bounded activation as their equation (5), so a few outliers cannot
  spend the modem's unit-RMS budget.
- Each feature is repeated onto distinct V8 carriers with alternating signs.
  That is the cross-subcarrier spread of their section III-B in a form the
  modem's own interleaver does not collapse: a faded carrier damages one
  copy, and the receiver Wiener-combines the rest.
- Even carrier slots keep a minus on the quadrature sample, their equation (6),
  so a real/imag pair is not two copies of the same sign.
- Training randomly zeros a suffix of the feature vector (their section III-C)
  so earlier coordinates carry the picture when a fade or a shorter burst
  drops the tail.

The picture contract is 32x18 RGB, two frames per one-second V8 GOP (2 fps).
Nothing here loads or adapts an existing AETV checkpoint.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.fft import dctn, idctn
from torch import nn
from torch.nn import functional as F

from .config import AETV_MODES, DATA_SYMS_PER_FRAME, FRAMES_PER_GOP
from .framing import GOP_DEINTERLEAVER_W
from .hfchannel import emulate
from .modem import demodulate_gop_stream, modulate_gop_stream

GOP_VALUES = 2816
LATENT_CARRIERS = 44
GAMMA = 1.5
HEIGHT = 18
WIDTH = 32
GOP_FRAMES = 2
FEATURES = 352
REPS = 8  # 352 * 8 = 2816
GLOBAL = 32


def unit_rms(wire: torch.Tensor) -> torch.Tensor:
    power = wire.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return wire / power.sqrt().to(dtype=wire.dtype)


def psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    mse = (reference.float() - reconstruction.float()).square().mean().clamp_min(0).item()
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def _carrier_slots() -> list[list[int]]:
    de = np.asarray(GOP_DEINTERLEAVER_W, dtype=np.int32)
    groups: list[list[tuple[int, int]]] = [[] for _ in range(LATENT_CARRIERS)]
    for wire_index, tx_slot in enumerate(de.tolist()):
        carrier = (int(tx_slot) // 2) % LATENT_CARRIERS
        groups[carrier].append((int(tx_slot), wire_index))
    ordered: list[list[int]] = []
    for group in groups:
        group.sort()
        ordered.append([wire_index for _tx, wire_index in group])
        if len(ordered[-1]) != GOP_VALUES // LATENT_CARRIERS:
            raise RuntimeError("each V8 carrier should own 64 real latents")
    return ordered


def build_placement(n_features: int, reps: int) -> tuple[np.ndarray, np.ndarray]:
    """Place each feature on ``reps`` distinct carriers. Signs include the Q flip."""
    if n_features * reps > GOP_VALUES:
        raise ValueError(f"{n_features} x {reps} does not fit in {GOP_VALUES}")
    slots = _carrier_slots()
    cursor = [0] * LATENT_CARRIERS
    index = np.zeros((n_features, reps), dtype=np.int64)
    signs = np.zeros((n_features, reps), dtype=np.float32)
    de = np.asarray(GOP_DEINTERLEAVER_W, dtype=np.int32)
    for feature in range(n_features):
        for rep in range(reps):
            carrier = (feature + rep * 5) % LATENT_CARRIERS
            # 5 is coprime with 44, and 8*5 < 44, so the reps of one feature
            # land on distinct carriers. 352/44 * 8 = 64 fills every slot when
            # n_features is 352 and reps is 8.
            if cursor[carrier] >= len(slots[carrier]):
                raise RuntimeError(f"carrier {carrier} ran out of slots")
            wire_index = slots[carrier][cursor[carrier]]
            cursor[carrier] += 1
            index[feature, rep] = wire_index
            quadrature = -1.0 if int(de[wire_index]) % 2 else 1.0
            repetition = 1.0 if rep % 2 == 0 else -1.0
            signs[feature, rep] = quadrature * repetition
    return index, signs


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DeepStreamHF(nn.Module):
    """32x18, 2-frame codec. The wire is always 2,816 unit-RMS reals."""

    architecture = "deepstream-hf-v1"

    def __init__(self):
        super().__init__()
        if FEATURES * REPS != GOP_VALUES:
            raise RuntimeError("feature repetition does not fill 2816 reals")
        index, signs = build_placement(FEATURES, REPS)
        if len(np.unique(index)) != GOP_VALUES:
            raise RuntimeError("placement must occupy every latent exactly once")
        self.register_buffer("index", torch.from_numpy(index), persistent=False)
        self.register_buffer("signs", torch.from_numpy(signs), persistent=False)
        self.power = nn.Parameter(torch.zeros(FEATURES))
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 48, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(48, 64, kernel_size=(1, 3, 3), stride=(1, 3, 2), padding=(0, 0, 1)),
            nn.GELU(),
            ResBlock(64),
        )
        self.to_features = nn.Linear(64 * GOP_FRAMES * 3 * 8, FEATURES)
        self.from_features = nn.Linear(FEATURES * 2, 64 * GOP_FRAMES * 3 * 8)
        self.decoder = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose3d(64, 48, kernel_size=(1, 3, 4), stride=(1, 3, 2), padding=(0, 0, 1)),
            nn.GELU(),
            nn.ConvTranspose3d(48, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def reset(self) -> None:
        return None

    def _amplitudes(self) -> torch.Tensor:
        weights = F.softplus(self.power) + 0.25
        return weights / weights.mean().clamp_min(1e-4)

    def _mask(self, features: torch.Tensor) -> torch.Tensor:
        """Section III-C: full vector with probability 1/2, else a random suffix is zero."""
        if not self.training or features.shape[0] == 0:
            return features
        kept = []
        for _ in range(features.shape[0]):
            if torch.rand((), device=features.device) < 0.5:
                kept.append(features.new_tensor(FEATURES))
            else:
                kept.append(torch.randint(GLOBAL, FEATURES, (), device=features.device))
        out = features.clone()
        for row, count in enumerate(kept):
            out[row, int(count) :] = 0
        return out

    def _place(self, features: torch.Tensor) -> torch.Tensor:
        weighted = (features * self._amplitudes()).clamp(-GAMMA, GAMMA)
        wire = features.new_zeros(features.shape[0], GOP_VALUES)
        placed = weighted.unsqueeze(-1) * self.signs
        wire.scatter_add_(1, self.index.reshape(-1).expand(features.shape[0], -1), placed.reshape(features.shape[0], -1))
        return unit_rms(wire)

    def _collect(self, received: torch.Tensor, confidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = received[:, self.index] * self.signs
        weights = confidence[:, self.index].clamp(0, 1)
        estimate = (gathered * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-4)
        feature_confidence = weights.mean(dim=-1)
        return estimate, feature_confidence

    def encode_gop(self, video: torch.Tensor, retain_state: bool = True, **kwargs) -> torch.Tensor:
        del retain_state, kwargs
        if video.ndim != 5 or video.shape[1:] != (3, GOP_FRAMES, HEIGHT, WIDTH):
            raise ValueError(f"expected (B, 3, 2, 18, 32), got {tuple(video.shape)}")
        grid = self.encoder(video)
        if grid.shape[-3:] != (GOP_FRAMES, 3, 8):
            raise RuntimeError(f"encoder grid {tuple(grid.shape)} is not (B, 64, 2, 3, 8)")
        features = self.to_features(grid.flatten(1))
        features = self._mask(features)
        return self._place(features)

    def decode_gop(
        self,
        received: torch.Tensor,
        confidence: torch.Tensor | None = None,
        retain_state: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del retain_state, kwargs
        if received.shape[-1] != GOP_VALUES:
            raise ValueError(f"wire length {received.shape[-1]} != {GOP_VALUES}")
        if confidence is None:
            confidence = torch.ones_like(received)
        features, feature_confidence = self._collect(received, confidence)
        hidden = self.from_features(torch.cat([features * feature_confidence, feature_confidence], dim=-1))
        hidden = hidden.reshape(received.shape[0], 64, GOP_FRAMES, 3, 8)
        logits = self.decoder(hidden)
        if logits.shape[-3:] != (GOP_FRAMES, HEIGHT, WIDTH):
            raise RuntimeError(f"decoder image {tuple(logits.shape)} is not 2x18x32")
        video = torch.sigmoid(logits)
        return video, feature_confidence


def apply_hf_fade(
    wire: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equalized two-path fade. Scoring uses the real V8 modem, not this surrogate.

    After a zero-forcing equalizer the latent is ``gain * x + n / |h|``. The
    0.78 gain matches the clean V8 loopback. ``|h|`` is a two-path response at
    the modem's 50 Hz carrier spacing and the mpp profile's 2 ms delay.
    """
    if wire.shape[-1] != GOP_VALUES:
        raise ValueError(f"wire length {wire.shape[-1]} != {GOP_VALUES}")
    batch = wire.shape[0]
    device = wire.device
    g1 = torch.randn(batch, LATENT_CARRIERS, device=device, generator=generator)
    g1 = g1 + 1j * torch.randn(batch, LATENT_CARRIERS, device=device, generator=generator)
    g2 = torch.randn(batch, LATENT_CARRIERS, device=device, generator=generator)
    g2 = g2 + 1j * torch.randn(batch, LATENT_CARRIERS, device=device, generator=generator)
    tones = torch.arange(LATENT_CARRIERS, device=device, dtype=torch.float32)
    phase = torch.exp(-2j * math.pi * tones * 0.1)
    response = (g1 + g2 * phase) / math.sqrt(2.0)
    tone = response.abs()
    tone = tone / tone.mean(dim=1, keepdim=True).clamp_min(1e-4)
    tone = tone.clamp(0.2, 4.0)
    de = torch.as_tensor(GOP_DEINTERLEAVER_W, device=device, dtype=torch.long)
    carrier = (de // 2) % LATENT_CARRIERS
    per_latent = tone[:, carrier]
    noise = torch.randn(wire.shape, device=device, dtype=wire.dtype, generator=generator)
    received = 0.78 * wire + noise * (float(sigma) / per_latent)
    strength = per_latent.square()
    confidence = (strength / (strength + float(sigma) ** 2)).to(dtype=wire.dtype)
    return received, confidence


def _scatter(values: np.ndarray, index: np.ndarray, signs: np.ndarray) -> np.ndarray:
    wire = np.zeros(GOP_VALUES, dtype=np.float32)
    flat_values = (values.reshape(-1, 1) * signs).reshape(-1)
    np.add.at(wire, index.reshape(-1), flat_values)
    return wire


def _combine(received: np.ndarray, confidence: np.ndarray, index: np.ndarray, signs: np.ndarray) -> np.ndarray:
    gathered = received[index] * signs
    weights = np.clip(confidence[index], 0.0, 1.0)
    return (gathered * weights).sum(axis=-1) / np.maximum(weights.sum(axis=-1), 1e-4)


# Fixed candidates. Training freezes one after a holdout sweep; eval must not retune on the val set.
REFERENCE_CANDIDATES: tuple[dict, ...] = (
    {"kind": "pixels", "gh": 6, "gw": 8, "reps": 8},
    {"kind": "pixels", "gh": 8, "gw": 8, "reps": 7},
    {"kind": "pixels", "gh": 9, "gw": 8, "reps": 6},
    {"kind": "pixels", "gh": 6, "gw": 10, "reps": 7},
    {"kind": "dct", "kh": 4, "kw": 6, "reps": 12},
    {"kind": "dct", "kh": 6, "kw": 8, "reps": 8},
    {"kind": "dct", "kh": 4, "kw": 8, "reps": 12},
    {"kind": "dct", "kh": 8, "kw": 10, "reps": 5},
)
PILOT_REPS = 96


def _payload_count(config: dict) -> int:
    if config["kind"] == "pixels":
        return GOP_FRAMES * 3 * int(config["gh"]) * int(config["gw"])
    return GOP_FRAMES * 3 * int(config["kh"]) * int(config["kw"])


def reference_fits(config: dict) -> bool:
    count = _payload_count(config)
    return count * int(config["reps"]) + PILOT_REPS <= GOP_VALUES


class DownsampleReference:
    """Non-learned reference at the same 32x18, 2-frame contract.

    Pixels: area-downsample, repeat across carriers, bilinear upsample.
    DCT: keep the low-frequency rectangle, with scales frozen from training
    clips, then inverse DCT. A repeated constant pilot removes the modem gain
    without using the transmitted RMS as side information.
    """

    def __init__(self, config: dict, scales: np.ndarray | None = None):
        if not reference_fits(config):
            raise ValueError(f"reference {config} does not fit in {GOP_VALUES}")
        self.config = dict(config)
        self.count = _payload_count(config)
        self.reps = int(config["reps"])
        payload_index, payload_signs, pilot_index, pilot_signs = _place_payload_and_pilot(
            self.count, self.reps, PILOT_REPS
        )
        self.payload_index = payload_index
        self.payload_signs = payload_signs
        self.pilot_index = pilot_index
        self.pilot_signs = pilot_signs
        if scales is None:
            scales = np.ones((3, HEIGHT, WIDTH), dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)

    def transmit(self, video: np.ndarray) -> np.ndarray:
        """``video`` is (2, 3, 18, 32) in 0..1. Returns a unit-RMS wire."""
        values = self._analyze(video)
        wire = _scatter(values, self.payload_index, self.payload_signs)
        pilot = _scatter(np.ones(1, dtype=np.float32), self.pilot_index, self.pilot_signs)
        wire = wire + pilot
        rms = float(np.sqrt(np.mean(wire**2) + 1e-12))
        return (wire / rms).astype(np.float32)

    def receive(self, received: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        payload = _combine(received, confidence, self.payload_index, self.payload_signs)
        pilot = _combine(received, confidence, self.pilot_index, self.pilot_signs)
        scale = float(pilot[0])
        if not np.isfinite(scale) or abs(scale) < 1e-3:
            scale = 1e-3 if scale >= 0 else -1e-3
        values = payload / scale
        return self._synthesize(values)

    def _analyze(self, video: np.ndarray) -> np.ndarray:
        if self.config["kind"] == "pixels":
            tensor = torch.from_numpy(np.ascontiguousarray(video))
            small = F.interpolate(tensor, size=(int(self.config["gh"]), int(self.config["gw"])), mode="area")
            return small.numpy().reshape(-1).astype(np.float32)
        kh, kw = int(self.config["kh"]), int(self.config["kw"])
        pieces = []
        for frame in range(GOP_FRAMES):
            for channel in range(3):
                coeff = dctn(video[frame, channel], norm="ortho")
                band = coeff[:kh, :kw] / self.scales[channel, :kh, :kw]
                pieces.append(band.reshape(-1))
        return np.concatenate(pieces).astype(np.float32)

    def _synthesize(self, values: np.ndarray) -> np.ndarray:
        if self.config["kind"] == "pixels":
            gh, gw = int(self.config["gh"]), int(self.config["gw"])
            small = torch.from_numpy(values.reshape(GOP_FRAMES, 3, gh, gw).copy())
            up = F.interpolate(small, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
            return up.numpy().astype(np.float32)
        kh, kw = int(self.config["kh"]), int(self.config["kw"])
        cursor = 0
        frames = np.zeros((GOP_FRAMES, 3, HEIGHT, WIDTH), dtype=np.float32)
        band_size = kh * kw
        for frame in range(GOP_FRAMES):
            for channel in range(3):
                band = values[cursor : cursor + band_size].reshape(kh, kw)
                cursor += band_size
                coeff = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                coeff[:kh, :kw] = band * self.scales[channel, :kh, :kw]
                frames[frame, channel] = idctn(coeff, norm="ortho").astype(np.float32)
        return frames


def _place_payload_and_pilot(
    count: int, reps: int, pilot_reps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Payload is (count, reps). The pilot is one known amplitude repeated ``pilot_reps`` times."""
    slots = _carrier_slots()
    cursor = [0] * LATENT_CARRIERS
    de = np.asarray(GOP_DEINTERLEAVER_W, dtype=np.int32)
    payload_index = np.zeros((count, reps), dtype=np.int64)
    payload_signs = np.zeros((count, reps), dtype=np.float32)
    pilot_index = np.zeros((1, pilot_reps), dtype=np.int64)
    pilot_signs = np.zeros((1, pilot_reps), dtype=np.float32)

    def take(feature: int, rep: int) -> tuple[int, float]:
        carrier = (feature * 3 + rep * 5) % LATENT_CARRIERS
        found = None
        for shift in range(LATENT_CARRIERS):
            candidate = (carrier + shift) % LATENT_CARRIERS
            if cursor[candidate] < len(slots[candidate]):
                found = candidate
                break
        if found is None:
            raise RuntimeError("reference placement exhausted the wire")
        wire_index = slots[found][cursor[found]]
        cursor[found] += 1
        quadrature = -1.0 if int(de[wire_index]) % 2 else 1.0
        repetition = 1.0 if rep % 2 == 0 else -1.0
        return wire_index, quadrature * repetition

    for feature in range(count):
        for rep in range(reps):
            wire_index, sign = take(feature, rep)
            payload_index[feature, rep] = wire_index
            payload_signs[feature, rep] = sign
    for rep in range(pilot_reps):
        wire_index, sign = take(count, rep)
        pilot_index[0, rep] = wire_index
        pilot_signs[0, rep] = sign
    occupied = np.concatenate([payload_index.reshape(-1), pilot_index.reshape(-1)])
    if len(np.unique(occupied)) != occupied.size:
        raise RuntimeError("reference placement reused a latent")
    return payload_index, payload_signs, pilot_index, pilot_signs


def dct_scales(videos: np.ndarray) -> np.ndarray:
    """Per-bin std of orthonormal DCT coefficients. ``videos`` is (N, 2, 3, 18, 32)."""
    accum = []
    for video in videos:
        coeff = np.stack(
            [dctn(video[frame, channel], norm="ortho") for frame in range(GOP_FRAMES) for channel in range(3)],
            axis=0,
        )
        accum.append(coeff.reshape(GOP_FRAMES, 3, HEIGHT, WIDTH))
    stack = np.stack(accum, axis=0)
    scale = stack.std(axis=(0, 1)).astype(np.float32)
    return np.maximum(scale, 0.02)


def load_deepstream_hf(path: str, device: torch.device | str = "cpu") -> tuple[DeepStreamHF, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    architecture = payload.get("architecture") if isinstance(payload, dict) else None
    if architecture != DeepStreamHF.architecture:
        raise RuntimeError(f"refusing to load architecture {architecture!r}; expected a fresh deepstream-hf-v1 checkpoint")
    model = DeepStreamHF()
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, payload


if FRAMES_PER_GOP * DATA_SYMS_PER_FRAME * LATENT_CARRIERS * 2 != GOP_VALUES:
    raise RuntimeError("V8 latent budget is no longer 2816 reals")


def v8_exchange(wire: np.ndarray, seed: int, profile: str) -> tuple[np.ndarray, np.ndarray]:
    """One GOP through the V8 modem. ``profile`` is ``clean`` or a channel key such as ``mpp12``."""
    mode = AETV_MODES["V8"]
    audio = modulate_gop_stream([np.ascontiguousarray(wire, dtype=np.float32)], mode_name="V8", callsign="EVAL")
    impaired = audio if profile == "clean" else emulate(audio, profile, seed=seed, fs=mode.geometry.fs)
    demod = demodulate_gop_stream(impaired, band=mode.band, drift_track="off")
    if not demod.gops_latents:
        zeros = np.zeros(GOP_VALUES, dtype=np.float32)
        return zeros, zeros
    received = np.asarray(demod.gops_latents[0], dtype=np.float32)
    confidence = np.clip(np.asarray(demod.gops_weights[0], dtype=np.float32), 0.0, 1.0)
    return received, confidence
