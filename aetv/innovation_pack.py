"""Conditional innovation packed into AC2K's unused 2.2 kHz coordinates.

AC2K already spends the whole 2,816-coordinate GOP. On this cache the weakest
384 P-frame coordinates (128 in each P-frame) change clean PSNR by about 0.13 dB
when they are zeroed, so they are spare channel uses. This model leaves the
AC2K PSNR fine-tune frozen and writes a second analog code into those slots.

The second code is a conditional innovation in the sense of DCVC: the encoder
sees the source and the frozen reconstruction, and the decoder sees only the
received innovation plus that same reconstruction. Power in the spare slots is
set so the full wire stays exactly unit-RMS and the kept AC2K coordinates are
unchanged. On a clean wire the decoder's unit-RMS restore recovers the
innovation; under AWGN the decoder also sees the Wiener confidence.

The base checkpoint is read, never rewritten.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .ac2k_psnr import load_psnr_checkpoint
from .narrowband_jscc import unit_rms

GOP_VALUES = 2816
P_SLICES = ((1280, 1792), (1792, 2304), (2304, 2816))
WEAK_PER_SLICE = 128
INDEX_PATH = Path(__file__).with_name("innovation_weak_index.pt")


def embed_innovation(raw: torch.Tensor, code: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Write ``code`` into ``index`` without changing the other coordinates.

    The inserted vector is scaled so the full wire has unit RMS. Its direction
    is ``unit_rms(code)``, which the decoder recovers on a clean channel.
    """
    if code.shape[-1] != index.numel():
        raise ValueError(f"code length {code.shape[-1]} != index {index.numel()}")
    mask = torch.ones(raw.shape[-1], device=raw.device, dtype=raw.dtype)
    mask[index] = 0
    kept = raw.detach() * mask
    budget = (float(raw.shape[-1]) - kept.square().sum(dim=-1, keepdim=True)).clamp_min(1e-6)
    direction = unit_rms(code)
    # direction has mean square 1, so scale by sqrt(budget / K) to spend exactly ``budget``.
    values = direction * (budget / code.shape[-1]).sqrt().to(dtype=direction.dtype)
    scattered = raw.detach().new_zeros(raw.shape[0], index.numel())
    scattered = scattered + values
    out = kept.clone()
    return out.index_copy(1, index.to(out.device), scattered)


def compute_weak_index(clips: torch.Tensor, device: torch.device, per_slice: int = WEAK_PER_SLICE) -> torch.Tensor:
    """Lowest-energy P-frame coordinates of the frozen AC2K wire, train clips only."""
    base, _ = load_psnr_checkpoint("models/ac2k-psnr-2.2khz-best.pt", device)
    base.eval()
    wires = []
    with torch.no_grad():
        for index in range(clips.shape[0]):
            video = clips[index].float().div(255.0).unsqueeze(0).to(device)
            base.reset()
            wires.append(base.encode_gop(video[:, :, :4]).squeeze(0).float().cpu())
    energy = torch.stack(wires).square().mean(0)
    chosen = []
    for start, stop in P_SLICES:
        order = torch.argsort(energy[start:stop])
        chosen.append(order[:per_slice] + start)
    return torch.cat(chosen).to(torch.int64).contiguous()


class InnovationPacker(nn.Module):
    """Map a GOP and its frozen reconstruction to K innovation coordinates and back."""

    def __init__(self, width: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 8, 1),
        )
        flat = 4 * 8 * 14 * 24
        self.to_code = nn.Linear(flat, WEAK_PER_SLICE * len(P_SLICES))
        self.from_code = nn.Linear(WEAK_PER_SLICE * len(P_SLICES), flat)
        self.upsample = nn.Sequential(
            nn.Conv2d(8, width, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(108, 192), mode="bilinear", align_corners=False),
            nn.Conv2d(width // 2, 16, 3, padding=1),
            nn.GELU(),
        )
        self.correct = nn.Conv2d(16 + 3 + 1, 3, 3, padding=1)
        nn.init.zeros_(self.correct.weight)
        nn.init.zeros_(self.correct.bias)

    def encode(self, video: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        batch = video.shape[0]
        frames = torch.cat([video, base], dim=1).permute(0, 2, 1, 3, 4).reshape(batch * 4, 6, 108, 192)
        feat = self.encoder(frames)
        if feat.shape[-2:] != (14, 24):
            raise RuntimeError(f"innovation grid {tuple(feat.shape[-2:])} is not 14x24")
        return self.to_code(feat.reshape(batch, -1))

    def decode(self, code: torch.Tensor, base: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        batch = base.shape[0]
        feat = self.from_code(code).reshape(batch * 4, 8, 14, 24)
        spatial = self.upsample(feat)
        rgb = base.permute(0, 2, 1, 3, 4).reshape(batch * 4, 3, 108, 192)
        level = confidence.reshape(batch, 1).repeat_interleave(4, dim=0).reshape(batch * 4, 1, 1, 1)
        level = level.expand(-1, -1, 108, 192)
        delta = self.correct(torch.cat([spatial, rgb, level], dim=1))
        return delta.reshape(batch, 4, 3, 108, 192).permute(0, 2, 1, 3, 4)


class InnovationPack(nn.Module):
    architecture = "innovation-pack-v1"

    def __init__(self, ac2k_path: str = "models/ac2k-psnr-2.2khz-best.pt", index: torch.Tensor | None = None):
        super().__init__()
        if index is None:
            index = torch.load(INDEX_PATH, map_location="cpu", weights_only=False)
        index = index.to(torch.int64).flatten().contiguous()
        if index.numel() != WEAK_PER_SLICE * len(P_SLICES):
            raise ValueError(f"expected {WEAK_PER_SLICE * len(P_SLICES)} spare coordinates, got {index.numel()}")
        self.register_buffer("index", index)
        self.ac2k_path = str(ac2k_path)
        self.base, _ = load_psnr_checkpoint(ac2k_path, "cpu")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.pack = InnovationPacker()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def reset(self) -> None:
        self.base.reset()

    def _confidence(self, confidence: torch.Tensor | None, batch: int, device, dtype) -> torch.Tensor:
        if confidence is None:
            return torch.ones(batch, device=device, dtype=dtype)
        return confidence[:, self.index].mean(dim=1).to(dtype=dtype)

    def _base_recon(
        self, wire: torch.Tensor, confidence: torch.Tensor | None, retain_state: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        damaged = wire.clone()
        damaged[:, self.index] = 0
        with torch.no_grad():
            recon, outage = self.base.decode_gop(damaged, confidence=confidence, retain_state=retain_state)
        return recon, outage

    def encode_gop(self, video: torch.Tensor, retain_state: bool = True, **kwargs) -> torch.Tensor:
        del kwargs
        if video.ndim != 5 or video.shape[1:] != (3, 4, 108, 192):
            raise ValueError(f"expected (B, 3, 4, 108, 192), got {tuple(video.shape)}")
        with torch.no_grad():
            raw = self.base.encode_gop(video, retain_state=retain_state)
            base, _ = self._base_recon(raw, None, retain_state=retain_state)
        code = self.pack.encode(video, base)
        return embed_innovation(raw, code, self.index)

    def decode_gop(
        self,
        wire: torch.Tensor,
        confidence: torch.Tensor | None = None,
        retain_state: bool = True,
        **kwargs,
    ):
        del kwargs
        if wire.shape[-1] != GOP_VALUES:
            raise ValueError(f"wire length {wire.shape[-1]} != {GOP_VALUES}")
        base, outage = self._base_recon(wire, confidence, retain_state)
        innov = unit_rms(wire[:, self.index])
        level = self._confidence(confidence, wire.shape[0], wire.device, base.dtype)
        delta = self.pack.decode(innov, base, level)
        return (base + delta).clamp(0, 1), outage


def load_innovation(
    path: str | Path, device: torch.device | str = "cpu", ac2k_path: str | None = None
) -> tuple[InnovationPack, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    index = payload.get("index")
    model = InnovationPack(
        ac2k_path=ac2k_path or payload.get("ac2k_checkpoint", "models/ac2k-psnr-2.2khz-best.pt"),
        index=index,
    )
    model.pack.load_state_dict(payload["pack"])
    model.to(device).eval()
    return model, payload
