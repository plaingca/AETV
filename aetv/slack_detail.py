"""Common-mode detail carried in AC2K's unused analog precision.

On this cache, 8-bit quantization of the 2,816-coordinate AC2K wire changes
clean PSNR by about 0.001 dB, and 6-bit quantization costs about 0.01 dB.
The reconstruction error that remains is mostly a temporally common high-
frequency residual: adding the true temporal-mean residual back is worth
several decibels, but a low-resolution code of it is not. A 4x4 patch PCA
with two components (2,592 reals) is small enough to hide inside the
quantization bins.

The encoder writes that code as a sub-bin dither on top of the quantized
AC2K wire. On a noiseless wire the decoder separates the bin center (AC2K)
from the dither (detail) exactly. Under AWGN the dither sits below the noise,
so a confidence value selects the frozen AC2K reconstruction and the detail
is not applied. The AC2K checkpoint is read and never rewritten.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .ac2k_psnr import load_psnr_checkpoint

GOP_VALUES = 2816
SPAN = 4.0
BITS = 8
LEVELS = 2**BITS
BIN = (2 * SPAN) / (LEVELS - 1)
LIMIT = 0.45 * (BIN / 2)
PATCH = 4
GRID_H = 108 // PATCH
GRID_W = 192 // PATCH
PATCHES = GRID_H * GRID_W
COMPONENTS = 2
CODE = COMPONENTS * PATCHES
GAIN_INDEX = CODE
PEAK_MAX = 8.0
CONFIDENCE_GATE = 0.999


def quantize_wire(wire: torch.Tensor) -> torch.Tensor:
    """Mid-riser quantizer on ``[-SPAN, SPAN]``. Interior bins are ``BIN`` wide."""
    clipped = wire.clamp(-SPAN, SPAN)
    level = torch.round((clipped + SPAN) / (2 * SPAN) * (LEVELS - 1))
    return level / (LEVELS - 1) * (2 * SPAN) - SPAN


def _patches(image: torch.Tensor) -> torch.Tensor:
    folded = image.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
    batch = image.shape[0]
    return folded.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(batch, PATCHES, 3 * PATCH * PATCH)


def project_patches(image: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    coeff = (_patches(image) - mean) @ basis
    return coeff.reshape(image.shape[0], CODE)


def synthesize_patches(coeff: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    batch = coeff.shape[0]
    flat = coeff.reshape(batch, PATCHES, COMPONENTS) @ basis.T + mean
    image = flat.reshape(batch, GRID_H, GRID_W, 3, PATCH, PATCH)
    return image.permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, 108, 192)


def _encode_peak(peak: torch.Tensor) -> torch.Tensor:
    return (peak / PEAK_MAX).clamp(0, 1) * (2 * LIMIT) - LIMIT


def _decode_peak(gain: torch.Tensor) -> torch.Tensor:
    unit = ((gain + LIMIT) / (2 * LIMIT)).clamp(0, 1)
    return unit * PEAK_MAX


class SlackDetail(nn.Module):
    architecture = "slack-detail-v1"

    def __init__(
        self,
        ac2k_path: str = "models/ac2k-psnr-2.2khz-best.pt",
        basis: torch.Tensor | None = None,
        patch_mean: torch.Tensor | None = None,
    ):
        super().__init__()
        if basis is None:
            basis = torch.zeros(3 * PATCH * PATCH, COMPONENTS)
            basis[0, 0] = 1
            basis[1, 1] = 1
        if patch_mean is None:
            patch_mean = torch.zeros(1, 3 * PATCH * PATCH)
        if tuple(basis.shape) != (3 * PATCH * PATCH, COMPONENTS):
            raise ValueError(f"basis shape {tuple(basis.shape)} is not {(3 * PATCH * PATCH, COMPONENTS)}")
        self.register_buffer("basis", basis.float().contiguous())
        self.register_buffer("patch_mean", patch_mean.float().reshape(1, -1).contiguous())
        self.ac2k_path = str(ac2k_path)
        self.base, _ = load_psnr_checkpoint(ac2k_path, "cpu")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def reset(self) -> None:
        self.base.reset()

    def _use_detail(self, confidence: torch.Tensor | None) -> bool:
        if confidence is None:
            return True
        return bool(float(confidence.detach().mean()) >= CONFIDENCE_GATE)

    def _write(self, raw: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        coeff = project_patches(residual, self.basis, self.patch_mean)
        coeff = coeff.clamp(-PEAK_MAX, PEAK_MAX)
        peak = coeff.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        sent = coeff * (LIMIT / peak)
        quantized = quantize_wire(raw)
        wire = quantized.clone()
        wire[:, :CODE] = quantized[:, :CODE] + sent
        wire[:, GAIN_INDEX] = quantized[:, GAIN_INDEX] + _encode_peak(peak)
        return wire

    def _read(self, wire: torch.Tensor) -> torch.Tensor:
        quantized = quantize_wire(wire)
        peak = _decode_peak(wire[:, GAIN_INDEX] - quantized[:, GAIN_INDEX]).clamp_min(1e-6)
        coeff = (wire[:, :CODE] - quantized[:, :CODE]) * (peak / LIMIT)
        return synthesize_patches(coeff, self.basis, self.patch_mean)

    def encode_gop(self, video: torch.Tensor, retain_state: bool = True, **kwargs) -> torch.Tensor:
        del kwargs
        if video.ndim != 5 or video.shape[1:] != (3, 4, 108, 192):
            raise ValueError(f"expected (B, 3, 4, 108, 192), got {tuple(video.shape)}")
        raw = self.base.encode_gop(video, retain_state=retain_state)
        quantized = quantize_wire(raw)
        recon, _ = self.base.decode_gop(quantized, retain_state=retain_state)
        residual = (video - recon).mean(dim=2)
        return self._write(raw, residual)

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
        if not self._use_detail(confidence):
            return self.base.decode_gop(wire, confidence=confidence, retain_state=retain_state)
        quantized = quantize_wire(wire)
        recon, outage = self.base.decode_gop(quantized, retain_state=retain_state)
        detail = self._read(wire).unsqueeze(2)
        return (recon + detail).clamp(0, 1), outage


def fit_patch_basis(model: SlackDetail, clips: torch.Tensor, device: torch.device, limit: int) -> None:
    """Replace ``basis`` and ``patch_mean`` with the top patch principal components.

    ``clips`` are uint8 training clips. Validation clips must not be passed here.
    """
    dim = 3 * PATCH * PATCH
    total = torch.zeros(dim, device=device)
    outer = torch.zeros(dim, dim, device=device)
    count = 0
    was = model.training
    model.eval()
    with torch.no_grad():
        for index in range(min(limit, clips.shape[0])):
            clip = clips[index].float().div(255.0).unsqueeze(0).to(device)
            model.reset()
            quantized = []
            gops = []
            for start in (0, 4, 8):
                gop = clip[:, :, start : start + 4]
                raw = model.base.encode_gop(gop, retain_state=True)
                quantized.append(quantize_wire(raw))
                gops.append(gop)
            model.reset()
            for gop, wire in zip(gops, quantized):
                recon, _ = model.base.decode_gop(wire, retain_state=True)
                residual = (gop - recon).mean(dim=2)
                patches = _patches(residual).reshape(-1, dim)
                total = total + patches.sum(dim=0)
                outer = outer + patches.T @ patches
                count += patches.shape[0]
    mean = total / count
    covariance = (outer - count * torch.outer(mean, mean)) / max(count - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    _values, vectors = torch.linalg.eigh(covariance)
    basis = torch.stack([vectors[:, -1], vectors[:, -2]], dim=1).contiguous()
    model.basis.copy_(basis)
    model.patch_mean.copy_(mean.reshape(1, -1))
    if was:
        model.train()


def load_slack_detail(
    path: str | Path, device: torch.device | str = "cpu", ac2k_path: str | None = None
) -> tuple[SlackDetail, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SlackDetail(
        ac2k_path=ac2k_path or payload.get("ac2k_checkpoint", "models/ac2k-psnr-2.2khz-best.pt"),
        basis=payload["basis"],
        patch_mean=payload["patch_mean"],
    )
    model.to(device).eval()
    return model, payload
