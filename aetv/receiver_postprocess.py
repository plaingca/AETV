"""Receiver-only restoration and frame interpolation experiments for V8.

The codec and wire format are deliberately outside this module.  Inputs and
outputs are RGB tensors in ``B,C,T,H,W`` layout and the channel estimate is an
explicit condition, so post-processing can be ablated without touching V8.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameInterpolator(Protocol):
    """Minimal interface shared by RIFE and deterministic controls."""

    def midpoint(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return the temporal midpoint for two ``B,C,H,W`` frames."""


class LinearFrameInterpolator:
    """Non-learned control; useful for tests and as an honest baseline."""

    def midpoint(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (left + right)


class RIFEInterpolator:
    """Adapter for the official hzwer Practical-RIFE checkout and weights.

    The upstream project is kept external to AETV.  ``repo`` supplies the
    shared ``model/`` helpers and ``weights`` contains ``flownet.pkl`` plus
    the matching ``RIFE_HDv3.py`` and ``IFNet_HDv3.py`` architecture files.
    """

    def __init__(
        self,
        repo: str | Path,
        weights: str | Path,
        *,
        device: str | torch.device = "cuda",
        scale: float = 1.0,
    ) -> None:
        repo = Path(repo).resolve()
        weights = Path(weights).resolve()
        if not (repo / "model" / "warplayer.py").is_file():
            raise FileNotFoundError(f"Practical-RIFE model support files not found under {repo}")
        if not weights.is_dir():
            raise FileNotFoundError(f"RIFE weights directory not found: {weights}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # Downloaded Practical-RIFE archives carry the architecture beside
        # flownet.pkl (normally train_log/RIFE_HDv3.py), while the checkout
        # supplies shared warping/loss helpers under model/.
        weights_parent = str(weights.parent)
        if weights_parent not in sys.path:
            sys.path.insert(0, weights_parent)
        module = importlib.import_module(f"{weights.name}.RIFE_HDv3")
        model = module.Model()
        model.load_model(str(weights), -1)
        model.eval()
        self.device = torch.device(device)
        if hasattr(model, "device"):
            model.device()
        elif hasattr(model, "to"):
            model.to(self.device)
        self.model = model
        self.scale = float(scale)

    @torch.inference_mode()
    def midpoint(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 4:
            raise ValueError("RIFE expects matching B,C,H,W frames")
        left = left.to(self.device)
        right = right.to(self.device)
        height, width = left.shape[-2:]
        # Current Practical-RIFE has five flow scales and upstream pads to
        # 128/scale.  Using 32 (sufficient for older RIFE) breaks v4.25-lite.
        multiple = max(128, int(128 / max(self.scale, 1e-6)))
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        padding = (0, pad_w, 0, pad_h)
        left_pad = F.pad(left, padding)
        right_pad = F.pad(right, padding)
        outputs = []
        # Practical-RIFE accepts batches in recent revisions, but iterating is
        # compatible with older checkpoints too and evaluation batches are tiny.
        for index in range(left_pad.shape[0]):
            output = self.model.inference(
                left_pad[index : index + 1],
                right_pad[index : index + 1],
                scale=self.scale,
            )
            outputs.append(output[..., :height, :width])
        return torch.cat(outputs).clamp(0.0, 1.0)


def interpolate_video(
    video: torch.Tensor,
    interpolator: FrameInterpolator,
    *,
    factor: int = 2,
    scene_cut_threshold: float = 0.35,
) -> torch.Tensor:
    """Increase frame count to ``(T-1)*factor+1`` while retaining originals.

    ``factor`` must be a power of two because RIFE is recursively evaluated at
    midpoints.  At a hard cut, held frames are inserted instead of asking the
    flow model to synthesize a ghosted blend across unrelated shots.
    """
    if video.ndim != 5:
        raise ValueError(f"expected B,C,T,H,W video, got {tuple(video.shape)}")
    if factor < 1 or factor & (factor - 1):
        raise ValueError("interpolation factor must be a positive power of two")
    if factor == 1 or video.shape[2] < 2:
        return video
    levels = int(math.log2(factor))

    def subdivide(left: torch.Tensor, right: torch.Tensor, depth: int) -> list[torch.Tensor]:
        if depth == 0:
            return [left, right]
        midpoint = interpolator.midpoint(left, right)
        first = subdivide(left, midpoint, depth - 1)
        second = subdivide(midpoint, right, depth - 1)
        return first[:-1] + second

    pieces = []
    for index in range(video.shape[2] - 1):
        left = video[:, :, index]
        right = video[:, :, index + 1]
        cut = (left - right).abs().mean(dim=(1, 2, 3)) > scene_cut_threshold
        segment = subdivide(left, right, levels)
        if cut.any():
            for position in range(1, len(segment) - 1):
                held = left if position * 2 <= factor else right
                segment[position] = torch.where(cut[:, None, None, None], held, segment[position])
        pieces.extend(segment[:-1])
    pieces.append(video[:, :, -1])
    return torch.stack(pieces, dim=2)


def _embedding(value: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    if half == 0:
        return value[:, None]
    frequencies = torch.exp(
        torch.arange(half, device=value.device, dtype=value.dtype)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = value[:, None] * frequencies[None]
    result = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(result, (0, width - result.shape[1]))


class _ConditionedBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, condition_width: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.conv1 = nn.Conv3d(input_channels, output_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, output_channels)
        self.film = nn.Linear(condition_width, output_channels * 2)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(value)))
        scale, bias = self.film(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None, None, None]) + bias[:, :, None, None, None]
        hidden = self.norm2(self.conv2(F.silu(hidden)))
        return F.silu(hidden + self.skip(value))


class SNRConditionedDenoiser(nn.Module):
    """Small temporal U-Net predicting diffusion noise on the correction residual."""

    def __init__(self, width: int = 32, condition_width: int = 128) -> None:
        super().__init__()
        self.width = int(width)
        self.condition_width = int(condition_width)
        # noisy correction (3), degraded RGB (3), receiver confidence (1)
        self.input = nn.Conv3d(7, width, 3, padding=1)
        self.condition = nn.Sequential(
            nn.Linear(3 * width, condition_width),
            nn.SiLU(),
            nn.Linear(condition_width, condition_width),
        )
        self.down1 = _ConditionedBlock(width, width, condition_width)
        self.downsample = nn.Conv3d(width, 2 * width, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.middle = _ConditionedBlock(2 * width, 2 * width, condition_width)
        self.up = _ConditionedBlock(3 * width, width, condition_width)
        self.output = nn.Conv3d(width, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        degraded: torch.Tensor,
        diffusion_time: torch.Tensor,
        channel_snr_db: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_residual.shape != degraded.shape or degraded.ndim != 5:
            raise ValueError("noisy residual and degraded video must share B,C,T,H,W shape")
        batch, _, frames, height, width = degraded.shape
        if confidence is None:
            confidence = torch.ones(
                batch, 1, frames, height, width, device=degraded.device, dtype=degraded.dtype
            )
        elif confidence.ndim == 1:
            confidence = confidence[:, None, None, None, None].expand(
                batch, 1, frames, height, width
            )
        else:
            confidence = F.interpolate(
                confidence,
                size=(frames, height, width),
                mode="trilinear",
                align_corners=False,
            )
        t_embed = _embedding(diffusion_time.float().clamp(0, 1) * 1000.0, self.width)
        snr_embed = _embedding(channel_snr_db.float().clamp(-12, 36), self.width)
        confidence_embed = _embedding(confidence.mean(dim=(1, 2, 3, 4)), self.width)
        condition = self.condition(torch.cat((t_embed, snr_embed, confidence_embed), dim=1))
        first = self.down1(self.input(torch.cat((noisy_residual, degraded, confidence), dim=1)), condition)
        middle = self.middle(self.downsample(first), condition)
        upsampled = F.interpolate(middle, size=first.shape[-3:], mode="trilinear", align_corners=False)
        return self.output(self.up(torch.cat((upsampled, first), dim=1), condition))


class SNRConditionedResidualDiffusion(nn.Module):
    """Bounded receiver restoration learned from clean-minus-V8 residuals."""

    def __init__(
        self,
        denoiser: SNRConditionedDenoiser | None = None,
        *,
        timesteps: int = 100,
        max_correction: float = 0.25,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("diffusion needs at least two timesteps")
        self.denoiser = denoiser or SNRConditionedDenoiser()
        self.timesteps = int(timesteps)
        self.max_correction = float(max_correction)
        # Cosine alpha-bar schedule reaches near-pure noise.  A short linear
        # beta schedule leaves substantial signal at its final step and is
        # therefore inconsistent with inference initialized from N(0, I).
        time = torch.linspace(0, 1, timesteps + 1, dtype=torch.float32)
        offset = 0.008
        alpha_bars = torch.cos((time + offset) / (1 + offset) * math.pi / 2).square()
        alpha_bars = (alpha_bars / alpha_bars[0]).clamp_min(1e-5)
        self.register_buffer("alpha_bars", alpha_bars[1:])

    def training_loss(
        self,
        clean: torch.Tensor,
        degraded: torch.Tensor,
        channel_snr_db: torch.Tensor,
        confidence: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        min_snr_gamma: float = 5.0,
    ) -> torch.Tensor:
        batch = clean.shape[0]
        steps = torch.randint(
            self.timesteps, (batch,), device=clean.device, generator=generator
        )
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        # Work in a unit residual domain so N(0, I) is the correct terminal
        # distribution; convert back to RGB correction units after sampling.
        residual = ((clean - degraded) / self.max_correction).clamp(-1.0, 1.0)
        alpha = self.alpha_bars[steps].to(clean.dtype)[:, None, None, None, None]
        noisy = alpha.sqrt() * residual + (1.0 - alpha).sqrt() * noise
        predicted = self.denoiser(
            noisy,
            degraded,
            steps.to(clean.dtype) / (self.timesteps - 1),
            channel_snr_db,
            confidence,
        )
        per_sample = (predicted - noise).square().flatten(1).mean(1)
        snr = self.alpha_bars[steps] / (1.0 - self.alpha_bars[steps]).clamp_min(1e-8)
        weight = snr.clamp(max=min_snr_gamma) / snr.clamp_min(1e-8)
        return (per_sample * weight).mean()

    @torch.inference_mode()
    def restore(
        self,
        degraded: torch.Tensor,
        channel_snr_db: torch.Tensor,
        confidence: torch.Tensor | None = None,
        *,
        steps: int = 12,
        seed: int = 0,
    ) -> torch.Tensor:
        if steps < 1:
            return degraded
        device = degraded.device
        generator = torch.Generator(device=device).manual_seed(seed)
        residual = torch.randn(
            degraded.shape, device=device, dtype=degraded.dtype, generator=generator
        )
        indices = torch.linspace(self.timesteps - 1, 0, steps, device=device).long().unique_consecutive()
        for position, index in enumerate(indices):
            batch_steps = torch.full(
                (degraded.shape[0],), int(index), device=device, dtype=torch.long
            )
            alpha = self.alpha_bars[index].to(degraded.dtype)
            predicted_noise = self.denoiser(
                residual,
                degraded,
                batch_steps.to(degraded.dtype) / (self.timesteps - 1),
                channel_snr_db,
                confidence,
            )
            clean_residual = (residual - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            clean_residual = clean_residual.clamp(-1.0, 1.0)
            if position + 1 == len(indices):
                residual = clean_residual
            else:
                next_alpha = self.alpha_bars[indices[position + 1]].to(degraded.dtype)
                residual = next_alpha.sqrt() * clean_residual + (1.0 - next_alpha).sqrt() * predicted_noise
        return (degraded + self.max_correction * residual).clamp(0.0, 1.0)
