"""Resettable receiver-side GOP continuity correction.

The corrector is deliberately separate from the released codec.  It consumes
the previous decoded GOP, adds no RF values, and can be bypassed exactly on
startup, loss, reacquisition, or an explicit reset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        return value + self.conv2(F.silu(self.norm2(hidden)))


class StatefulGOPCorrector(nn.Module):
    """Small learned correction with a bounded, confidence-gated memory."""

    checkpoint_kind = "aetv-stateful-gop-corrector"

    def __init__(
        self,
        width: int = 24,
        blocks: int = 3,
        spatial_scale: int = 4,
        max_residual: float = 0.12,
        context_mode: str = "last",
        taper_floor: float = 0.0,
    ):
        super().__init__()
        if context_mode not in {"last", "full"}:
            raise ValueError(f"unknown context mode {context_mode!r}")
        if not 0.0 <= taper_floor <= 1.0:
            raise ValueError("taper_floor must be between zero and one")
        self.width = width
        self.blocks = blocks
        self.spatial_scale = spatial_scale
        self.max_residual = max_residual
        self.context_mode = context_mode
        self.taper_floor = taper_floor
        input_channels = 9 if context_mode == "last" else 12
        self.input = nn.Conv3d(input_channels, width, 3, padding=1)
        self.body = nn.Sequential(*(_ResidualBlock(width) for _ in range(blocks)))
        self.output = nn.Conv3d(width, 3, 3, padding=1)

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor | None,
        confidence: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if previous is None:
            return current
        if current.ndim != 5 or previous.ndim != 5 or current.shape != previous.shape:
            raise ValueError("current and previous GOPs must have equal BCTHW shapes")
        batch, _, frames, height, width = current.shape
        previous_frame = previous[:, :, -1:].expand(-1, -1, frames, -1, -1)
        values = [current, previous_frame, current - previous_frame]
        if self.context_mode == "full":
            values.append(previous)
        features = torch.cat(values, dim=1)
        low_height = max(1, math.ceil(height / self.spatial_scale))
        low_width = max(1, math.ceil(width / self.spatial_scale))
        features = F.interpolate(features, (frames, low_height, low_width), mode="trilinear", align_corners=False)
        residual = self.output(self.body(F.silu(self.input(features))))
        residual = F.interpolate(residual, (frames, height, width), mode="trilinear", align_corners=False)
        taper = torch.linspace(1.0, self.taper_floor, frames, device=current.device, dtype=current.dtype)
        gate = current.new_ones((batch, 1, 1, 1, 1))
        if confidence is not None:
            gate = torch.as_tensor(confidence, device=current.device, dtype=current.dtype).reshape(batch, 1, 1, 1, 1).clamp(0, 1)
        return (current + self.max_residual * torch.tanh(residual) * taper.view(1, 1, frames, 1, 1) * gate).clamp(0, 1)


def load_stateful_gop_corrector(path: str, device: torch.device) -> tuple[StatefulGOPCorrector, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != StatefulGOPCorrector.checkpoint_kind:
        raise ValueError(f"{path} is not a {StatefulGOPCorrector.checkpoint_kind} checkpoint")
    corrector = StatefulGOPCorrector(**payload["adapter_config"]).to(device).eval()
    corrector.load_state_dict(payload["adapter_state_dict"], strict=True)
    return corrector, payload
