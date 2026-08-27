"""Rate-compatible overlapping-GOP AETV models.

Each source GOP is still encoded to exactly one on-air latent vector.  The
receiver jointly decodes an odd GOP window and emits its centered GOP block.
Consecutive blocks overlap in latent context without adding symbols to the
waveform or carrying an unbounded recurrent state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import AETV_MODES, AETVModeSpec
from .video_backbone import VideoDecoder, VideoEncoder


def split_video_gops(video: torch.Tensor, frames_per_gop: int) -> torch.Tensor:
    """Convert BCTHW video to BGCTHW GOPs without changing frame order."""
    if video.ndim != 5:
        raise ValueError(f"expected BCTHW video, got {tuple(video.shape)}")
    batch, channels, frames, height, width = video.shape
    if frames % frames_per_gop:
        raise ValueError(
            f"{frames} frames is not divisible by {frames_per_gop} frames/GOP"
        )
    count = frames // frames_per_gop
    return video.reshape(
        batch, channels, count, frames_per_gop, height, width
    ).permute(0, 2, 1, 3, 4, 5)


def join_video_gops(gops: torch.Tensor) -> torch.Tensor:
    """Convert BGCTHW GOPs to BCTHW video without changing frame order."""
    if gops.ndim != 6:
        raise ValueError(f"expected BGCTHW GOPs, got {tuple(gops.shape)}")
    batch, count, channels, frames, height, width = gops.shape
    return gops.permute(0, 2, 1, 3, 4, 5).reshape(
        batch, channels, count * frames, height, width
    )


class OverlappingGOPDecoder(nn.Module):
    """Jointly decode an odd latent window and return its centered GOP block."""

    def __init__(
        self,
        mode: AETVModeSpec | str = "V8",
        width: int = 192,
        latent_channels: int = 3,
        window_gops: int = 3,
        emit_gops: int = 1,
        synthesis_halo_frames: int = 2,
        resize_conv_upsampling: bool = True,
        group_norm: bool = True,
        smooth_temporal_skip: bool = True,
        bilinear_upsampling: bool = True,
    ):
        super().__init__()
        self.mode = AETV_MODES[mode] if isinstance(mode, str) else mode
        if window_gops < 3 or window_gops % 2 == 0:
            raise ValueError("window_gops must be an odd integer of at least three")
        if emit_gops < 1 or emit_gops > window_gops - 2:
            raise ValueError("emit_gops must leave at least one context GOP per side")
        if (window_gops - emit_gops) % 2:
            raise ValueError("window_gops - emit_gops must be even")
        self.width = width
        self.latent_channels = latent_channels
        self.window_gops = window_gops
        self.emit_gops = emit_gops
        if synthesis_halo_frames < 0:
            raise ValueError("synthesis_halo_frames must be non-negative")
        self.synthesis_halo_frames = synthesis_halo_frames
        self.latent_budget = self.mode.latents_per_gop
        self.latent_frames = math.ceil(self.mode.gop_frames / 2)
        self.grid_height = max(1, self.mode.height // 8)
        self.grid_width = max(1, self.mode.width // 8)
        self.grid_elements = (
            latent_channels * self.latent_frames * self.grid_height * self.grid_width
        )

        # Unlike the released V8 decoder, which ignores its final eight values,
        # this learned unpacker consumes every transmitted coordinate.  Masking
        # occurs before projection, so coordinate erasures remain visible.
        self.latent_unpack = nn.Linear(
            self.latent_budget, self.grid_elements, bias=False
        )
        self.decoder = VideoDecoder(
            width=width,
            latent_channels=latent_channels,
            compact=False,
            resize_conv_upsampling=resize_conv_upsampling,
            causal=False,
            group_norm=group_norm,
            smooth_temporal_skip=smooth_temporal_skip,
            bilinear_upsampling=bilinear_upsampling,
            deep_tail=width >= 64,
            deeper=width >= 64,
            deepest=width >= 64,
            deep4=(width >= 64 and self.mode.width * self.mode.height <= 96 * 72),
        )

    @property
    def lookahead_gops(self) -> int:
        return (self.window_gops - self.emit_gops) // 2

    def _window_grids(
        self, latents: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 3:
            raise ValueError(f"expected BGL latent window, got {tuple(latents.shape)}")
        if latents.shape != weights.shape:
            raise ValueError("latents and weights must have identical BGL shapes")
        batch, count, budget = latents.shape
        if count != self.window_gops or budget != self.latent_budget:
            raise ValueError(
                f"expected {(batch, self.window_gops, self.latent_budget)}, "
                f"got {tuple(latents.shape)}"
            )
        masked = latents * weights
        grids = self.latent_unpack(masked).reshape(
            batch,
            count,
            self.latent_channels,
            self.latent_frames,
            self.grid_height,
            self.grid_width,
        )
        grids = grids.permute(0, 2, 1, 3, 4, 5).reshape(
            batch,
            self.latent_channels,
            count * self.latent_frames,
            self.grid_height,
            self.grid_width,
        )
        # Preserve a confidence input for the decoder while letting the learned
        # unpacker express the coordinate-specific corruption in the grid.
        confidence = weights.float().mean(dim=-1).to(latents.dtype)
        confidence = confidence[:, None, :, None, None, None].expand(
            -1,
            self.latent_channels,
            -1,
            self.latent_frames,
            self.grid_height,
            self.grid_width,
        ).reshape_as(grids)
        return grids, confidence

    def forward(
        self, latents: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if weights is None:
            weights = torch.ones_like(latents)
        grids, confidence = self._window_grids(latents, weights)
        decoded = self._decode_center(grids, confidence)
        return decoded

    def _decode_center(
        self, grids: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        """Keep full temporal context cheaply, synthesize only center + halo."""
        inner = self.decoder
        window_frames = self.window_gops * self.mode.gop_frames
        height, width = self.mode.height, self.mode.width
        first = self.lookahead_gops * self.mode.gop_frames
        if inner.smooth_temporal_skip:
            temporal_skip = F.interpolate(
                inner.temporal_skip(grids * confidence),
                size=(window_frames, height, width),
                mode="trilinear",
                align_corners=False,
            )
        else:
            temporal_skip = F.interpolate(
                inner.temporal_skip(grids * confidence),
                size=(window_frames, height, width),
                mode="nearest",
            )
        emit_frames = self.emit_gops * self.mode.gop_frames
        temporal_skip = temporal_skip[:, :, first : first + emit_frames]

        x = inner.attn(
            inner.r0b(
                inner.r0(inner.input(torch.cat([grids * confidence, confidence], dim=1)))
            )
        )
        if inner.deeper:
            x = inner.m0b(inner.m0a(x))
        if inner.deepest:
            x = inner.m0c(x)
        if inner.deep4:
            x = inner.m0d(x)
        if inner.compact:
            x = inner.r0c(inner.up0(x))
        x = inner.r1b(inner.r1(inner.up1(x)))
        if inner.deeper:
            x = inner.m1b(inner.m1a(x))
        if inner.deepest:
            x = inner.attn1(inner.m1c(x))

        # Attention and low-resolution convolutions see the complete latent
        # window. Only after that context has mixed do we discard frames that
        # can never be emitted. A small halo lets high-resolution temporal
        # convolutions see across both edges of the center GOP.
        x = F.interpolate(
            x, size=(window_frames, height // 4, width // 4), mode="nearest"
        )
        halo = min(self.synthesis_halo_frames, first)
        synthesis_first = first - halo
        synthesis_last = first + emit_frames + halo
        x = x[:, :, synthesis_first:synthesis_last]

        x = inner.r2b(inner.r2(inner.up2(x)))
        if inner.deep_tail:
            x = inner.r2c(x)
        if inner.deeper:
            x = inner.r2d(x)
        if inner.deepest:
            x = inner.r2e(x)
        if inner.deep4:
            x = inner.attn2(x)
        x = F.silu(inner.up3(x))
        if inner.deep_tail:
            x = inner.r3b(inner.r3(x))
        if inner.deeper:
            x = inner.r3d(inner.r3c(x))
        if inner.deepest:
            x = inner.r3f(inner.r3e(x))
        if inner.deep4:
            x = inner.r3h(inner.r3g(x))
        logits = inner.output(x)
        center = first - synthesis_first
        logits = logits[:, :, center : center + emit_frames]
        return torch.sigmoid(logits + temporal_skip)


class OverlappingGOPEncoder(nn.Module):
    """Encode a GOP through a wider grid and learned fixed-budget packer."""

    def __init__(
        self,
        mode: AETVModeSpec | str = "V8",
        width: int = 192,
        latent_channels: int = 8,
    ):
        super().__init__()
        self.mode = AETV_MODES[mode] if isinstance(mode, str) else mode
        self.width = width
        self.latent_channels = latent_channels
        self.latent_budget = self.mode.latents_per_gop
        self.grid_frames = math.ceil(self.mode.gop_frames / 2)
        self.grid_height = math.ceil(self.mode.height / 8)
        self.grid_width = math.ceil(self.mode.width / 8)
        self.grid_elements = (
            latent_channels * self.grid_frames * self.grid_height * self.grid_width
        )
        self.encoder = VideoEncoder(
            width=width,
            latent_channels=latent_channels,
            compact=False,
            preserve_time=False,
            causal=False,
            group_norm=True,
            clip_rms_latents=True,
            deep=width >= 64,
            deep2=width >= 64,
            deep3=width >= 64,
        )
        # This replaces positional truncation: every internal grid coordinate
        # can contribute to every transmitted coordinate at the fixed rate.
        self.latent_pack = nn.Linear(self.grid_elements, self.latent_budget, bias=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        grid = self.encoder(video)
        flat = grid.flatten(1)
        if flat.shape[1] != self.grid_elements:
            raise RuntimeError(
                f"encoder grid changed: expected {self.grid_elements}, got {flat.shape[1]}"
            )
        latents = self.latent_pack(flat)
        rms = latents.pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()
        return latents / rms


class OverlappingGOPAutoencoder(nn.Module):
    """One fixed-rate latent per GOP with centered sliding-window decoding."""

    checkpoint_kind = "aetv-overlapping-gop"

    def __init__(
        self,
        mode: AETVModeSpec | str = "V8",
        width: int = 192,
        latent_channels: int = 8,
        window_gops: int = 3,
        emit_gops: int = 1,
        synthesis_halo_frames: int = 2,
    ):
        super().__init__()
        self.mode = AETV_MODES[mode] if isinstance(mode, str) else mode
        self.width = width
        self.latent_channels = latent_channels
        self.window_gops = window_gops
        self.emit_gops = emit_gops
        self.synthesis_halo_frames = synthesis_halo_frames
        self.encoder = OverlappingGOPEncoder(
            mode=self.mode,
            width=width,
            latent_channels=latent_channels,
        )
        self.decoder = OverlappingGOPDecoder(
            mode=self.mode,
            width=width,
            latent_channels=latent_channels,
            window_gops=window_gops,
            emit_gops=emit_gops,
            synthesis_halo_frames=synthesis_halo_frames,
        )

    @property
    def latent_budget(self) -> int:
        return self.mode.latents_per_gop

    @property
    def lookahead_gops(self) -> int:
        return self.decoder.lookahead_gops

    def config(self) -> dict:
        return {
            "mode": self.mode.name,
            "width": self.width,
            "latent_channels": self.latent_channels,
            "window_gops": self.window_gops,
            "emit_gops": self.emit_gops,
            "synthesis_halo_frames": self.synthesis_halo_frames,
            "latents_per_gop": self.latent_budget,
            "lookahead_gops": self.lookahead_gops,
            "wire_values_per_second": self.latent_budget,
        }

    def encode_gops(
        self, gops: torch.Tensor, *, checkpoint_encoder: bool = False
    ) -> torch.Tensor:
        if gops.ndim != 6:
            raise ValueError(f"expected BGCTHW GOPs, got {tuple(gops.shape)}")
        batch, count = gops.shape[:2]
        flat = gops.flatten(0, 1)
        if checkpoint_encoder and self.training:
            latents = checkpoint(self.encoder, flat, use_reentrant=False)
        else:
            latents = self.encoder(flat)
        return latents.reshape(batch, count, self.latent_budget)

    def encode_sequence(
        self, video: torch.Tensor, *, checkpoint_encoder: bool = False
    ) -> torch.Tensor:
        return self.encode_gops(
            split_video_gops(video, self.mode.gop_frames),
            checkpoint_encoder=checkpoint_encoder,
        )

    def decode_sequence(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        checkpoint_windows: bool = False,
    ) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError(f"expected BGL latents, got {tuple(latents.shape)}")
        if latents.shape[1] < self.window_gops:
            raise ValueError(
                f"need at least {self.window_gops} GOP latents, got {latents.shape[1]}"
            )
        if weights is None:
            weights = torch.ones_like(latents)
        if weights.shape != latents.shape:
            raise ValueError("latents and weights must have identical shapes")
        remainder = (latents.shape[1] - self.window_gops) % self.emit_gops
        if remainder:
            raise ValueError(
                "latent GOP count must tile decode windows at emit_gops stride"
            )
        windows = (latents.shape[1] - self.window_gops) // self.emit_gops + 1
        decoded_rows = []
        for window_index in range(windows):
            index = window_index * self.emit_gops
            latent_window = latents[:, index : index + self.window_gops]
            weight_window = weights[:, index : index + self.window_gops]
            if checkpoint_windows and self.training:
                decoded = checkpoint(
                    self.decoder,
                    latent_window,
                    weight_window,
                    use_reentrant=False,
                )
            else:
                decoded = self.decoder(latent_window, weight_window)
            decoded_rows.append(decoded)
        # Each decoder result is a contiguous centered block of emit_gops GOPs.
        return torch.cat(decoded_rows, dim=2)

    def target_for_sequence(self, video: torch.Tensor) -> torch.Tensor:
        gops = split_video_gops(video, self.mode.gop_frames)
        trim = self.lookahead_gops
        if gops.shape[1] < self.window_gops:
            raise ValueError(
                f"need at least {self.window_gops} source GOPs, got {gops.shape[1]}"
            )
        return join_video_gops(gops[:, trim:-trim])

    def forward(
        self, video: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.decode_sequence(self.encode_sequence(video), weights)
