"""V8-preserving overlapping-GOP latent context adapter.

The released V8 encoder, 2,816-value latent layout, and decoder remain intact.
A zero-initialized residual adapter is the only new trainable component.  It
sees five directly unpacked V8 latent grids and adjusts the centered three
before the original decoder reconstructs each GOP.  At initialization the
model is exactly the released GOP-local codec.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AETVAutoencoder
from .overlap_models import join_video_gops, split_video_gops


class ContextResidualBlock(nn.Module):
    def __init__(self, channels: int, temporal_dilation: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            3,
            padding=(temporal_dilation, 1, 1),
            dilation=(temporal_dilation, 1, 1),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        return values + self.conv2(F.silu(self.norm2(hidden)))


class ZeroInitLatentContextAdapter(nn.Module):
    """Predict a bounded residual from neighboring V8 latent grids."""

    def __init__(self, latent_channels: int = 3, width: int = 64):
        super().__init__()
        self.input = nn.Conv3d(2 * latent_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            ContextResidualBlock(width, dilation) for dilation in (1, 2, 4, 2, 1)
        )
        self.output = nn.Conv3d(width, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, grids: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input(torch.cat((grids * confidence, confidence), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        # Tanh bounds a learned correction without changing the exact zero-init
        # function. The base latent itself remains in the released V8 domain.
        return grids + torch.tanh(self.output(F.silu(hidden)))


class V8OverlapAdapter(nn.Module):
    """Released V8 plus a centered five-GOP latent context adapter."""

    checkpoint_kind = "aetv-v8-overlap-adapter"

    def __init__(
        self,
        base: AETVAutoencoder,
        *,
        window_gops: int = 5,
        emit_gops: int = 3,
        adapter_width: int = 64,
        freeze_base: bool = True,
    ):
        super().__init__()
        if base.mode.name != "V8" or base.encoder.latent_channels != 3:
            raise ValueError("the overlap adapter requires released-layout V8")
        if window_gops < 3 or (window_gops - emit_gops) < 2:
            raise ValueError("decode window must retain context on both sides")
        if (window_gops - emit_gops) % 2:
            raise ValueError("window_gops - emit_gops must be even")
        self.mode = base.mode
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.window_gops = window_gops
        self.emit_gops = emit_gops
        self.adapter_width = adapter_width
        self.latent_budget = self.mode.latents_per_gop
        self.latent_channels = 3
        self.latent_frames = self.mode.gop_frames // 2
        self.grid_height = self.mode.height // 8
        self.grid_width = self.mode.width // 8
        self.grid_elements = (
            self.latent_channels
            * self.latent_frames
            * self.grid_height
            * self.grid_width
        )
        self.adapter = ZeroInitLatentContextAdapter(
            latent_channels=self.latent_channels, width=adapter_width
        )
        if freeze_base:
            self.freeze_base()

    @classmethod
    def from_v8_checkpoint(
        cls,
        checkpoint: str,
        *,
        window_gops: int = 5,
        emit_gops: int = 3,
        adapter_width: int = 64,
        freeze_base: bool = True,
    ) -> "V8OverlapAdapter":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        base = AETVAutoencoder(
            mode="V8",
            width=int(args.get("model_width", 128)),
            latent_channels=int(args.get("latent_channels", 3)),
            compact=bool(args.get("compact", False)),
        )
        base.load_state_dict(payload["model_state_dict"], strict=True)
        return cls(
            base,
            window_gops=window_gops,
            emit_gops=emit_gops,
            adapter_width=adapter_width,
            freeze_base=freeze_base,
        )

    @property
    def lookahead_gops(self) -> int:
        return (self.window_gops - self.emit_gops) // 2

    def freeze_base(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    def config(self) -> dict:
        return {
            "mode": self.mode.name,
            "window_gops": self.window_gops,
            "emit_gops": self.emit_gops,
            "adapter_width": self.adapter_width,
            "lookahead_gops": self.lookahead_gops,
            "latents_per_gop": self.latent_budget,
        }

    def encode_sequence(self, video: torch.Tensor) -> torch.Tensor:
        gops = split_video_gops(video, self.mode.gop_frames)
        batch, count = gops.shape[:2]
        latents = self.encoder(gops.flatten(0, 1))
        return latents.reshape(batch, count, self.latent_budget)

    def _direct_grids(
        self, latents: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, budget = latents.shape
        if budget != self.latent_budget or weights.shape != latents.shape:
            raise ValueError("invalid V8 latent or confidence shape")
        values = latents[..., : self.grid_elements].reshape(
            batch,
            count,
            self.latent_channels,
            self.latent_frames,
            self.grid_height,
            self.grid_width,
        )
        confidence = weights[..., : self.grid_elements].reshape_as(values)
        values = values.permute(0, 2, 1, 3, 4, 5).reshape(
            batch,
            self.latent_channels,
            count * self.latent_frames,
            self.grid_height,
            self.grid_width,
        )
        confidence = confidence.permute(0, 2, 1, 3, 4, 5).reshape_as(values)
        return values, confidence

    def decode_window(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor,
        *,
        use_adapter: bool = True,
    ) -> torch.Tensor:
        if latents.shape[1] != self.window_gops:
            raise ValueError(f"expected {self.window_gops} GOPs in a decode window")
        grids, confidence = self._direct_grids(latents, weights)
        if use_adapter:
            grids = self.adapter(grids, confidence)
        first = self.lookahead_gops * self.latent_frames
        count = self.emit_gops * self.latent_frames
        grids = grids[:, :, first : first + count]
        confidence = confidence[:, :, first : first + count]
        batch = grids.shape[0]
        grids = grids.reshape(
            batch,
            self.latent_channels,
            self.emit_gops,
            self.latent_frames,
            self.grid_height,
            self.grid_width,
        ).permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        confidence = confidence.reshape(
            batch,
            self.latent_channels,
            self.emit_gops,
            self.latent_frames,
            self.grid_height,
            self.grid_width,
        ).permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        decoded = self.decoder.decoder(
            grids,
            confidence,
            (self.mode.gop_frames, self.mode.height, self.mode.width),
        )
        decoded = decoded.reshape(
            batch,
            self.emit_gops,
            3,
            self.mode.gop_frames,
            self.mode.height,
            self.mode.width,
        )
        return join_video_gops(decoded)

    def decode_sequence(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        use_adapter: bool = True,
    ) -> torch.Tensor:
        if weights is None:
            weights = torch.ones_like(latents)
        if latents.shape != weights.shape or latents.ndim != 3:
            raise ValueError("latents and weights must have identical BGL shapes")
        remainder = (latents.shape[1] - self.window_gops) % self.emit_gops
        if latents.shape[1] < self.window_gops or remainder:
            raise ValueError("latent sequence does not tile complete decode windows")
        rows = []
        for index in range(0, latents.shape[1] - self.window_gops + 1, self.emit_gops):
            rows.append(
                self.decode_window(
                    latents[:, index : index + self.window_gops],
                    weights[:, index : index + self.window_gops],
                    use_adapter=use_adapter,
                )
            )
        return torch.cat(rows, dim=2)

    def target_for_sequence(self, video: torch.Tensor) -> torch.Tensor:
        gops = split_video_gops(video, self.mode.gop_frames)
        trim = self.lookahead_gops
        return join_video_gops(gops[:, trim:-trim])

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        latents = self.encode_sequence(video)
        return self.decode_sequence(latents)
