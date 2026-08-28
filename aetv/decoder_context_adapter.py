"""Left-context adapter inside the released V8 decoder.

The encoder, transmitted 2,816-value GOP latent, and released decoder weights
remain unchanged.  Each current GOP is decoded from its normal bottleneck
feature while a zero-initialized residual adapter reads the previous GOP's
bottleneck feature.  The first GOP is an exact stock-V8 decode, and a context
reset is an exact bypass for scene cuts or receiver discontinuities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AETVAutoencoder
from .overlap_models import join_video_gops, split_video_gops


def warp_nchw(values: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp NCHW features with pixel-unit flow (dx, dy)."""
    if values.ndim != 4 or flow.shape[1] != 2 or flow.shape[0] != values.shape[0]:
        raise ValueError("warp expects NCHW values and matching N,2,H,W flow")
    batch, _, height, width = values.shape
    if flow.shape[-2:] != (height, width):
        scale = values.new_tensor(
            (width / max(flow.shape[-1], 1), height / max(flow.shape[-2], 1))
        ).view(1, 2, 1, 1)
        flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
        flow = flow * scale
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=values.device, dtype=values.dtype),
        torch.linspace(-1, 1, width, device=values.device, dtype=values.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    scale_x = 2.0 / max(width - 1, 1)
    scale_y = 2.0 / max(height - 1, 1)
    grid = torch.stack(
        (grid[..., 0] + flow[:, 0] * scale_x, grid[..., 1] + flow[:, 1] * scale_y),
        dim=-1,
    )
    return F.grid_sample(
        values, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def warp_bcthw(values: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp each time slice of a BCTHW tensor with B,2,T,H,W flow."""
    if values.ndim != 5 or flow.shape[1] != 2:
        raise ValueError("warp_bcthw expects BCTHW values and B,2,T,H,W flow")
    batch, channels, frames, height, width = values.shape
    if flow.shape[2] == 1 and frames > 1:
        flow = flow.expand(-1, -1, frames, -1, -1)
    flat = values.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flow_flat = flow.permute(0, 2, 1, 3, 4).reshape(batch * frames, 2, *flow.shape[-2:])
    warped = warp_nchw(flat, flow_flat)
    return warped.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _groups(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return groups


class DecoderContextResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        return values + self.conv2(F.silu(self.norm2(hidden)))


class CrossGOPBottleneckAdapter(nn.Module):
    """Fuse the previous GOP edge into the current decoder bottleneck.

    Current bottleneck positions attend to the previous GOP's final latent-time
    plane after a DCVC-style learned warp.  A compact convolutional branch then
    combines the aligned content, local differences, and receive confidence.
    Flow and the final convolution are zero initialized, so construction and
    reset remain exact stock-decoder functions.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        width: int = 128,
        attention_dim: int = 64,
        heads: int = 4,
        blocks: int = 3,
        temporal_taper: tuple[float, ...] = (1.0, 0.65, 0.30),
        preserve_highpass: bool = False,
        highpass_kernel: int = 5,
        context_bias: float = 0.0,
    ):
        super().__init__()
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.feature_channels = feature_channels
        self.width = width
        self.attention_dim = attention_dim
        self.heads = heads
        self.blocks = blocks
        self.temporal_taper = tuple(temporal_taper)
        self.preserve_highpass = bool(preserve_highpass)
        self.highpass_kernel = int(highpass_kernel)

        groups = _groups(feature_channels)
        self.current_norm = nn.GroupNorm(groups, feature_channels)
        self.previous_norm = nn.GroupNorm(groups, feature_channels)
        self.query = nn.Conv3d(feature_channels, attention_dim, 1)
        self.key = nn.Conv3d(feature_channels, attention_dim, 1)
        self.value = nn.Conv3d(feature_channels, attention_dim, 1)
        self.difference = nn.Conv3d(feature_channels, attention_dim, 1)
        self.input = nn.Conv3d(4 * attention_dim + 2, width, 3, padding=1)
        self.body = nn.Sequential(
            *(DecoderContextResidualBlock(width) for _ in range(blocks))
        )
        self.output = nn.Conv3d(width, feature_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.flow = nn.Conv3d(2 * feature_channels, 2, 3, padding=1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)
        self.max_flow = 4.0

        # Global cut gate: edge difference, RMS difference, cosine similarity,
        # feature energies, and receive confidence.  Synthetic discontinuities
        # provide direct supervision during the gated experiment.
        self.scene_gate = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        if context_bias:
            nn.init.constant_(self.scene_gate[-1].bias, float(context_bias))

    def config(self) -> dict:
        return {
            "feature_channels": self.feature_channels,
            "width": self.width,
            "attention_dim": self.attention_dim,
            "heads": self.heads,
            "blocks": self.blocks,
            "temporal_taper": self.temporal_taper,
            "preserve_highpass": self.preserve_highpass,
            "highpass_kernel": self.highpass_kernel,
        }

    def _cross_attention(
        self, current: torch.Tensor, previous_edge: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, frames, height, width = current.shape
        head_dim = self.attention_dim // self.heads
        query_grid = self.query(current)
        key_grid = self.key(previous_edge)
        value_grid = self.value(previous_edge)
        query = query_grid.flatten(2).reshape(
            batch, self.heads, head_dim, frames * height * width
        ).transpose(-1, -2)
        key = key_grid.flatten(2).reshape(
            batch, self.heads, head_dim, height * width
        ).transpose(-1, -2)
        value = value_grid.flatten(2).reshape(
            batch, self.heads, head_dim, height * width
        ).transpose(-1, -2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(-1, -2).reshape(
            batch, self.attention_dim, frames, height, width
        )
        return query_grid, attended, value_grid

    def _scene_statistics(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        first = current[:, :, 0].float()
        last = previous[:, :, -1].float()
        difference = first - last
        l1 = difference.abs().mean(dim=(1, 2, 3))
        rms = difference.square().mean(dim=(1, 2, 3)).sqrt()
        cosine = F.cosine_similarity(first, last, dim=1).mean(dim=(1, 2))
        first_rms = first.square().mean(dim=(1, 2, 3)).sqrt()
        last_rms = last.square().mean(dim=(1, 2, 3)).sqrt()
        return torch.stack((l1, rms, cosine, first_rms, last_rms, confidence.float()), dim=1)

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        confidence: torch.Tensor,
        *,
        reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current.shape != previous.shape or current.ndim != 5:
            raise ValueError("current and previous bottlenecks must have equal BCTHW shapes")
        if current.shape[1] != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels")
        batch, _, frames, _, _ = current.shape
        if confidence.shape != (batch,):
            raise ValueError("confidence must contain one scalar per GOP pair")

        statistics = self._scene_statistics(current, previous, confidence)
        scene_gate = torch.sigmoid(self.scene_gate(statistics)).flatten()
        scene_gate = scene_gate * confidence.to(scene_gate.dtype).clamp(0, 1)
        if reset is not None:
            if reset.shape != (batch,):
                raise ValueError("reset must contain one flag per GOP pair")
            scene_gate = scene_gate * (~reset.bool()).to(scene_gate.dtype)

        current_norm = self.current_norm(current)
        previous_norm = self.previous_norm(previous)
        previous_edge = previous_norm[:, :, -1:]
        flow = self.max_flow * torch.tanh(
            self.flow(torch.cat((current_norm[:, :, :1], previous_edge), dim=1))
        )
        previous_edge = warp_bcthw(previous_edge, flow)
        previous_local = previous_edge.expand(-1, -1, frames, -1, -1)
        query, attended, value = self._cross_attention(current_norm, previous_edge)
        local_value = value.expand(-1, -1, frames, -1, -1)
        difference = self.difference(current_norm - previous_local)
        confidence_map = confidence.to(current.dtype).view(batch, 1, 1, 1, 1)
        confidence_map = confidence_map.expand(-1, 1, *current.shape[2:])
        gate_map = scene_gate.to(current.dtype).view(batch, 1, 1, 1, 1)
        gate_map = gate_map.expand_as(confidence_map)
        features = torch.cat(
            (query, attended, local_value, difference, confidence_map, gate_map), dim=1
        )
        residual = self.output(self.body(F.silu(self.input(features))))
        taper = current.new_tensor(self.temporal_taper)
        if taper.numel() != frames:
            taper = F.interpolate(
                taper.view(1, 1, -1), size=frames, mode="linear", align_corners=False
            ).flatten()
        taper = taper.view(1, 1, frames, 1, 1)
        delta = torch.tanh(residual) * taper * gate_map
        if self.preserve_highpass:
            kernel = max(3, self.highpass_kernel | 1)
            pad = kernel // 2
            delta = F.avg_pool3d(delta, (1, kernel, kernel), stride=1, padding=(0, pad, pad))
        corrected = current + delta
        return corrected, scene_gate


class V8DecoderContextAdapter(nn.Module):
    """Released V8 with one previous-GOP decoder-bottleneck context."""

    checkpoint_kind = "aetv-v8-decoder-context-adapter"

    def __init__(
        self,
        base: AETVAutoencoder,
        *,
        adapter_width: int = 128,
        attention_dim: int = 64,
        attention_heads: int = 4,
        adapter_blocks: int = 3,
        freeze_base: bool = True,
        temporal_taper: tuple[float, ...] = (1.0, 0.65, 0.30),
        preserve_highpass: bool = False,
        context_bias: float = 0.0,
    ):
        super().__init__()
        if base.mode.name != "V8" or base.encoder.latent_channels != 3:
            raise ValueError("the decoder context adapter requires released-layout V8")
        if base.decoder.compact:
            raise ValueError("the released-layout decoder must not be compact")
        self.mode = base.mode
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.latent_budget = self.mode.latents_per_gop
        inner = self.decoder.decoder
        feature_channels = inner.input.conv.out_channels
        self.context_adapter = CrossGOPBottleneckAdapter(
            feature_channels,
            width=adapter_width,
            attention_dim=attention_dim,
            heads=attention_heads,
            blocks=adapter_blocks,
            temporal_taper=temporal_taper,
            preserve_highpass=preserve_highpass,
            context_bias=context_bias,
        )
        if freeze_base:
            self.freeze_base()

    @classmethod
    def from_v8_checkpoint(
        cls,
        checkpoint: str,
        **adapter_kwargs,
    ) -> "V8DecoderContextAdapter":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload.get("args", {}) or {}
        if hasattr(args, "__dict__"):
            args = vars(args)
        base = AETVAutoencoder(
            mode="V8",
            width=int(args.get("model_width", 128)),
            latent_channels=int(args.get("latent_channels", 3)),
            compact=bool(args.get("compact", False)),
        )
        base.load_state_dict(payload["model_state_dict"], strict=True)
        return cls(base, **adapter_kwargs)

    @classmethod
    def from_adapter_checkpoint(
        cls,
        checkpoint: str,
        *,
        base_checkpoint: str | None = None,
        freeze_base: bool = True,
    ) -> "V8DecoderContextAdapter":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("kind") != cls.checkpoint_kind:
            raise ValueError(f"{checkpoint} is not a decoder context adapter checkpoint")
        config = payload["model_config"]["adapter"]
        model = cls.from_v8_checkpoint(
            base_checkpoint or payload["base_checkpoint"],
            adapter_width=int(config["width"]),
            attention_dim=int(config["attention_dim"]),
            attention_heads=int(config["heads"]),
            adapter_blocks=int(config["blocks"]),
            freeze_base=freeze_base,
            temporal_taper=tuple(config.get("temporal_taper", (1.0, 0.65, 0.30))),
            preserve_highpass=bool(config.get("preserve_highpass", False)),
        )
        model.context_adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
        return model

    def freeze_base(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    @property
    def lookback_gops(self) -> int:
        return 1

    def config(self) -> dict:
        return {
            "mode": self.mode.name,
            "latents_per_gop": self.latent_budget,
            "lookback_gops": self.lookback_gops,
            "lookahead_gops": 0,
            "adapter": self.context_adapter.config(),
        }

    def encode_sequence(self, video: torch.Tensor) -> torch.Tensor:
        gops = split_video_gops(video, self.mode.gop_frames)
        batch, count = gops.shape[:2]
        latents = self.encoder(gops.flatten(0, 1))
        return latents.reshape(batch, count, self.latent_budget)

    def _latent_grids(
        self, latents: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 3 or latents.shape != weights.shape:
            raise ValueError("latents and weights must have identical BGL shapes")
        batch, count, _ = latents.shape
        output_shape = (self.mode.gop_frames, self.mode.height, self.mode.width)
        t_latent, h_latent, w_latent = self.decoder._get_grid_shape(output_shape)
        total = self.decoder.latent_channels * t_latent * h_latent * w_latent
        z_flat = latents.new_zeros((batch, count, total))
        w_flat = weights.new_zeros((batch, count, total))
        copy_len = min(latents.shape[-1], total)
        z_flat[..., :copy_len] = latents[..., :copy_len]
        w_flat[..., :copy_len] = weights[..., :copy_len]
        shape = (batch, count, self.decoder.latent_channels, t_latent, h_latent, w_latent)
        return z_flat.reshape(shape), w_flat.reshape(shape)

    def _stem(self, z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        inner = self.decoder.decoder
        hidden = inner.input(torch.cat((z * weights, weights), dim=1))
        hidden = inner.attn(inner.r0b(inner.r0(hidden)))
        if inner.deeper:
            hidden = inner.m0b(inner.m0a(hidden))
        if inner.deepest:
            hidden = inner.m0c(hidden)
        if inner.deep4:
            hidden = inner.m0d(hidden)
        return hidden

    def _tail(
        self,
        hidden: torch.Tensor,
        z: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        inner = self.decoder.decoder
        frames, height, width = (
            self.mode.gop_frames,
            self.mode.height,
            self.mode.width,
        )
        output_shape = (frames, height, width)
        if inner.smooth_temporal_skip:
            temporal_skip = F.interpolate(
                inner.temporal_skip(z * weights),
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            )
        else:
            temporal_skip = F.interpolate(
                inner.temporal_skip(z * weights), size=output_shape, mode="nearest"
            )
        if inner.compact:
            hidden = inner.r0c(inner.up0(hidden))
        hidden = inner.r1b(inner.r1(inner.up1(hidden)))
        if inner.deeper:
            hidden = inner.m1b(inner.m1a(hidden))
        if inner.deepest:
            hidden = inner.attn1(inner.m1c(hidden))
        hidden = F.interpolate(
            hidden, size=(frames, height // 4, width // 4), mode="nearest"
        )
        hidden = inner.r2b(inner.r2(inner.up2(hidden)))
        if inner.deep_tail:
            hidden = inner.r2c(hidden)
        if inner.deeper:
            hidden = inner.r2d(hidden)
        if inner.deepest:
            hidden = inner.r2e(hidden)
        if inner.deep4:
            hidden = inner.attn2(hidden)
        hidden = F.silu(inner.up3(hidden))
        if inner.deep_tail:
            hidden = inner.r3b(inner.r3(hidden))
        if inner.deeper:
            hidden = inner.r3d(inner.r3c(hidden))
        if inner.deepest:
            hidden = inner.r3f(inner.r3e(hidden))
        if inner.deep4:
            hidden = inner.r3h(inner.r3g(hidden))
        return torch.sigmoid(inner.output(hidden) + temporal_skip)

    def decode_sequence(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        use_adapter: bool = True,
        context_source_indices: torch.Tensor | None = None,
        context_reset: torch.Tensor | None = None,
        recurrent_state: bool = False,
        return_gates: bool = False,
        return_base: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if weights is None:
            weights = torch.ones_like(latents)
        z, confidence_grid = self._latent_grids(latents, weights)
        batch, count = z.shape[:2]
        if count < 1:
            raise ValueError("at least one GOP is required")
        flat_z = z.flatten(0, 1)
        flat_confidence_grid = confidence_grid.flatten(0, 1)
        base_hidden = self._stem(flat_z, flat_confidence_grid).reshape(
            batch, count, -1, *z.shape[-3:]
        )
        corrected = base_hidden
        gates = latents.new_zeros((batch, 0))
        if use_adapter and count > 1 and recurrent_state:
            # Carry the corrected bottleneck forward.  The original adapter
            # deliberately used only the previous *base* feature, which made
            # it a pairwise boundary filter rather than a stream state.
            adapted_items = [base_hidden[:, 0]]
            gate_items = []
            for index in range(1, count):
                previous = adapted_items[-1]
                previous_confidence = weights[:, index - 1].mean(dim=-1)
                current_confidence = weights[:, index].mean(dim=-1)
                pair_confidence = torch.minimum(previous_confidence, current_confidence)
                reset = None if context_reset is None else context_reset[:, index - 1]
                adapted, gate = self.context_adapter(
                    base_hidden[:, index], previous, pair_confidence, reset=reset
                )
                adapted_items.append(adapted)
                gate_items.append(gate)
            corrected = torch.stack(adapted_items, dim=1)
            gates = torch.stack(gate_items, dim=1)
        elif use_adapter and count > 1:
            current = base_hidden[:, 1:].flatten(0, 1)
            if context_source_indices is None:
                previous = base_hidden[:, :-1].flatten(0, 1)
                previous_confidence = weights[:, :-1].mean(dim=-1).flatten()
            else:
                if context_source_indices.shape != (batch, count - 1):
                    raise ValueError("context_source_indices must have shape B,G-1")
                indices = context_source_indices.to(base_hidden.device).long().flatten()
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= batch * count
                ):
                    raise ValueError("context source index is outside the latent sequence")
                previous = base_hidden.flatten(0, 1).index_select(0, indices)
                previous_confidence = weights.reshape(batch * count, -1).mean(dim=-1).index_select(
                    0, indices
                )
            current_confidence = weights[:, 1:].mean(dim=-1).flatten()
            pair_confidence = torch.minimum(previous_confidence, current_confidence)
            reset = None if context_reset is None else context_reset.flatten()
            adapted, gates = self.context_adapter(
                current, previous, pair_confidence, reset=reset
            )
            corrected = torch.cat(
                (base_hidden[:, :1], adapted.reshape(batch, count - 1, *adapted.shape[1:])),
                dim=1,
            )
            gates = gates.reshape(batch, count - 1)
        decoded = self._tail(corrected.flatten(0, 1), flat_z, flat_confidence_grid)
        gops = decoded.reshape(batch, count, *decoded.shape[1:])
        video = join_video_gops(gops)
        if return_base:
            base_decoded = self._tail(
                base_hidden.flatten(0, 1), flat_z, flat_confidence_grid
            )
            base_gops = base_decoded.reshape(batch, count, *base_decoded.shape[1:])
            base_video = join_video_gops(base_gops)
            if return_gates:
                return video, base_video, gates
            return video, base_video
        if return_gates:
            return video, gates
        return video

    def target_for_sequence(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape[2] % self.mode.gop_frames:
            raise ValueError("target video must contain complete GOPs")
        return video
