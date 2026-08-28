"""Fixed-rate V8 with an explicit transmitted transition/appearance anchor."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .recurrent_joint_codec import V8RecurrentJointCodec


class TransitionAppearanceEncoder(nn.Module):
    def __init__(self, values: int):
        super().__init__()
        self.values = values
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.output = nn.Linear(64 * 4 * 8, values)

    def forward(self, previous_last: torch.Tensor, current_first: torch.Tensor) -> torch.Tensor:
        appearance_and_delta = torch.cat((current_first, current_first - previous_last), dim=1)
        learned = self.output(self.features(appearance_and_delta).flatten(1))

        # Guarantee that the reserved coordinates actually carry the new
        # source information from step zero.  Half is low-pass current-frame
        # appearance and half is the source transition across the GOP join;
        # the learned branch can refine, but cannot begin as another opaque
        # receiver-capacity experiment.
        current_y = (
            0.299 * current_first[:, 0:1]
            + 0.587 * current_first[:, 1:2]
            + 0.114 * current_first[:, 2:3]
        )
        previous_y = (
            0.299 * previous_last[:, 0:1]
            + 0.587 * previous_last[:, 1:2]
            + 0.114 * previous_last[:, 2:3]
        )
        grid = (4, 8) if self.values == 64 else (4, 4)
        appearance = F.adaptive_avg_pool2d(current_y, grid).flatten(1)
        transition = F.adaptive_avg_pool2d(current_y - previous_y, grid).flatten(1)
        fixed = torch.cat((2.0 * appearance - 1.0, 2.0 * transition), dim=1)
        return fixed + 0.1 * torch.tanh(learned)


class TransitionAppearanceDecoder(nn.Module):
    def __init__(self, values: int, feature_channels: int):
        super().__init__()
        self.values = values
        self.input = nn.Linear(2 * values, 64 * 3 * 4 * 8)
        self.body = nn.Sequential(
            nn.Conv3d(64, 96, 3, padding=1), nn.SiLU(),
            nn.Conv3d(96, feature_channels, 3, padding=1),
        )
        # The side channel starts as a decoder no-op; the TX representation is
        # new, but receiver capacity cannot conceal its effect during training.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden = self.input(torch.cat((values * weights, weights), dim=1))
        hidden = hidden.reshape(values.shape[0], 64, 3, 4, 8)
        hidden = F.interpolate(hidden, size=output_shape, mode="trilinear", align_corners=False)
        return self.body(hidden)


class V8TransitionAnchorCodec(V8RecurrentJointCodec):
    """Reserve 32/64 of 2,816 values for true cross-boundary source data."""

    checkpoint_kind = "aetv-v8-transition-anchor-codec"

    def __init__(self, model, *, anchor_values: int = 64):
        super().__init__(model)
        if anchor_values not in (32, 64):
            raise ValueError("transition anchor must contain 32 or 64 values")
        self.anchor_values = anchor_values
        feature_channels = model.decoder.decoder.input.conv.out_channels
        self.anchor_encoder = TransitionAppearanceEncoder(anchor_values)
        self.anchor_decoder = TransitionAppearanceDecoder(anchor_values, feature_channels)

    @classmethod
    def from_released(
        cls,
        checkpoint_path: str | Path = "models/v8-hf3k-face-gan.pt",
        *,
        anchor_values: int = 64,
        adapter_width: int = 192,
        attention_dim: int = 96,
        adapter_blocks: int = 5,
    ) -> "V8TransitionAnchorCodec":
        base = V8RecurrentJointCodec.from_released(
            checkpoint_path,
            adapter_width=adapter_width,
            attention_dim=attention_dim,
            adapter_blocks=adapter_blocks,
        )
        return cls(base.model, anchor_values=anchor_values)

    def set_trainable_contract(self) -> dict[str, int]:
        counts = super().set_trainable_contract()
        for module in (self.anchor_encoder, self.anchor_decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        counts["transition_anchor"] = sum(
            p.numel()
            for module in (self.anchor_encoder, self.anchor_decoder)
            for p in module.parameters()
            if p.requires_grad
        )
        return counts

    def encode_gops(self, video: torch.Tensor, *, use_checkpoint: bool = False) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] % 6:
            raise ValueError("video must contain complete six-frame GOPs")
        count = video.shape[2] // 6
        transmitted = []
        previous_last = video[:, :, 0]
        for index in range(count):
            gop = video[:, :, index * 6 : (index + 1) * 6]
            if use_checkpoint and self.training:
                main = checkpoint(self.encoder, gop, use_reentrant=False)
                anchor = checkpoint(
                    self.anchor_encoder, previous_last, gop[:, :, 0], use_reentrant=False
                )
            else:
                main = self.encoder(gop)
                anchor = self.anchor_encoder(previous_last, gop[:, :, 0])
            combined = torch.cat((main[:, : -self.anchor_values], anchor), dim=1)
            combined = combined / combined.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
            transmitted.append(combined)
            previous_last = gop[:, :, -1]
        result = torch.stack(transmitted, dim=1)
        if result.shape[-1] != 2816:
            raise RuntimeError("transition anchor changed the V8 wire budget")
        return result

    def decode_gops(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        reset: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        use_checkpoint: bool = False,
        return_gates: bool = False,
    ):
        if latents.ndim != 3 or latents.shape[-1] != 2816:
            raise ValueError("latents must have shape B,G,2816")
        if weights is None:
            weights = torch.ones_like(latents)
        batch, count, _ = latents.shape
        if reset is None:
            reset = torch.zeros(batch, count, dtype=torch.bool, device=latents.device)
        if missing is None:
            missing = weights.mean(dim=-1) <= 1e-6

        anchor = latents[..., -self.anchor_values :]
        anchor_weights = weights[..., -self.anchor_values :]
        main = latents.clone()
        main_weights = weights.clone()
        main[..., -self.anchor_values :] = 0
        main_weights[..., -self.anchor_values :] = 0
        z, confidence_grid = self.model._latent_grids(main, main_weights)
        outputs, gates = [], []
        state = None
        for index in range(count):
            z_item, weight_item = z[:, index], confidence_grid[:, index]
            if use_checkpoint and self.training:
                base = checkpoint(self.model._stem, z_item, weight_item, use_reentrant=False)
                side = checkpoint(
                    self.anchor_decoder,
                    anchor[:, index],
                    anchor_weights[:, index],
                    tuple(base.shape[-3:]),
                    use_reentrant=False,
                )
            else:
                base = self.model._stem(z_item, weight_item)
                side = self.anchor_decoder(
                    anchor[:, index], anchor_weights[:, index], tuple(base.shape[-3:])
                )
            base = base + side
            if state is None:
                corrected, gate = base, latents.new_zeros(batch)
            else:
                confidence = weights[:, index].mean(dim=-1)
                corrected, gate = self.context_adapter(
                    base, state, confidence, reset=reset[:, index]
                )
            gates.append(gate)
            if state is None:
                state = corrected
            else:
                state = torch.where((~missing[:, index]).view(batch, 1, 1, 1, 1), corrected, state)
            if use_checkpoint and self.training:
                decoded = checkpoint(
                    self.model._tail, corrected, z_item, weight_item, use_reentrant=False
                )
            else:
                decoded = self.model._tail(corrected, z_item, weight_item)
            outputs.append(decoded)
        result = torch.cat(outputs, dim=2)
        gate_tensor = torch.stack(gates, dim=1)
        return (result, gate_tensor) if return_gates else result
