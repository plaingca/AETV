"""Warm-started recurrent V8 codec for zero-rate-change GOP experiments.

The transmitter still emits one independently normalized 2,816-value vector
for every six source frames.  Only the learned representation and receiver
state change: the current-GOP encoder is trainable, it may condition on the
previous source frame, and the decoder carries its corrected bottleneck
causally between GOPs.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .decoder_context_adapter import V8DecoderContextAdapter, warp_nchw


class PreviousFrameStemConditioner(nn.Module):
    """Zero-init residual from the motion-aligned previous last source frame.

    DCVC-style: warp the previous last frame toward the current first frame,
    then inject at the encoder stem.  After a cut/reset the residual is
    multiplied by zero, so the GOP is an exact independent current-GOP encode.
    """

    def __init__(self, channels: int, frames: int = 6):
        super().__init__()
        self.channels = channels
        self.frames = frames
        self.features = nn.Sequential(
            nn.Conv2d(6, channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
        )
        self.project = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.flow = nn.Conv2d(6, 2, 7, padding=3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)
        self.max_flow = 12.0
        self.register_buffer(
            "taper",
            torch.tensor((1.0, 0.85, 0.65, 0.45, 0.2, 0.0)).view(1, 1, -1, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        previous_last: torch.Tensor,
        current_first: torch.Tensor,
        temporal_size: int,
    ) -> torch.Tensor:
        flow = self.max_flow * torch.tanh(
            self.flow(torch.cat((current_first, previous_last), dim=1))
        )
        warped_previous = warp_nchw(previous_last, flow)
        hidden = self.features(torch.cat((current_first, warped_previous), dim=1))
        hidden = hidden.unsqueeze(2).expand(-1, -1, temporal_size, -1, -1)
        residual = self.project(hidden)
        taper = self.taper
        if taper.shape[2] != temporal_size:
            taper = F.interpolate(taper, size=(temporal_size, 1, 1), mode="nearest")
        return residual * taper


class V8RecurrentJointCodec(nn.Module):
    """Released V8 plus a resettable recurrent decoder-bottleneck state."""

    checkpoint_kind = "aetv-v8-recurrent-joint-codec"

    def __init__(self, model: V8DecoderContextAdapter):
        super().__init__()
        self.model = model
        self.mode = model.mode
        self.latent_budget = model.latent_budget
        stem_channels = int(model.encoder.encoder.net[0].conv.out_channels)
        self.encoder_context = PreviousFrameStemConditioner(
            stem_channels, frames=self.mode.gop_frames
        )

    @classmethod
    def from_released(
        cls,
        checkpoint_path: str | Path = "models/v8-hf3k-face-gan.pt",
        *,
        adapter_width: int = 192,
        attention_dim: int = 96,
        adapter_blocks: int = 5,
    ) -> "V8RecurrentJointCodec":
        model = V8DecoderContextAdapter.from_v8_checkpoint(
            str(checkpoint_path),
            adapter_width=adapter_width,
            attention_dim=attention_dim,
            adapter_blocks=adapter_blocks,
            freeze_base=False,
            temporal_taper=(1.0, 0.5, 0.0),
            preserve_highpass=True,
            context_bias=1.5,
        )
        return cls(model)

    @property
    def encoder(self) -> nn.Module:
        return self.model.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.model.decoder

    @property
    def context_adapter(self) -> nn.Module:
        return self.model.context_adapter

    def set_trainable_contract(self) -> dict[str, int]:
        """Train the whole current-GOP encoder, recurrent state, and RX tail.

        The decoder stem stays at the released operating point.  Every module
        after the bottleneck, including the direct temporal skip, is jointly
        trainable rather than acting as a frozen image synthesizer.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.encoder_context.parameters():
            parameter.requires_grad_(True)
        for parameter in self.context_adapter.parameters():
            parameter.requires_grad_(True)

        inner = self.decoder.decoder
        tail_names = (
            "temporal_skip", "up0", "r0c", "up1", "r1", "r1b", "m1a",
            "m1b", "m1c", "attn1", "up2", "r2", "r2b", "r2c", "r2d",
            "r2e", "attn2", "up3", "r3", "r3b", "r3c", "r3d", "r3e",
            "r3f", "r3g", "r3h", "output",
        )
        for name in tail_names:
            module = getattr(inner, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            "encoder_context": sum(
                p.numel() for p in self.encoder_context.parameters() if p.requires_grad
            ),
            "state": sum(p.numel() for p in self.context_adapter.parameters() if p.requires_grad),
            "decoder_tail": sum(p.numel() for p in self.decoder.parameters() if p.requires_grad),
        }
        required = {key: value for key, value in counts.items() if key != "encoder_context"}
        if not all(required.values()) or counts["encoder_context"] <= 0:
            raise RuntimeError(f"incomplete trainable contract: {counts}")
        return counts

    def _encode_gop(
        self,
        gop: torch.Tensor,
        previous_last: torch.Tensor,
        use_context: torch.Tensor,
    ) -> torch.Tensor:
        stem = self.encoder_context(previous_last, gop[:, :, 0], gop.shape[2])
        stem = stem * use_context.to(dtype=stem.dtype).view(-1, 1, 1, 1, 1)
        encoded = self.encoder(gop, stem_residual=stem)
        if encoded.shape[-1] != self.latent_budget:
            raise RuntimeError("encoder violated the 2,816-value V8 wire budget")
        return encoded

    def encode_gops(
        self,
        video: torch.Tensor,
        *,
        reset: torch.Tensor | None = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must be B,C,T,H,W")
        frames = self.mode.gop_frames
        if video.shape[2] % frames:
            raise ValueError("video must contain complete six-frame GOPs")
        count = video.shape[2] // frames
        batch = video.shape[0]
        if reset is None:
            reset = torch.zeros(batch, count, dtype=torch.bool, device=video.device)
        if reset.shape != (batch, count):
            raise ValueError("reset must have shape B,G")
        items = []
        previous_last = video[:, :, 0]
        for index in range(count):
            gop = video[:, :, index * frames : (index + 1) * frames]
            use_context = (~reset[:, index]).to(video.dtype)
            if index == 0:
                use_context = torch.zeros_like(use_context)
            if use_checkpoint and self.training:
                encoded = checkpoint(
                    self._encode_gop, gop, previous_last, use_context, use_reentrant=False
                )
            else:
                encoded = self._encode_gop(gop, previous_last, use_context)
            items.append(encoded)
            previous_last = gop[:, :, -1]
        return torch.stack(items, dim=1)

    def decode_gops(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        reset: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        use_checkpoint: bool = False,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 3 or latents.shape[-1] != self.latent_budget:
            raise ValueError("latents must have shape B,G,2816")
        if weights is None:
            weights = torch.ones_like(latents)
        if weights.shape != latents.shape:
            raise ValueError("weights must match latents")
        batch, count, _ = latents.shape
        if reset is None:
            reset = torch.zeros(batch, count, dtype=torch.bool, device=latents.device)
        if missing is None:
            missing = weights.mean(dim=-1) <= 1e-6
        if reset.shape != (batch, count) or missing.shape != (batch, count):
            raise ValueError("reset and missing must have shape B,G")

        z, confidence_grid = self.model._latent_grids(latents, weights)
        outputs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        state: torch.Tensor | None = None
        for index in range(count):
            z_item = z[:, index]
            weight_item = confidence_grid[:, index]
            if use_checkpoint and self.training:
                base = checkpoint(self.model._stem, z_item, weight_item, use_reentrant=False)
            else:
                base = self.model._stem(z_item, weight_item)

            reset_item = reset[:, index]
            valid_item = ~missing[:, index]
            if state is None:
                corrected = base
                gate = latents.new_zeros(batch)
            else:
                confidence = weights[:, index].mean(dim=-1)
                corrected, gate = self.context_adapter(
                    base, state, confidence, reset=reset_item
                )
            gates.append(gate)

            # Missing GOPs are decoded for concealment training but do not
            # overwrite the last reliable recurrent state.  A reset/cut does
            # replace state with the current GOP after exact adapter bypass.
            if state is None:
                state = corrected
            else:
                keep = valid_item.view(batch, 1, 1, 1, 1)
                state = torch.where(keep, corrected, state)

            if use_checkpoint and self.training:
                decoded = checkpoint(
                    self.model._tail, corrected, z_item, weight_item,
                    use_reentrant=False,
                )
            else:
                decoded = self.model._tail(corrected, z_item, weight_item)
            outputs.append(decoded)
        video = torch.cat(outputs, dim=2)
        gate_tensor = torch.stack(gates, dim=1)
        return (video, gate_tensor) if return_gates else video

    def forward(
        self,
        video: torch.Tensor,
        channel: nn.Module,
        *,
        reset: torch.Tensor | None = None,
        progress: float = 1.0,
        use_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        transmitted = self.encode_gops(
            video, reset=reset, use_checkpoint=use_checkpoint
        )
        if transmitted.shape[-1] != 2816:
            raise RuntimeError("V8 training must transmit exactly 2,816 values/GOP")
        received, weights, events = channel(transmitted, progress=progress)
        reconstructed, gates = self.decode_gops(
            received,
            weights,
            reset=reset,
            missing=events["missing"],
            use_checkpoint=use_checkpoint,
            return_gates=True,
        )
        return reconstructed, {
            **events,
            "transmitted": transmitted,
            "received": received,
            "weights": weights,
            "gates": gates,
        }
