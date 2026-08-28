"""Receiver-side DCVC-TCM for independently encoded V8 GOPs.

The released encoder still emits one 2,816-value vector per six-frame GOP.
There is no transmitter previous-frame conditioner.  The decoder estimates a
real RGB flow field with frozen RAFT-Small, warps the previous GOP's last
full-resolution pre-RGB feature, and residual-fuses it before the synthesis
tail.  A photometric/confidence/reset gate starts closed on cuts; zero-init
keeps construction an exact stock decode.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .models import AETVAutoencoder
from .optical_flow import RAFTAligner


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class _FeatureResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        return value + self.conv2(F.silu(self.norm2(hidden)))


def photometric_reliability(
    warped: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.10,
    softness: float = 0.02,
    scene_multiplier: float = 1.5,
) -> torch.Tensor:
    """Soft occlusion/cut gate from warped-previous vs current-first RGB."""
    error = (warped - target).abs().mean(dim=1, keepdim=True)
    pixel = torch.sigmoid((threshold - error) / softness)
    frame = error.mean(dim=(2, 3), keepdim=True)
    scene = torch.sigmoid((threshold * scene_multiplier - frame) / softness)
    return pixel * scene


class FeatureTCMFuser(nn.Module):
    """Zero-init residual fuse of current and RAFT-aligned previous features."""

    def __init__(
        self,
        feature_channels: int,
        *,
        width: int = 64,
        blocks: int = 6,
        spatial_scale: int = 2,
        max_residual: float = 1.0,
        taper: tuple[float, ...] = (1.0, 0.7, 0.35, 0.1, 0.0, 0.0),
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.width = width
        self.blocks = blocks
        self.spatial_scale = spatial_scale
        self.max_residual = max_residual
        self.taper = tuple(taper)
        self.input = nn.Conv3d(3 * feature_channels + 1, width, 3, padding=1)
        self.body = nn.Sequential(
            *(_FeatureResidualBlock(width) for _ in range(blocks))
        )
        self.output = nn.Conv3d(width, feature_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def config(self) -> dict:
        return {
            "feature_channels": self.feature_channels,
            "width": self.width,
            "blocks": self.blocks,
            "spatial_scale": self.spatial_scale,
            "max_residual": self.max_residual,
            "taper": self.taper,
        }

    def forward(
        self,
        current: torch.Tensor,
        aligned_previous: torch.Tensor,
        reliability: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if current.shape != aligned_previous.shape or current.ndim != 5:
            raise ValueError("current and aligned features must have equal BCTHW shapes")
        batch, channels, frames, height, width = current.shape
        if channels != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels")
        if reliability.shape[0] != batch:
            raise ValueError("reliability batch must match features")
        if reliability.ndim == 4:
            reliability = reliability.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        if reliability.shape[1] != 1:
            raise ValueError("reliability must be a single-channel map")
        if reliability.shape[-2:] != (height, width):
            reliability = F.interpolate(
                reliability,
                size=(frames, height, width),
                mode="trilinear",
                align_corners=False,
            )
        features = torch.cat(
            (current, aligned_previous, current - aligned_previous, reliability),
            dim=1,
        )
        low_height = max(1, (height + self.spatial_scale - 1) // self.spatial_scale)
        low_width = max(1, (width + self.spatial_scale - 1) // self.spatial_scale)
        features = F.interpolate(
            features,
            size=(frames, low_height, low_width),
            mode="trilinear",
            align_corners=False,
        )
        residual = self.output(self.body(F.silu(self.input(features))))
        residual = F.interpolate(
            residual, size=(frames, height, width), mode="trilinear", align_corners=False
        )
        taper = current.new_tensor(self.taper)
        if taper.numel() != frames:
            taper = F.interpolate(
                taper.view(1, 1, -1), size=frames, mode="linear", align_corners=False
            ).flatten()
        gate = confidence.to(current.dtype).reshape(batch, 1, 1, 1, 1).clamp(0, 1)
        return current + self.max_residual * torch.tanh(residual) * taper.view(
            1, 1, frames, 1, 1
        ) * gate


class V8RecurrentFeatureTCM(nn.Module):
    """Frozen released encoder plus RAFT-aligned full-resolution feature TCM."""

    checkpoint_kind = "aetv-v8-recurrent-feature-tcm"

    def __init__(
        self,
        base: AETVAutoencoder,
        *,
        fuser: FeatureTCMFuser | None = None,
        aligner: nn.Module | None = None,
        photometric_threshold: float = 0.10,
        photometric_softness: float = 0.02,
        scene_multiplier: float = 1.5,
    ):
        super().__init__()
        if base.mode.name != "V8" or base.encoder.latent_channels != 3:
            raise ValueError("feature TCM requires released-layout V8")
        if base.decoder.compact:
            raise ValueError("the released-layout decoder must not be compact")
        self.mode = base.mode
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.latent_budget = self.mode.latents_per_gop
        feature_channels = int(self.decoder.decoder.output.conv.in_channels)
        self.fuser = fuser or FeatureTCMFuser(feature_channels)
        if self.fuser.feature_channels != feature_channels:
            raise ValueError("fuser feature channels must match the decoder tail")
        if aligner is not None:
            self.aligner = aligner
        else:
            self.aligner = RAFTAligner("cpu")
        for parameter in self.aligner.parameters():
            parameter.requires_grad_(False)
        self.photometric_threshold = float(photometric_threshold)
        self.photometric_softness = float(photometric_softness)
        self.scene_multiplier = float(scene_multiplier)

    @classmethod
    def from_released(
        cls,
        checkpoint_path: str | Path = "models/v8-hf3k-face-gan.pt",
        *,
        aligner: nn.Module | None = None,
    ) -> "V8RecurrentFeatureTCM":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        return cls(base, aligner=aligner)

    @property
    def context_adapter(self) -> FeatureTCMFuser:
        return self.fuser

    def set_trainable_contract(self) -> dict[str, int]:
        """Freeze the released encoder; train the fuser and full-res tail."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.fuser.parameters():
            parameter.requires_grad_(True)
        inner = self.decoder.decoder
        tail_names = (
            "temporal_skip",
            "r3",
            "r3b",
            "r3c",
            "r3d",
            "r3e",
            "r3f",
            "r3g",
            "r3h",
            "output",
        )
        for name in tail_names:
            module = getattr(inner, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for parameter in self.aligner.parameters():
            parameter.requires_grad_(False)
        counts = {
            "encoder": 0,
            "encoder_context": 0,
            "state": sum(p.numel() for p in self.fuser.parameters() if p.requires_grad),
            "decoder_tail": sum(
                p.numel() for p in self.decoder.parameters() if p.requires_grad
            ),
        }
        if counts["state"] <= 0 or counts["decoder_tail"] <= 0:
            raise RuntimeError(f"incomplete trainable contract: {counts}")
        return counts

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
        items = []
        for index in range(count):
            gop = video[:, :, index * frames : (index + 1) * frames]
            if use_checkpoint and self.training:
                encoded = checkpoint(self.encoder, gop, use_reentrant=False)
            else:
                encoded = self.encoder(gop)
            if encoded.shape[-1] != self.latent_budget:
                raise RuntimeError("encoder violated the 2,816-value V8 wire budget")
            items.append(encoded)
        return torch.stack(items, dim=1)

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

    def _to_up3(
        self, z: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        hidden = inner.input(torch.cat((z * weights, weights), dim=1))
        hidden = inner.attn(inner.r0b(inner.r0(hidden)))
        if inner.deeper:
            hidden = inner.m0b(inner.m0a(hidden))
        if inner.deepest:
            hidden = inner.m0c(hidden)
        if inner.deep4:
            hidden = inner.m0d(hidden)
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
        return hidden, temporal_skip

    def _from_up3(self, hidden: torch.Tensor, temporal_skip: torch.Tensor) -> torch.Tensor:
        inner = self.decoder.decoder
        if inner.deep_tail:
            hidden = inner.r3b(inner.r3(hidden))
        if inner.deeper:
            hidden = inner.r3d(inner.r3c(hidden))
        if inner.deepest:
            hidden = inner.r3f(inner.r3e(hidden))
        if inner.deep4:
            hidden = inner.r3h(inner.r3g(hidden))
        return torch.sigmoid(inner.output(hidden) + temporal_skip)

    def _align_previous(
        self,
        previous_rgb: torch.Tensor,
        current_first: torch.Tensor,
        previous_feature: torch.Tensor,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp previous last RGB/feature into the current first-frame grid."""
        device_type = previous_rgb.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            reference = previous_rgb.float()
            target = current_first.float()
            with torch.no_grad():
                flow = self.aligner.estimate_flow(reference, target)
            warped_rgb = self.aligner.warp_with_flow(reference, flow)
            warped_feature = self.aligner.warp_with_flow(previous_feature.float(), flow)
        warped_rgb = warped_rgb.to(dtype=current_first.dtype)
        warped_feature = warped_feature.to(dtype=previous_feature.dtype)
        warped_feature = warped_feature.unsqueeze(2).expand(
            -1, -1, frames, -1, -1
        ).contiguous()
        return warped_rgb, warped_feature

    def decode_gops(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        reset: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        use_checkpoint: bool = False,
        return_gates: bool = False,
        use_context: bool = True,
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

        z, confidence_grid = self._latent_grids(latents, weights)
        frames = self.mode.gop_frames
        outputs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        state_rgb: torch.Tensor | None = None
        state_feature: torch.Tensor | None = None
        for index in range(count):
            z_item = z[:, index]
            weight_item = confidence_grid[:, index]
            if use_checkpoint and self.training:
                hidden, skip = checkpoint(
                    self._to_up3, z_item, weight_item, use_reentrant=False
                )
            else:
                hidden, skip = self._to_up3(z_item, weight_item)

            reset_item = reset[:, index]
            valid_item = ~missing[:, index]
            gate = latents.new_zeros(batch)
            fused = hidden
            use_aligned = (
                use_context
                and state_rgb is not None
                and state_feature is not None
                and not bool(reset_item.all().item())
            )
            if use_aligned:
                with torch.no_grad():
                    stock_first = self._from_up3(hidden, skip)[:, :, 0]
                    warped_rgb, warped_feature = self._align_previous(
                        state_rgb.detach(),
                        stock_first,
                        state_feature.detach(),
                        frames,
                    )
                    reliability = photometric_reliability(
                        warped_rgb,
                        stock_first,
                        threshold=self.photometric_threshold,
                        softness=self.photometric_softness,
                        scene_multiplier=self.scene_multiplier,
                    )
                live = (~reset_item).to(dtype=hidden.dtype)
                confidence = weights[:, index].mean(dim=-1) * live
                fused = self.fuser(hidden, warped_feature, reliability, confidence)
                if bool(reset_item.any().item()):
                    fused = torch.where(
                        reset_item.view(batch, 1, 1, 1, 1), hidden, fused
                    )
                gate = reliability.mean(dim=(1, 2, 3)) * confidence.to(reliability.dtype)
            if use_checkpoint and self.training:
                decoded = checkpoint(
                    self._from_up3, fused, skip, use_reentrant=False
                )
            else:
                decoded = self._from_up3(fused, skip)
            gates.append(gate)

            if state_rgb is None:
                state_rgb = decoded[:, :, -1]
                state_feature = fused[:, :, -1]
            else:
                keep = valid_item.view(batch, 1, 1, 1)
                state_rgb = torch.where(keep, decoded[:, :, -1], state_rgb)
                state_feature = torch.where(keep, fused[:, :, -1], state_feature)
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
