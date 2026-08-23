"""Spatiotemporal GOP video autoencoders for AETV modes V0-V7.

Uses spatiotemporal 3D ResNets, axial attention, and smooth temporal skips,
mapping between video GOPs (B, 3, T, H, W) and 1D analog latents at exact
on-air GOP budgets (1,472 for N-band, 2,816 for W-band).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_backbone import VideoDecoder, VideoEncoder
from .config import AETV_MODES, AETVModeSpec, LATENTS_PER_GOP_N, LATENTS_PER_GOP_W


class AETVEncoder(nn.Module):
    """Spatiotemporal video encoder producing exact GOP latent budgets."""

    def __init__(
        self,
        mode: AETVModeSpec | str = "V1",
        width: int = 192,
        latent_channels: int = 12,
        compact: bool = False,
        preserve_time: bool = False,
        causal: bool = False,
        group_norm: bool = True,
        clip_rms_latents: bool = True,
        deep: bool = True,
        deep2: bool = True,
        deep3: bool = True,
    ):
        super().__init__()
        if isinstance(mode, str):
            self.mode = AETV_MODES[mode]
        else:
            self.mode = mode

        self.latent_budget = self.mode.latents_per_gop
        self.latent_channels = latent_channels
        is_deep = deep and (width >= 64)
        is_deep2 = deep2 and (width >= 64)
        is_deep3 = deep3 and (width >= 64)
        self.encoder = VideoEncoder(
            width=width,
            latent_channels=latent_channels,
            compact=compact,
            preserve_time=preserve_time,
            causal=causal or self.mode.causal,
            group_norm=group_norm,
            clip_rms_latents=clip_rms_latents,
            deep=is_deep,
            deep2=is_deep2,
            deep3=is_deep3,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, T, H, W) -> (B, latent_budget)."""
        z_grid = self.encoder(video)  # (B, C, T', H', W')
        b = z_grid.shape[0]
        flat_z = z_grid.reshape(b, -1)
        if flat_z.shape[1] >= self.latent_budget:
            out_z = flat_z[:, : self.latent_budget]
        else:
            # Pad with zeros if latent grid is smaller than budget
            pad = torch.zeros(b, self.latent_budget - flat_z.shape[1], device=video.device, dtype=flat_z.dtype)
            out_z = torch.cat([flat_z, pad], dim=1)

        # Whole-GOP unit-RMS normalization
        rms = torch.sqrt(torch.mean(out_z**2, dim=1, keepdim=True) + 1e-6)
        return out_z / rms


class AETVDecoder(nn.Module):
    """Spatiotemporal video decoder reconstructing video from latents & weights."""

    def __init__(
        self,
        mode: AETVModeSpec | str = "V1",
        width: int = 192,
        latent_channels: int = 12,
        compact: bool = False,
        resize_conv_upsampling: bool = True,
        causal: bool = False,
        group_norm: bool = True,
        smooth_temporal_skip: bool = True,
        bilinear_upsampling: bool = False,
        deep_tail: bool = True,
        deeper: bool = True,
        deepest: bool = True,
        deep4: bool = True,
    ):
        super().__init__()
        if isinstance(mode, str):
            self.mode = AETV_MODES[mode]
        else:
            self.mode = mode

        self.latent_budget = self.mode.latents_per_gop
        self.latent_channels = latent_channels
        is_deep_tail = deep_tail and (width >= 64)
        is_deeper = deeper and (width >= 64)
        is_deepest = deepest and (width >= 64)
        is_deep4 = deep4 and (width >= 64)
        self.decoder = VideoDecoder(
            width=width,
            latent_channels=latent_channels,
            compact=compact,
            resize_conv_upsampling=resize_conv_upsampling,
            causal=causal or self.mode.causal,
            group_norm=group_norm,
            smooth_temporal_skip=smooth_temporal_skip,
            bilinear_upsampling=bilinear_upsampling,
            deep_tail=is_deep_tail,
            deeper=is_deeper,
            deepest=is_deepest,
            deep4=is_deep4,
        )

    def _get_grid_shape(self, output_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        frames, height, width = output_shape
        # Spatial downsample is 8x, temporal downsample is 2x
        t_latent = max(1, math.ceil(frames / 2.0))
        h_latent = max(1, height // 8)
        w_latent = max(1, width // 8)
        return (t_latent, h_latent, w_latent)

    def forward(
        self,
        latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        output_shape: Tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Decode (B, latent_budget) -> (B, 3, T, H, W)."""
        if output_shape is None:
            output_shape = (self.mode.gop_frames, self.mode.height, self.mode.width)

        t_lat, h_lat, w_lat = self._get_grid_shape(output_shape)
        total_grid_elements = self.latent_channels * t_lat * h_lat * w_lat
        b = latents.shape[0]

        # Allocate full grid
        z_grid_flat = torch.zeros(b, total_grid_elements, device=latents.device, dtype=latents.dtype)
        if weights is not None:
            w_grid_flat = torch.zeros(b, total_grid_elements, device=weights.device, dtype=weights.dtype)
        else:
            w_grid_flat = torch.zeros(b, total_grid_elements, device=latents.device, dtype=latents.dtype)

        copy_len = min(latents.shape[1], total_grid_elements)
        z_grid_flat[:, :copy_len] = latents[:, :copy_len]
        if weights is not None:
            w_grid_flat[:, :copy_len] = weights[:, :copy_len]
        else:
            w_grid_flat[:, :copy_len] = 1.0

        z_grid = z_grid_flat.reshape(b, self.latent_channels, t_lat, h_lat, w_lat)
        w_grid = w_grid_flat.reshape(b, self.latent_channels, t_lat, h_lat, w_lat)

        return self.decoder(z_grid, w_grid, output_shape)


class AETVAutoencoder(nn.Module):
    """Full end-to-end spatiotemporal AETV Autoencoder."""

    def __init__(
        self,
        mode: AETVModeSpec | str = "V1",
        width: int = 192,
        latent_channels: int = 12,
        compact: bool = False,
        resize_conv_upsampling: bool = True,
        preserve_time: bool = False,
        causal: bool = False,
        group_norm: bool = True,
        clip_rms_latents: bool = True,
        smooth_temporal_skip: bool = True,
        bilinear_upsampling: bool = True,
        deep: bool = True,
        deep2: bool = True,
        deep3: bool = True,
        deep_tail: bool = True,
        deeper: bool = True,
        deepest: bool = True,
        deep4: bool | None = None,
    ):
        super().__init__()
        if isinstance(mode, str):
            self.mode = AETV_MODES[mode]
        else:
            self.mode = mode

        if deep4 is None:
            deep4 = (self.mode.width * self.mode.height <= 96 * 72)

        self.encoder = AETVEncoder(
            mode=self.mode,
            width=width,
            latent_channels=latent_channels,
            compact=compact,
            preserve_time=preserve_time,
            causal=causal,
            group_norm=group_norm,
            clip_rms_latents=clip_rms_latents,
            deep=deep,
            deep2=deep2,
            deep3=deep3,
        )
        self.decoder = AETVDecoder(
            mode=self.mode,
            width=width,
            latent_channels=latent_channels,
            compact=compact,
            resize_conv_upsampling=resize_conv_upsampling,
            causal=causal,
            group_norm=group_norm,
            smooth_temporal_skip=smooth_temporal_skip,
            bilinear_upsampling=bilinear_upsampling,
            deep_tail=deep_tail,
            deeper=deeper,
            deepest=deepest,
            deep4=deep4,
        )


    def load_pretrained_weights(self, checkpoint_path: str, device: torch.device | str = "cpu"):
        """Load pretrained weights from either native AETV or SSTVAE checkpoint dictionary."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

        # Check if direct state dict matches self
        incompatible = self.load_state_dict(state_dict, strict=False)
        if len(incompatible.missing_keys) == 0:
            print(f"Loaded exact matching native AETV weights from {checkpoint_path}")
            return

        # Otherwise extract encoder and decoder submodules from SSTVAE checkpoint
        enc_state = {}
        dec_state = {}
        for k, v in state_dict.items():
            clean_k = k
            while clean_k.startswith("encoder."):
                clean_k = clean_k[len("encoder."):]
            if k.startswith("encoder."):
                enc_state[clean_k] = v

            clean_k = k
            while clean_k.startswith("decoder."):
                clean_k = clean_k[len("decoder."):]
            if k.startswith("decoder."):
                dec_state[clean_k] = v

        if enc_state:
            self.encoder.encoder.load_state_dict(enc_state, strict=False)
        if dec_state:
            self.decoder.decoder.load_state_dict(dec_state, strict=False)
        print(f"Loaded pretrained backbone weights from {checkpoint_path}")

    def forward(
        self,
        video: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encoder(video)
        return self.decoder(z, weights, (video.shape[2], video.shape[3], video.shape[4]))


def _unit_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale each spatial location's feature vector to unit length across channels."""
    return x / x.pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()


class MultiLayerVGGPerceptualLoss(nn.Module):
    """LPIPS-style multi-layer VGG16 perceptual loss, applied frame-wise to video.

    Five layer groups (relu1_2 .. relu5_3) span edges through semantics; the deep
    groups are what supply plausible texture instead of blur at high compression.
    Features are unit-normalized across channels before the squared distance, so
    the value is not dominated by whichever channels happen to have the largest
    activations, and layers contribute comparably regardless of their scale.

    Temporal supervision is deliberately not part of this module. Frame
    differences are far off VGG's input distribution, so features taken from
    them are weak and nearly content-independent; the 3D DWT loss in the trainer
    covers temporal high frequencies on a basis that is valid for them.
    """

    LAYER_SLICES = ((0, 4), (4, 9), (9, 16), (16, 23), (23, 30))

    def __init__(self, layer_weights: tuple[float, ...] | None = None):
        super().__init__()
        import torchvision.models as tv_models

        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleList(
            nn.Sequential(*[vgg[i] for i in range(a, b)]) for a, b in self.LAYER_SLICES
        )
        for param in self.parameters():
            param.requires_grad = False
        super().train(False)

        weights = layer_weights or (1.0,) * len(self.LAYER_SLICES)
        if len(weights) != len(self.LAYER_SLICES):
            raise ValueError(
                f"expected {len(self.LAYER_SLICES)} layer weights, got {len(weights)}"
            )
        self.register_buffer("layer_weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # A frozen feature extractor must stay in eval even when the parent trains.
        return super().train(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 3, T, H, W) in [0, 1]
        b, c, t, h, w = pred.shape
        p = (pred.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w) - self.mean) / self.std
        q = (target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w) - self.mean) / self.std

        total = pred.new_zeros(())
        for i, layer in enumerate(self.slices):
            p = layer(p)
            q = layer(q)
            diff = _unit_normalize(p) - _unit_normalize(q)
            total = total + self.layer_weights[i] * diff.pow(2).sum(dim=1).mean()
        return total


# Aliases. `ShallowVGGPerceptualLoss` is the historical name and is what
# aetv/__init__.py exports; the loss is no longer shallow.
SpatioTemporalVideoPerceptualLoss = MultiLayerVGGPerceptualLoss
ShallowVGGPerceptualLoss = MultiLayerVGGPerceptualLoss
VideoPerceptualLoss = MultiLayerVGGPerceptualLoss



class SpatioTemporalPatchGAN3D(nn.Module):
    """
    Local 3D Spatio-Temporal PatchGAN Discriminator with Spectral Normalization.
    Receptive field is strictly confined to local ~32x32 pixel patches and 3 temporal frames.
    Extracts intermediate multi-scale feature representations for Feature Matching Loss (L_FM).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Layer 1: Stride 2 spatial, Stride 1 temporal (RF: 4x4)
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_channels, base_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Layer 2: Stride 2 spatial, Stride 1 temporal (RF: 10x10)
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Layer 3: Stride 2 spatial, Stride 2 temporal (RF: 22x22)
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Final head: Local 1x1x1 conv projecting to patch logits (RF remains ~22x22 to 32x32)
        self.head = nn.utils.spectral_norm(
            nn.Conv3d(base_channels * 4, 1, kernel_size=(1, 1, 1), stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Normalize input [0, 1] to [-1, 1]
        x_norm = 2.0 * x - 1.0
        h1 = self.conv1(x_norm)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        logits = self.head(h3)
        return logits, [h1, h2, h3]


# Backwards compatibility alias
SpatioTemporalDiscriminator3D = SpatioTemporalPatchGAN3D

