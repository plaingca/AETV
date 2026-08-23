"""Small chunked video autoencoder sized for a single 16 GB GPU.

This is deliberately not an LTX-sized model. It borrows the useful idea of
spatiotemporal latents while keeping AETV's radio contract: bounded unit-RMS
coordinates and an explicit confidence tensor at the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _triple(value):
    return value if isinstance(value, tuple) else (value, value, value)


class CausalConv3d(nn.Module):
    """3D convolution with LTX-style first-frame temporal padding.

    Spatial axes are padded symmetrically; the time axis sees only the current
    and previous frames.  Repeating the first frame gives a stable temporal
    origin instead of introducing a black pre-roll.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1,
        causal: bool = True,
    ):
        super().__init__()
        kernel = _triple(kernel_size)
        stride = _triple(stride)
        self.temporal_pad = kernel[0] - 1 if causal else 0
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel, stride=stride,
            padding=(0 if causal else kernel[0] // 2, kernel[1] // 2, kernel[2] // 2),
        )

    def forward(self, x):
        if self.temporal_pad:
            first = x[:, :, :1].expand(-1, -1, self.temporal_pad, -1, -1)
            x = torch.cat([first, x], dim=2)
        return self.conv(x)


class PixelNorm3d(nn.Module):
    """Per-voxel channel RMS normalization; never mixes time positions."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps) * self.gain


def _feature_norm(channels: int, group_norm: bool):
    if not group_norm:
        return PixelNorm3d(channels)
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, causal: bool = True, group_norm: bool = False):
        super().__init__()
        self.norm1 = _feature_norm(channels, group_norm)
        self.conv1 = CausalConv3d(channels, channels, causal=causal)
        self.norm2 = _feature_norm(channels, group_norm)
        self.conv2 = CausalConv3d(channels, channels, causal=causal)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


class SpatialUpsample3d(nn.Module):
    """Learned 2x spatial upsampling without mixing temporal positions."""

    def __init__(
        self, in_channels: int, out_channels: int, resize_conv: bool = False,
        causal: bool = True, bilinear: bool = False,
    ):
        super().__init__()
        self.resize_conv = resize_conv
        self.bilinear = bilinear
        self.up = (
            CausalConv3d(in_channels, out_channels, (1, 3, 3), causal=causal)
            if resize_conv
            else nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=(1, 4, 4),
                stride=(1, 2, 2), padding=(0, 1, 1),
            )
        )

    def forward(self, x):
        if self.resize_conv:
            # Nearest 2x leaves a 2x2 block correlation that one 3x3 conv
            # cannot fully erase; chained through three stages that reads as
            # a 4-8 pixel block texture.  Trilinear with a unit temporal
            # factor is spatially bilinear and never mixes time positions.
            if self.bilinear:
                x = F.interpolate(
                    x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
                )
            else:
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.up(x)


class AxisAttention3d(nn.Module):
    """Factorized spatial then causal-temporal attention.

    LTX's convolutional VAE uses per-frame spatial attention and its diffusion
    decoder uses local 3D attention.  At our small bottleneck, axial attention
    is cheaper and gives every spatial position a causal temporal history.
    """

    def __init__(
        self, channels: int, head_dim: int = 32, causal: bool = True,
        group_norm: bool = False,
    ):
        super().__init__()
        if channels % head_dim:
            raise ValueError(f"attention channels {channels} must divide head_dim {head_dim}")
        self.heads = channels // head_dim
        self.head_dim = head_dim
        self.causal = causal
        self.spatial_norm = _feature_norm(channels, group_norm)
        self.spatial_qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.spatial_out = nn.Conv3d(channels, channels, 1)
        self.temporal_norm = _feature_norm(channels, group_norm)
        self.temporal_qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.temporal_out = nn.Conv3d(channels, channels, 1)

    def _attend(self, x, qkv_layer, out_layer, temporal: bool):
        b, c, t, h, w = x.shape
        q, k, v = qkv_layer(x).chunk(3, dim=1)
        if temporal:
            # One temporal sequence per spatial location.
            def pack(y):
                return y.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.heads, self.head_dim, t).transpose(-1, -2)
            unpack_shape = (b, h, w, self.heads, t, self.head_dim)
            out = F.scaled_dot_product_attention(
                pack(q), pack(k), pack(v), is_causal=self.causal
            )
            out = out.reshape(unpack_shape).permute(0, 3, 5, 4, 1, 2).reshape(b, c, t, h, w)
        else:
            # LTX-style per-frame spatial attention.
            def pack(y):
                return y.permute(0, 2, 1, 3, 4).reshape(b * t, self.heads, self.head_dim, h * w).transpose(-1, -2)
            out = F.scaled_dot_product_attention(pack(q), pack(k), pack(v))
            out = out.transpose(-1, -2).reshape(b, t, self.heads, self.head_dim, h, w)
            out = out.permute(0, 2, 3, 1, 4, 5).reshape(b, c, t, h, w)
        return out_layer(out)

    def forward(self, x):
        x = x + self._attend(self.spatial_norm(x), self.spatial_qkv, self.spatial_out, False)
        return x + self._attend(self.temporal_norm(x), self.temporal_qkv, self.temporal_out, True)


class VideoEncoder(nn.Module):
    def __init__(
        self, width: int = 48, latent_channels: int = 12, compact: bool = False,
        preserve_time: bool = False, causal: bool = True,
        group_norm: bool = False,         clip_rms_latents: bool = False,
        deep: bool = False, deep2: bool = False, deep3: bool = False,
    ):
        super().__init__()
        self.clip_rms_latents = clip_rms_latents
        self.deep = deep
        self.deep2 = deep2
        self.deep3 = deep3
        if (deep or deep2 or deep3) and compact:
            raise ValueError("deep encoder is only supported in non-compact mode")
        if deep2 and not deep:
            raise ValueError("deep2 encoder requires deep")
        if deep3 and not deep2:
            raise ValueError("deep3 encoder requires deep2")
        # Two extra residual blocks per scale, held outside self.net so the
        # Sequential's parameter indices — and therefore existing checkpoint
        # keys — do not shift.  The forward interleaves them by slicing.
        if deep:
            self.e0a = ResBlock3d(width // 2, causal, group_norm)
            self.e0b = ResBlock3d(width // 2, causal, group_norm)
            self.e1a = ResBlock3d(width, causal, group_norm)
            self.e1b = ResBlock3d(width, causal, group_norm)
            self.e2a = ResBlock3d(width * 2, causal, group_norm)
            self.e2b = ResBlock3d(width * 2, causal, group_norm)
        if deep2:
            self.e0c = ResBlock3d(width // 2, causal, group_norm)
            self.e1c = ResBlock3d(width, causal, group_norm)
            self.e2c = ResBlock3d(width * 2, causal, group_norm)
        # Fourth tier is attention-weighted: representation quality gains at
        # this point come from context, not from more local convolution.
        if deep3:
            self.e1d = ResBlock3d(width, causal, group_norm)
            self.e_attn1 = AxisAttention3d(width, causal=causal, group_norm=group_norm)
            self.e2d = ResBlock3d(width * 2, causal, group_norm)
        if compact:
            self.net = nn.Sequential(
                CausalConv3d(3, width // 2, (3, 5, 5), stride=(1, 2, 2), causal=causal),
                nn.SiLU(), ResBlock3d(width // 2, causal, group_norm),
                # Compact mode spends its reduced spatial/channel budget on a
                # latent slice for every source frame. Temporal compression
                # repeatedly converged to a static local optimum in overfit
                # gates, even under strongly normalized motion supervision.
                CausalConv3d(width // 2, width, 3, stride=(1, 2, 2), causal=causal),
                nn.SiLU(), ResBlock3d(width, causal, group_norm), ResBlock3d(width, causal, group_norm),
                # Rate reduction comes from space/channels, not time.
                CausalConv3d(width, width * 2, 3, stride=(1, 2, 2), causal=causal),
                nn.SiLU(), ResBlock3d(width * 2, causal, group_norm), ResBlock3d(width * 2, causal, group_norm),
                CausalConv3d(width * 2, width * 2, 3, stride=(1, 2, 2), causal=causal),
                nn.SiLU(), ResBlock3d(width * 2, causal, group_norm),
                AxisAttention3d(width * 2, causal=causal, group_norm=group_norm),
                CausalConv3d(width * 2, latent_channels, 3, causal=causal),
            )
        else:
            temporal_stride = 1 if preserve_time else 2
            self.net = nn.Sequential(
                CausalConv3d(3, width // 2, (3, 5, 5), stride=(1, 2, 2), causal=causal),
                nn.SiLU(), ResBlock3d(width // 2, causal, group_norm),
                CausalConv3d(
                    width // 2, width, 3, stride=(temporal_stride, 2, 2),
                    causal=causal,
                ),
                nn.SiLU(), ResBlock3d(width, causal, group_norm), ResBlock3d(width, causal, group_norm),
                # Preserve five temporal slices from a nine-frame clip.  The
                # compact rate-compatible model restores three slices but adds
                # another spatial reduction plus a stronger bottleneck stack.
                CausalConv3d(width, width * 2, 3, stride=(1, 2, 2), causal=causal),
                nn.SiLU(), ResBlock3d(width * 2, causal, group_norm), ResBlock3d(width * 2, causal, group_norm),
                AxisAttention3d(width * 2, causal=causal, group_norm=group_norm),
                CausalConv3d(width * 2, latent_channels, 3, causal=causal),
            )

    def forward(self, video):
        x = video * 2 - 1
        if self.deep:
            # net indices: [0:3] stem+res at 1/2, [3:7] down+res at 1/4,
            # [7:12] down+res+attention at 1/8, [12] latent projection.
            x = self.e0b(self.e0a(self.net[0:3](x)))
            if self.deep2:
                x = self.e0c(x)
            x = self.e1b(self.e1a(self.net[3:7](x)))
            if self.deep2:
                x = self.e1c(x)
            if self.deep3:
                x = self.e_attn1(self.e1d(x))
            x = self.e2b(self.e2a(self.net[7:12](x)))
            if self.deep2:
                x = self.e2c(x)
            if self.deep3:
                x = self.e2d(x)
            z = torch.tanh(self.net[12](x))
        else:
            z = torch.tanh(self.net(x))
        if self.clip_rms_latents:
            rms = z.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            rms = rms[:, :, None, None, None]
        else:
            # Streaming-causal mode cannot leak future energy into an earlier slice.
            rms = z.pow(2).mean(dim=(1, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        return z / rms


class VideoDecoder(nn.Module):
    def __init__(
        self, width: int = 48, latent_channels: int = 12, compact: bool = False,
        resize_conv_upsampling: bool = False, causal: bool = True,
        group_norm: bool = False, smooth_temporal_skip: bool = False,
        bilinear_upsampling: bool = False, deep_tail: bool = False,
        deeper: bool = False, deepest: bool = False, deep4: bool = False,
    ):
        super().__init__()
        self.compact = compact
        self.smooth_temporal_skip = smooth_temporal_skip
        self.deep_tail = deep_tail
        self.deeper = deeper
        self.deepest = deepest
        self.deep4 = deep4
        if deepest and not deeper:
            raise ValueError("deepest decoder tier requires deeper")
        if deep4 and not deepest:
            raise ValueError("deep4 decoder tier requires deepest")
        # A short path from transmitted time slices to output logits prevents
        # the deep spatial decoder from settling into a temporally constant
        # solution. The full decoder still supplies nearly all spatial detail.
        self.temporal_skip = nn.Conv3d(latent_channels, 3, 1)
        self.input = CausalConv3d(2 * latent_channels, width * 2, 3, causal=causal)
        self.r0 = ResBlock3d(width * 2, causal, group_norm)
        self.r0b = ResBlock3d(width * 2, causal, group_norm)
        self.attn = AxisAttention3d(
            width * 2, causal=causal, group_norm=group_norm
        )
        if compact:
            self.up0 = SpatialUpsample3d(
                width * 2, width * 2, resize_conv_upsampling, causal,
                bilinear_upsampling,
            )
            self.r0c = ResBlock3d(width * 2, causal, group_norm)
        self.up1 = SpatialUpsample3d(
            width * 2, width, resize_conv_upsampling, causal, bilinear_upsampling
        )
        self.r1 = ResBlock3d(width, causal, group_norm)
        self.r1b = ResBlock3d(width, causal, group_norm)
        self.up2 = SpatialUpsample3d(
            width, width // 2, resize_conv_upsampling, causal, bilinear_upsampling
        )
        self.r2 = ResBlock3d(width // 2, causal, group_norm)
        self.r2b = ResBlock3d(width // 2, causal, group_norm)
        self.up3 = SpatialUpsample3d(
            width // 2, width // 4, resize_conv_upsampling, causal,
            bilinear_upsampling,
        )
        # The stock decoder is two convolutions deep at full resolution, so
        # fine detail has to be synthesized by the output conv alone.  The
        # deep tail adds one half-resolution and two full-resolution residual
        # blocks — cheap at 96x72 — where detail and block-texture repair
        # actually happen.  Transplanting from a shallow checkpoint zero
        # initializes each block's second conv, so the loaded function is
        # exactly the shallow one.
        if deep_tail:
            self.r2c = ResBlock3d(width // 2, causal, group_norm)
            self.r3 = ResBlock3d(width // 4, causal, group_norm)
            self.r3b = ResBlock3d(width // 4, causal, group_norm)
        # Second depth tier: two extra blocks at the bottleneck and quarter
        # scales (where robust latent decoding happens) plus one more at half
        # and two more at full resolution (where detail is synthesized).
        if deeper:
            self.m0a = ResBlock3d(width * 2, causal, group_norm)
            self.m0b = ResBlock3d(width * 2, causal, group_norm)
            self.m1a = ResBlock3d(width, causal, group_norm)
            self.m1b = ResBlock3d(width, causal, group_norm)
            self.r2d = ResBlock3d(width // 2, causal, group_norm)
            self.r3c = ResBlock3d(width // 4, causal, group_norm)
            self.r3d = ResBlock3d(width // 4, causal, group_norm)
        # Third depth tier: one more block at bottleneck and quarter scale,
        # axial attention at quarter scale (residual, so it transplants as
        # identity with zeroed output projections), one more at half and two
        # more at full resolution.
        if deepest:
            self.m0c = ResBlock3d(width * 2, causal, group_norm)
            self.m1c = ResBlock3d(width, causal, group_norm)
            self.attn1 = AxisAttention3d(width, causal=causal, group_norm=group_norm)
            self.r2e = ResBlock3d(width // 2, causal, group_norm)
            self.r3e = ResBlock3d(width // 4, causal, group_norm)
            self.r3f = ResBlock3d(width // 4, causal, group_norm)
        # Fourth tier: attention-weighted (half-scale attention gives the
        # detail-synthesis stages spatial context), plus one bottleneck block
        # and two more at full resolution.
        if deep4:
            self.m0d = ResBlock3d(width * 2, causal, group_norm)
            self.attn2 = AxisAttention3d(
                width // 2, causal=causal, group_norm=group_norm
            )
            self.r3g = ResBlock3d(width // 4, causal, group_norm)
            self.r3h = ResBlock3d(width // 4, causal, group_norm)
        self.output = CausalConv3d(width // 4, 3, 3, causal=causal)

    def forward(self, z, weights, output_shape: tuple[int, int, int]):
        frames, height, width = output_shape
        # Nearest upsampling here paints literal 8x8 blocks into the output
        # logits with no convolution after them; trilinear removes the block
        # grid at zero parameter cost.  Kept switchable because deployed
        # checkpoints were trained against the nearest-neighbor function.
        if self.smooth_temporal_skip:
            temporal_skip = F.interpolate(
                self.temporal_skip(z * weights), size=output_shape,
                mode="trilinear", align_corners=False,
            )
        else:
            temporal_skip = F.interpolate(
                self.temporal_skip(z * weights), size=output_shape, mode="nearest"
            )
        x = self.attn(self.r0b(self.r0(self.input(torch.cat([z * weights, weights], dim=1)))))
        if self.deeper:
            x = self.m0b(self.m0a(x))
        if self.deepest:
            x = self.m0c(x)
        if self.deep4:
            x = self.m0d(x)
        if self.compact:
            x = self.r0c(self.up0(x))
        x = self.r1b(self.r1(self.up1(x)))
        if self.deeper:
            x = self.m1b(self.m1a(x))
        if self.deepest:
            x = self.attn1(self.m1c(x))
        # Temporal interpolation is causal under the decoder's prefix test:
        # frame zero maps exactly to latent slice zero.  Learned operations on
        # either side remain causal convolutions.
        x = F.interpolate(x, size=(frames, height // 4, width // 4), mode="nearest")
        x = self.r2b(self.r2(self.up2(x)))
        if self.deep_tail:
            x = self.r2c(x)
        if self.deeper:
            x = self.r2d(x)
        if self.deepest:
            x = self.r2e(x)
        if self.deep4:
            x = self.attn2(x)
        x = F.silu(self.up3(x))
        if self.deep_tail:
            x = self.r3b(self.r3(x))
        if self.deeper:
            x = self.r3d(self.r3c(x))
        if self.deepest:
            x = self.r3f(self.r3e(x))
        if self.deep4:
            x = self.r3h(self.r3g(x))
        return torch.sigmoid(self.output(x) + temporal_skip)


class VideoAutoencoder(nn.Module):
    def __init__(
        self, width: int = 48, latent_channels: int = 12, compact: bool = False,
        resize_conv_upsampling: bool = False, preserve_time: bool = False,
        causal: bool = True, group_norm: bool = False,
        clip_rms_latents: bool = False, smooth_temporal_skip: bool = False,
        bilinear_upsampling: bool = False, deep_decoder: bool = False,
        deep_encoder: bool = False, deeper_decoder: bool = False,
        deep3: bool = False, deep4: bool = False,
    ):
        super().__init__()
        if latent_channels % 3:
            raise ValueError("latent_channels must split into three progressive groups")
        self.encoder = VideoEncoder(
            width, latent_channels, compact, preserve_time, causal,
            group_norm, clip_rms_latents, deep_encoder, deep3, deep4,
        )
        self.decoder = VideoDecoder(
            width, latent_channels, compact, resize_conv_upsampling,
            causal, group_norm, smooth_temporal_skip,
            bilinear_upsampling, deep_decoder, deeper_decoder, deep3, deep4,
        )
        self.latent_channels = latent_channels
        self.compact = compact

    def forward(self, video, weights=None):
        z = self.encoder(video)
        if weights is None:
            weights = torch.ones_like(z)
        return self.decoder(z, weights, tuple(video.shape[-3:]))
