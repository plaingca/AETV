"""Production 2.2 kHz RF Video Codec (Mode AC6).

State-of-the-Art Architecture integrating arXiv:2601.01729 (RVDJSCC):
"Robust Deep Joint Source-Channel Coding for Video Transmission over Multipath Fading Channel"
Xiao, Zou, Meng, Liu, Liang (Shenzhen Univ, Jan 2026).

Key Architectural Innovations:
1. Scale-Space Flow (SSF) & Feature-Space Warping (FSW):
   - Computes a continuous 3D flow field (dx, dy, dz), where dz indexes into a multi-scale
     Gaussian smoothing volume of reference features [f, f*G(sigma_0), f*G(2*sigma_0)].
   - 3D grid sampling (ss_warp) allows flow to select blurred scale levels for motion blur,
     rapid zoom, and disocclusions, completely preventing aliasing and spatial tearing artifacts.
2. Conditional Contextual Coding (replaces fragile additive residual subtraction):
   - Replaces pixel delta subtraction with non-linear conditional contextual synthesis.
   - P-frames are synthesized from content codes conditioned on multi-scale warped context,
     eliminating horizontal streak artifacts and blurry convergence traps.
3. Decoupled Wire Latent Equalization / Denoising:
   - Dedicated equalizer cleans channel noise before semantic synthesis.
   - Trained with dual loss: L_total = L_rec + lambda * L_latent (lambda = 0.7).
4. Strict Band W RF Budget (2,816 coordinates / second at 192x108 6.0 fps):
   - 6-frame GOP: 1 I-frame + 5 P-frames (1 second duration).
   - Frame 0 (I-frame): 896 coordinates (32 channels x 28 coords from 7x12 bottleneck).
   - Frames 1-5 (5 P-frames): 384 coordinates each:
     - 3D Scale-Space Flow: 96 coordinates (3 channels [dx, dy, dz] x 32 coords).
     - Conditional Content: 288 coordinates (12 channels x 24 coords).
   - Total: 896 + 5 * 384 = 2,816 real coordinates (exact on-air Band W budget, 0 waste).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aetv.ac4 import TXState, RXState, ReceivedFrame

ARCHITECTURE = "ac6-rvdjscc-v1"

# Geometry constants
WIDTH = 192
HEIGHT = 108
PAD_TOP = 2
PAD_BOTTOM = 2
PADDED_HEIGHT = 112
PADDED_WIDTH = 192
SPATIAL_SHAPE = (7, 12)
SPATIAL_SIZE = 7 * 12  # 84

# Budget constants
I_VALUES = 896
MOTION_VALUES = 96
CONTENT_VALUES = 288
P_VALUES = MOTION_VALUES + CONTENT_VALUES  # 384
GOP_VALUES = I_VALUES + 5 * P_VALUES       # 2816 (exact Band W budget)
GOP_FRAMES = 6
FRAME_SIZES = (I_VALUES,) + (P_VALUES,) * (GOP_FRAMES - 1)


def pad_frame(x: torch.Tensor) -> torch.Tensor:
    """Pad 108-height frame to 112 for exact 16x downsampling/upsampling."""
    if x.shape[-2] == HEIGHT:
        return F.pad(x, (0, 0, PAD_TOP, PAD_BOTTOM), mode="replicate")
    return x


def unpad_frame(x: torch.Tensor) -> torch.Tensor:
    """Crop 112-height frame back to original 108-height."""
    if x.shape[-2] == PADDED_HEIGHT:
        return x[..., PAD_TOP : PAD_TOP + HEIGHT, :]
    return x


def normalize_payload(x: torch.Tensor) -> torch.Tensor:
    """Unit RMS normalization for RF wire coordinates."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True).clamp_min(1e-8))


class GDN(nn.Module):
    """Positive cross-channel generalized divisive normalization."""

    def __init__(self, channels: int, inverse: bool = False):
        super().__init__()
        self.beta = nn.Parameter(torch.full((channels,), 0.54132485))
        self.gamma = nn.Parameter(torch.eye(channels) * 3.0 - 5.0)
        self.inverse = inverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            norm = F.conv2d(
                x.float().square(),
                F.softplus(self.gamma)[:, :, None, None],
                F.softplus(self.beta) + 1e-6,
            ).sqrt()
            y = x.float() * norm if self.inverse else x.float() / norm
        return y.to(dtype)


class _LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, floor):
        ctx.save_for_backward(value)
        ctx.floor = floor
        return value.clamp_min(floor)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        allowed = (value >= ctx.floor) | (gradient < 0)
        return gradient * allowed, None


class DiagonalGDN(GDN):
    """GDN with diagonal initialization and bounded square parameterization."""

    pedestal = (2**-18) ** 2

    def __init__(self, channels: int, inverse: bool = False):
        super().__init__(channels, inverse)
        self.beta = nn.Parameter(torch.full((channels,), (1 + self.pedestal) ** 0.5))
        self.gamma = nn.Parameter((0.1 * torch.eye(channels) + self.pedestal).sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            beta = _LowerBound.apply(self.beta, (1e-6 + self.pedestal) ** 0.5)
            gamma = _LowerBound.apply(self.gamma, self.pedestal**0.5)
            norm = F.conv2d(
                x.float().square(),
                (gamma.square() - self.pedestal)[:, :, None, None],
                beta.square() - self.pedestal,
            ).sqrt()
            y = x.float() * norm if self.inverse else x.float() / norm
        return y.to(x.dtype)


class Residual(nn.Module):
    def __init__(self, cin: int, cout: int | None = None, down: bool = False, up: bool = False):
        super().__init__()
        cout = cout or cin
        if down and up:
            raise ValueError("Residual cannot both downsample and upsample")

        def conv(a: int, b: int):
            return (
                nn.Sequential(nn.Conv2d(a, b * 4, 3, padding=1), nn.PixelShuffle(2))
                if up
                else nn.Conv2d(a, b, 3, stride=2 if down else 1, padding=1)
            )

        self.body = nn.Sequential(
            conv(cin, cout), nn.LeakyReLU(0.1), nn.Conv2d(cout, cout, 3, padding=1)
        )
        self.skip = conv(cin, cout) if cin != cout or down or up else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.body(x)


class DepthConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first = Residual(channels)
        self.down = Residual(channels, channels * 2, down=True)
        self.mid = nn.Sequential(DepthConv(channels * 2), Residual(channels * 2))
        self.up = Residual(channels * 2, channels, up=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), Residual(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.first(x)
        low = self.up(self.mid(self.down(skip)))
        return x + self.fuse(torch.cat([skip, low], 1))


class Refine(nn.Module):
    def __init__(self, features: int, width: int, deep_tail: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(features, width, 3, padding=1),
            nn.LeakyReLU(0.1),
            Residual(width),
            Residual(width),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        self.deep_tail = deep_tail
        if deep_tail:
            tail_ch = max(64, width * 2)
            self.tail_in = nn.Sequential(
                nn.Conv2d(features + 3, tail_ch, 3, padding=1),
                nn.LeakyReLU(0.1),
                Residual(tail_ch),
            )
            self.down = Residual(tail_ch, tail_ch, down=True)
            self.mid = nn.Sequential(
                nn.Conv2d(tail_ch, tail_ch, 3, padding=2, dilation=2),
                nn.LeakyReLU(0.1),
                Residual(tail_ch),
            )
            self.up = Residual(tail_ch, tail_ch, up=True)
            self.fuse = nn.Sequential(
                nn.Conv2d(tail_ch * 2, tail_ch, 1),
                nn.LeakyReLU(0.1),
                Residual(tail_ch),
                Residual(tail_ch),
            )
            self.edge_head = nn.Sequential(
                nn.Conv2d(tail_ch, width, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(width, 3, 3, padding=1),
            )
            nn.init.zeros_(self.edge_head[-1].weight)
            nn.init.zeros_(self.edge_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_logits = self.net(x)
        if not getattr(self, "deep_tail", False):
            return base_logits
        base_rgb = torch.sigmoid(base_logits)
        x_in = torch.cat([x, base_rgb], dim=1)
        feat = self.tail_in(x_in)
        low = self.up(self.mid(self.down(feat)))
        fused = self.fuse(torch.cat([feat, low], dim=1))
        delta = 1.5 * torch.tanh(self.edge_head(fused))
        return base_logits + delta


class FixedAnalysis(nn.Module):
    def __init__(self, cin: int, channels: int, coordinates: int, spatial_shape: tuple[int, int] = SPATIAL_SHAPE):
        super().__init__()
        if coordinates % channels:
            raise ValueError(f"Projection dimensions must divide exactly: {coordinates} % {channels}")
        self.spatial_shape = spatial_shape
        self.spatial_size = spatial_shape[0] * spatial_shape[1]
        self.map = nn.Conv2d(cin, channels, 1)
        self.spatial = nn.Linear(self.spatial_size, coordinates // channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.spatial_shape:
            raise ValueError(f"Analysis requires a {self.spatial_shape} feature plane, got {x.shape[-2:]}")
        return self.spatial(self.map(x).flatten(2)).flatten(1)


class FixedSynthesis(nn.Module):
    def __init__(self, channels: int, coordinates: int, cout: int, spatial_shape: tuple[int, int] = SPATIAL_SHAPE):
        super().__init__()
        self.channels, self.coordinates = channels, coordinates
        self.spatial_shape = spatial_shape
        self.spatial_size = spatial_shape[0] * spatial_shape[1]
        self.spatial = nn.Linear(coordinates // channels, self.spatial_size)
        self.map = nn.Conv2d(channels * 2, cout, 1)

    def forward(self, z: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        a = self.spatial((z * confidence).reshape(b, self.channels, -1))
        w = self.spatial(confidence.reshape(b, self.channels, -1))
        h, width = self.spatial_shape
        return self.map(torch.cat([a, w], 1).reshape(b, self.channels * 2, h, width))


class GaussianSmoothing(nn.Module):
    """Depthwise Gaussian filter from arXiv:2601.01729."""

    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.padding = kernel_size // 2
        mean = (kernel_size - 1) / 2
        grid = torch.arange(kernel_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")
        kernel = torch.exp(-((grid_x - mean) ** 2 + (grid_y - mean) ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=self.padding)


def ss_warp(vol: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
    """Scale-Space Warping: 3D grid sample across (x, y) spatial coordinates and z blur scale.
    vol: (B, C, D, H, W) where D is number of scale levels
    flo: (B, 3, H, W) where channels are (dx, dy, dz)
    """
    b, c, d, h, w = vol.shape
    ys = (torch.arange(h, device=vol.device, dtype=torch.float32) + 0.5) * 2 / h - 1
    xs = (torch.arange(w, device=vol.device, dtype=torch.float32) + 0.5) * 2 / w - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    grid_x = xx[None].expand(b, -1, -1) + flo[:, 0] * (2.0 / max(w - 1, 1))
    grid_y = yy[None].expand(b, -1, -1) + flo[:, 1] * (2.0 / max(h - 1, 1))
    grid_z = flo[:, 2].clamp(-1.0, 1.0)

    vgrid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(1)
    warped = F.grid_sample(vol.float(), vgrid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped.squeeze(2).to(vol.dtype)


class ScaleSpaceFlowCodec(nn.Module):
    """Learned Scale-Space Flow (SSF) estimator & codec projecting to 96 wire coordinates.
    Flow field has 3 channels: (dx, dy, dz) where dz selects Gaussian blur level.
    """

    def __init__(self, width: int = 40):
        super().__init__()
        self.estimate = nn.Sequential(
            nn.Conv2d(6, width, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            Residual(width, width, down=True),
            Residual(width),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        self.analysis = nn.Sequential(
            Residual(3, width, down=True),
            DepthConv(width),
            Residual(width, width, down=True),
            DepthConv(width),
            Residual(width, width, down=True),
            DepthConv(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
        )
        # 3 channels x 32 coordinates = 96 wire coordinates
        self.to_wire = FixedAnalysis(width, 3, MOTION_VALUES)
        self.from_wire = FixedSynthesis(3, MOTION_VALUES, width)
        self.synthesis = nn.Sequential(
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            nn.Conv2d(width, 3, 1),
        )

    def _bound_flow(self, raw_flow: torch.Tensor) -> torch.Tensor:
        dx_dy = 24.0 * torch.tanh(raw_flow[:, :2] / 24.0)
        dz = torch.tanh(raw_flow[:, 2:3])
        return torch.cat([dx_dy, dz], dim=1)

    def encode(self, previous: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_pad = pad_frame(previous)
        curr_pad = pad_frame(current)
        raw_flow = F.interpolate(
            self.estimate(torch.cat([prev_pad, curr_pad], 1) * 2 - 1),
            curr_pad.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        flow = self._bound_flow(raw_flow)
        wire_input = self.analysis(flow)
        return normalize_payload(self.to_wire(wire_input)), flow

    def decode(self, z: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        raw_flow = self.synthesis(self.from_wire(z, confidence))
        return self._bound_flow(raw_flow)


class ScaleSpaceContext(nn.Module):
    """Context conditioning using Scale-Space Warping across a Gaussian smoothed feature volume."""

    def __init__(self, features: int = 20, width: int = 48, ss_levels: int = 2, sigma: float = 1.0):
        super().__init__()
        self.features = features
        self.reference = nn.Conv2d(features + 3, features, 3, padding=1)
        self.gaussian_kernels = nn.ModuleList(
            [GaussianSmoothing(features, kernel_size=5, sigma=(2**i) * sigma) for i in range(ss_levels)]
        )
        self.refine = nn.Sequential(
            Residual(features),
            nn.Conv2d(features, features, 3, padding=1),
        )
        self.full = Residual(features)
        self.half_scale = Residual(features, width, down=True)
        self.quarter = Residual(width, width * 3 // 2, down=True)

    def generate_ss_volume(self, x: torch.Tensor) -> torch.Tensor:
        vols = [x]
        for kernel in self.gaussian_kernels:
            vols.append(kernel(x))
        return torch.stack(vols, dim=2)

    def forward(
        self, reference: torch.Tensor, features: torch.Tensor, ss_flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_pad = pad_frame(reference)
        feat_pad = pad_frame(features)
        source = self.reference(torch.cat([ref_pad * 2 - 1, feat_pad], 1))

        ssf_vol = self.generate_ss_volume(source)
        warped = ss_warp(ssf_vol, ss_flow)
        refined = self.refine(warped)

        full = self.full(refined)
        half = self.half_scale(full)
        quarter = self.quarter(half)
        return full, half, quarter


class IAnchorAdapter(nn.Module):
    """Loose feature anchor adapter for smooth inter-GOP boundary transitions."""

    def __init__(
        self,
        features: int = 20,
        width: int = 48,
        anchor_gate_max: float = 0.35,
        anchor_style_max: float = 0.30,
    ):
        super().__init__()
        self.features = features
        self.anchor_gate_max = float(anchor_gate_max)
        self.anchor_style_max = float(anchor_style_max)
        in_channels = features * 2 + 3
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(width, width, 3, padding=1, groups=width // 2 if width >= 4 else 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(width, features, 1),
        )
        self.gate = nn.Parameter(torch.tensor([-2.0]))
        self.style_weight = nn.Parameter(torch.tensor([-2.5]))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        prev_features: torch.Tensor | None,
        prev_rgb: torch.Tensor | None,
    ) -> torch.Tensor:
        if prev_features is None or prev_rgb is None:
            return features
        prev_feat = pad_frame(prev_features)
        p_rgb = pad_frame(prev_rgb)
        alpha = torch.sigmoid(self.style_weight) * self.anchor_style_max
        f_mean = features.mean(dim=(-2, -1), keepdim=True)
        f_std = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        p_mean = prev_feat.mean(dim=(-2, -1), keepdim=True)
        p_std = prev_feat.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        target_mean = (1 - alpha) * f_mean + alpha * p_mean
        target_std = (1 - alpha) * f_std + alpha * p_std
        f_normalized = (features - f_mean) / f_std * target_std + target_mean
        rgb_norm = p_rgb * 2 - 1 if p_rgb.min() >= 0 else p_rgb
        x_in = torch.cat([f_normalized, prev_feat, rgb_norm], dim=1)
        delta = self.adapter(x_in)
        feat_diff = (f_mean - p_mean).abs().mean(dim=1, keepdim=True)
        scene_gate = torch.exp(-feat_diff.clamp_min(0.0) / 0.5)
        g = torch.sigmoid(self.gate) * self.anchor_gate_max * scene_gate
        return f_normalized + g * delta


class MEMTransform(nn.Sequential):
    def __init__(self, channels: int = 96):
        layers = []
        for _ in range(2):
            layers.extend(
                [
                    Residual(channels),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    GDN(channels),
                    nn.LeakyReLU(0.1),
                ]
            )
        super().__init__(*layers, nn.Conv2d(channels, channels, 3, padding=1))


class ICodec(nn.Module):
    """I-frame analysis and synthesis projecting to exact 896 coordinates."""

    def __init__(
        self,
        width: int = 48,
        features: int = 20,
        iframe_mem: bool = True,
        diagonal_gdn: bool = True,
        anchor_gate_max: float = 0.35,
        anchor_style_max: float = 0.30,
        deep_tail: bool = True,
    ):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        gdn_cls = DiagonalGDN if diagonal_gdn else GDN
        blocks = []
        previous = 3
        for channels in (a, b, c):
            blocks.extend(
                [
                    Residual(previous, channels, down=True),
                    DepthConv(channels),
                    gdn_cls(channels),
                ]
            )
            previous = channels
        analysis_layers = [*blocks, nn.Conv2d(c, c, 3, stride=2, padding=1)]
        if iframe_mem:
            analysis_layers.append(MEMTransform(c))
        self.analysis = nn.Sequential(*analysis_layers)

        # 32 channels x 28 coordinates = 896 coordinates
        self.to_wire = FixedAnalysis(c, 32, I_VALUES)
        self.from_wire = FixedSynthesis(32, I_VALUES, c)

        synthesis_layers = []
        if iframe_mem:
            synthesis_layers.append(MEMTransform(c))
        synthesis_layers.extend(
            [
                DepthConv(c),
                Residual(c, c, up=True),
                gdn_cls(c, True),
                DepthConv(c),
                Residual(c, b, up=True),
                gdn_cls(b, True),
                DepthConv(b),
                Residual(b, a, up=True),
                gdn_cls(a, True),
                DepthConv(a),
                Residual(a, features, up=True),
                UNet(features),
            ]
        )
        self.synthesis = nn.Sequential(*synthesis_layers)
        self.anchor = IAnchorAdapter(
            features,
            width,
            anchor_gate_max=anchor_gate_max,
            anchor_style_max=anchor_style_max,
        )
        self.rgb = Refine(features, width, deep_tail=deep_tail)
        self.tx_projection = nn.Conv2d(3, features, 3, padding=1)

    def encode(
        self, x: torch.Tensor, prev_context: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pad = pad_frame(x)
        wire = normalize_payload(self.to_wire(self.analysis(x_pad * 2 - 1)))
        feat = self.tx_projection(x_pad * 2 - 1)
        if prev_context is not None:
            prev_feat, prev_rgb = prev_context
            feat = self.anchor(feat, prev_feat, prev_rgb)
        return wire, feat

    def decode(
        self,
        z: torch.Tensor,
        confidence: torch.Tensor,
        prev_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.synthesis(self.from_wire(z, confidence))
        if prev_context is not None:
            prev_feat, prev_rgb = prev_context
            features = self.anchor(features, prev_feat, prev_rgb)
        rgb = unpad_frame(torch.sigmoid(self.rgb(features)))
        return rgb, features


class PAnalysis(nn.Module):
    """Conditional context analysis projecting to 288 wire coordinates."""

    def __init__(self, width: int = 48, features: int = 20, diagonal_gdn: bool = True):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        gdn_cls = DiagonalGDN if diagonal_gdn else GDN
        self.first = nn.Sequential(
            nn.Conv2d(3 + features, a, 3, stride=2, padding=1), gdn_cls(a)
        )
        self.second = nn.Sequential(
            Residual(a * 2), nn.Conv2d(a * 2, b, 3, stride=2, padding=1), gdn_cls(b)
        )
        self.third = nn.Sequential(
            Residual(b * 2),
            nn.Conv2d(b * 2, c, 3, stride=2, padding=1),
            gdn_cls(c),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
        )
        # 12 channels x 24 coordinates = 288 coordinates
        self.to_wire = FixedAnalysis(c, 12, CONTENT_VALUES)

    def forward(
        self, x: torch.Tensor, contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pad = pad_frame(x)
        full, half, quarter = contexts
        h = self.first(torch.cat([x_pad * 2 - 1, full], 1))
        h = self.second(torch.cat([h, half], 1))
        h = self.third(torch.cat([h, quarter], 1))
        return normalize_payload(self.to_wire(h)), h


class PSynthesis(nn.Module):
    """Multi-scale contextual synthesis from 288 wire coordinates."""

    def __init__(
        self,
        width: int = 48,
        features: int = 20,
        diagonal_gdn: bool = True,
        refine_blocks: int = 2,
    ):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        gdn_cls = DiagonalGDN if diagonal_gdn else GDN
        self.first = nn.Sequential(
            Residual(c, c, up=True), gdn_cls(c, True), Residual(c, b, up=True), gdn_cls(b, True)
        )
        self.second = nn.Sequential(
            Residual(b * 2), Residual(b * 2, a, up=True), gdn_cls(a, True)
        )
        self.third = nn.Sequential(Residual(a * 2), Residual(a * 2, features, up=True))
        refine_layers = [
            nn.Conv2d(features * 2, features, 3, padding=1),
            DepthConv(features),
        ]
        for _ in range(refine_blocks):
            refine_layers.append(UNet(features))
        self.refine = nn.Sequential(*refine_layers)

    def forward(
        self, h: torch.Tensor, contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        full, half, quarter = contexts
        h = self.second(torch.cat([self.first(h), quarter], 1))
        h = self.third(torch.cat([h, half], 1))
        return self.refine(torch.cat([h, full], 1))


class WireEqualizer(nn.Module):
    """Lightweight wire denoising module (equalizer) from arXiv:2601.01729 Fig 3."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        feat = torch.stack([z, confidence], dim=-1)
        delta = self.net(feat).squeeze(-1)
        return z + delta


def load_warmstart_from_ac2k(model: AC6Codec, ac2k_checkpoint_path: str | Path) -> dict[str, int]:
    """Transfers compatible multi-scale backbone weights and slices linear projections from AC2K."""
    ckpt = torch.load(ac2k_checkpoint_path, map_location="cpu")
    ac2k_sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    model_sd = model.state_dict()

    exact_count = 0
    sliced_count = 0

    for k, v in model_sd.items():
        if k in ac2k_sd:
            src = ac2k_sd[k]
            if src.shape == v.shape:
                v.copy_(src)
                exact_count += 1
            else:
                slices_src = []
                slices_dst = []
                for s_dim, d_dim in zip(src.shape, v.shape):
                    m = min(s_dim, d_dim)
                    slices_src.append(slice(0, m))
                    slices_dst.append(slice(0, m))
                v.zero_()
                v[tuple(slices_dst)].copy_(src[tuple(slices_src)])
                sliced_count += 1

    model.load_state_dict(model_sd)
    return {"exact": exact_count, "sliced": sliced_count}


class LooseIAnchor(nn.Module):
    """Loose feature anchor adapter for smooth inter-GOP boundary transitions (legacy checkpoint format)."""

    def __init__(self, in_channels: int = 14, out_channels: int = 8, gate_max: float = 0.35):
        super().__init__()
        self.gate_max = float(gate_max)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.gate = nn.Parameter(torch.tensor([-2.0]))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, feat: torch.Tensor, prev_feat: torch.Tensor | None) -> torch.Tensor:
        if prev_feat is None:
            return feat
        delta = self.adapter(torch.cat([feat, prev_feat], dim=1))
        g = torch.sigmoid(self.gate) * self.gate_max
        return feat + g * delta


class AC6Codec(nn.Module):
    """Production 2.2 kHz RF Video Codec (Mode AC6: 192x108 @ 6.0 fps, 2,816 coords/sec).

    Integrates arXiv:2601.01729 (RVDJSCC) Scale-Space Flow, Conditional Contextual Coding,
    and Decoupled Latent Equalization. Fully backwards-compatible with legacy checkpoints.
    """

    architecture = ARCHITECTURE

    def __init__(
        self,
        arch: str = "rvdjscc",
        width: int = 48,
        features: int = 20,
        motion_width: int = 40,
        refine_blocks: int = 2,
        confidence_threshold: float = 0.02,
        diagonal_gdn: bool = True,
        iframe_mem: bool = True,
        anchor_gate_max: float = 0.35,
        anchor_style_max: float = 0.39,
        deep_tail: bool = True,
        spatial_pooling: bool | None = None,
        model_width: int | None = None,
        **kwargs,
    ):
        super().__init__()
        # Handle parameter aliases
        width = model_width if model_width is not None else width
        self.height = HEIGHT
        self.width = WIDTH
        self.padded_height = PADDED_HEIGHT
        self.padded_width = PADDED_WIDTH
        self.gop_frames = GOP_FRAMES
        self.i_values = I_VALUES
        self.motion_values = MOTION_VALUES
        self.content_values = CONTENT_VALUES
        self.p_values = P_VALUES
        self.gop_values = GOP_VALUES
        self.frame_sizes = FRAME_SIZES
        self.features = features
        self.confidence_threshold = confidence_threshold
        self.anchor_gate_max = float(anchor_gate_max)
        self.anchor_style_max = float(anchor_style_max)

        # Determine architecture: legacy 1D mode is only used if explicitly spatial_pooling=False
        self.is_legacy = spatial_pooling is False
        self.config = dict(
            arch="legacy" if self.is_legacy else "rvdjscc",
            width=width,
            features=features,
            motion_width=motion_width,
            refine_blocks=refine_blocks,
            confidence_threshold=confidence_threshold,
            diagonal_gdn=diagonal_gdn,
            iframe_mem=iframe_mem,
            anchor_gate_max=anchor_gate_max,
            anchor_style_max=anchor_style_max,
            deep_tail=deep_tail,
        )

        if not self.is_legacy:
            # SOTA arXiv:2601.01729 RVDJSCC Contextual Architecture
            self.i_codec = ICodec(
                width=width,
                features=features,
                iframe_mem=iframe_mem,
                diagonal_gdn=diagonal_gdn,
                anchor_gate_max=anchor_gate_max,
                anchor_style_max=anchor_style_max,
                deep_tail=deep_tail,
            )
            self.motion = ScaleSpaceFlowCodec(motion_width)
            self.tx_context = ScaleSpaceContext(features, width, ss_levels=2, sigma=1.0)
            self.rx_context = ScaleSpaceContext(features, width, ss_levels=2, sigma=1.0)
            self.p_analysis = PAnalysis(width, features, diagonal_gdn=diagonal_gdn)
            self.p_from_wire = FixedSynthesis(12, CONTENT_VALUES, width * 2)
            self.tx_projection = PSynthesis(
                width, features, diagonal_gdn=diagonal_gdn, refine_blocks=refine_blocks
            )
            self.p_synthesis = PSynthesis(
                width, features, diagonal_gdn=diagonal_gdn, refine_blocks=refine_blocks
            )
            self.p_rgb = Refine(features, width, deep_tail=deep_tail)
            self.equalizer = WireEqualizer(hidden_dim=64)

        else:
            # Legacy 1D Linear Mapping Path (retained for models/ac6-best-inference.pt backward compatibility)
            mw = width
            mot_w = motion_width
            self.i_enc = nn.Sequential(
                nn.Conv2d(3, mw, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw, mw * 3 // 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 3 // 2, mw * 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, mw * 2, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, 8, 1),
            )
            self.i_to_wire = nn.Linear(336, 112)
            self.i_from_wire = nn.Linear(112, 336)
            self.i_dec = nn.Sequential(
                nn.Conv2d(8 * 2, mw * 2, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, mw * 2, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, (mw * 3 // 2) * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 3 // 2, mw * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw, mw * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw, 3, 3, padding=1),
                nn.Sigmoid(),
            )
            self.i_anchor = LooseIAnchor(in_channels=14, out_channels=8, gate_max=anchor_gate_max)

            self.flow_enc = nn.Sequential(
                nn.Conv2d(6, mot_w, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mot_w, mot_w, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mot_w, mot_w, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mot_w, 2, 1),
            )
            self.flow_to_wire = nn.Linear(336, 48)
            self.flow_from_wire = nn.Linear(48, 336)
            self.flow_dec = nn.Sequential(
                nn.Conv2d(2 * 2, mot_w, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mot_w, mot_w * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mot_w, max(mot_w // 2, 8) * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(max(mot_w // 2, 8), 2 * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )

            self.res_enc = nn.Sequential(
                nn.Conv2d(6, mw, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw, mw * 3 // 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 3 // 2, mw * 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, 6, 1),
            )
            self.res_to_wire = nn.Linear(336, 48)
            self.res_from_wire = nn.Linear(48, 336)
            self.res_dec = nn.Sequential(
                nn.Conv2d(6 * 2, mw * 2, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, mw * 2, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 2, (mw * 3 // 2) * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw * 3 // 2, mw * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw, mw * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mw, 4, 3, padding=1),
            )

        self._tx_state: TXState | None = None
        self._rx_state: RXState | None = None

    def reset(self) -> None:
        self._tx_state = None
        self._rx_state = None

    reset_state = reset

    @property
    def tx_state(self) -> TXState | None:
        return self._tx_state

    @property
    def rx_state(self) -> RXState | None:
        return self._rx_state

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        return pad_frame(x)

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        return unpad_frame(x)

    def encode(
        self, frame: torch.Tensor, position: int, state: TXState | None = None
    ) -> tuple[torch.Tensor, TXState]:
        if not self.is_legacy:
            # SOTA arXiv:2601.01729 Path
            if position == 0:
                prev_context = (state.features, state.original) if state is not None else None
                z, feature = self.i_codec.encode(frame, prev_context=prev_context)
                next_state = TXState(frame, feature, 1)
            else:
                if state is None or state.next_position != position:
                    raise ValueError(f"P encoding at pos {position} requires preceding history (got {state})")
                mz, ss_flow = self.motion.encode(state.original, frame)
                context = self.tx_context(state.original, state.features, ss_flow)
                cz, hidden = self.p_analysis(frame, context)
                feature = self.tx_projection(hidden, context)
                z = torch.cat([mz, cz], 1)
                next_state = TXState(frame, feature, (position + 1) % self.gop_frames)
            return z, next_state

        else:
            # Legacy 1D Linear Path
            b = frame.shape[0]
            f_pad = self._pad(frame)
            if position == 0:
                feat = self.i_enc(f_pad)
                feat = self.i_anchor(feat, state.features if state is not None else None)
                wire = self.i_to_wire(feat.reshape(b, 8, 336)).flatten(1)
                wire = wire / (wire.square().mean().sqrt() + 1e-6)
                next_state = TXState(frame, feat, 1)
                return wire, next_state
            else:
                if state is None:
                    raise ValueError("P encoding requires preceding state")
                p_pad = self._pad(state.original)
                m_feat = self.flow_enc(torch.cat([p_pad, f_pad], dim=1))
                m_wire = self.flow_to_wire(m_feat.reshape(b, 2, 336)).flatten(1)
                m_wire = m_wire / (m_wire.square().mean().sqrt() + 1e-6)

                fl_in = self.flow_from_wire(m_wire.reshape(b, 2, 48)).reshape(b, 2, 14, 24)
                fl = 16.0 * torch.tanh(self.flow_dec(torch.cat([fl_in, torch.ones_like(fl_in)], dim=1)) / 16.0)
                from aetv.ac4 import warp
                warped = warp(p_pad, fl)

                r_feat = self.res_enc(torch.cat([f_pad, warped], dim=1))
                r_wire = self.res_to_wire(r_feat.reshape(b, 6, 336)).flatten(1)
                r_wire = r_wire / (r_wire.square().mean().sqrt() + 1e-6)

                wire = torch.cat([m_wire, r_wire], dim=1)
                next_state = TXState(frame, r_feat, (position + 1) % self.gop_frames)
                return wire, next_state

    def decode(
        self,
        received: ReceivedFrame,
        state: RXState | None = None,
        equalize: bool = True,
        return_outage: bool = False,
    ) -> tuple[torch.Tensor, RXState] | tuple[torch.Tensor, RXState, torch.Tensor]:
        position, z = received.position, received.payload
        confidence = received.confidence
        if confidence is None:
            confidence = torch.ones_like(z)

        if not self.is_legacy:
            # SOTA arXiv:2601.01729 Path
            if equalize:
                z_clean = self.equalizer(z, confidence)
            else:
                z_clean = z

            b = z_clean.shape[0]
            finite = torch.isfinite(z_clean) & torch.isfinite(confidence)
            confidence = torch.where(finite, confidence.clamp(0, 1), 0).float()
            z_clean = torch.where(finite, z_clean, 0)

            usable = (confidence.mean(1) > self.confidence_threshold) & finite.all(1)
            if position:
                usable = usable & (confidence[:, :self.motion_values].mean(1) > self.confidence_threshold)
            if received.usable is not None:
                usable = usable & received.usable.bool()

            if position == 0:
                prev_context = None
                if state is not None and bool(state.valid.any()):
                    prev_context = (
                        torch.where(state.valid[:, None, None, None], state.features, 0.0),
                        torch.where(state.valid[:, None, None, None], state.reference, 0.0),
                    )
                rgb, feature = self.i_codec.decode(z_clean, confidence, prev_context=prev_context)
                next_state = RXState(rgb, feature, usable, 1)
                outage = ~usable
            else:
                if state is None or state.next_position != position:
                    old_rgb = z_clean.new_full((b, 3, self.height, self.width), 0.5)
                    old_feat = z_clean.new_zeros((b, self.features, self.padded_height, self.padded_width))
                    state = RXState(old_rgb, old_feat, torch.zeros(b, device=z_clean.device, dtype=torch.bool), position)

                mz = z_clean[:, :self.motion_values]
                mw = confidence[:, :self.motion_values]
                cz = z_clean[:, self.motion_values:]
                cw = confidence[:, self.motion_values:]

                ss_flow = self.motion.decode(mz, mw)
                context = self.rx_context(state.reference, state.features, ss_flow)
                hidden = self.p_from_wire(cz, cw)
                feature = self.p_synthesis(hidden, context)
                rgb = unpad_frame(torch.sigmoid(self.p_rgb(feature)))

                valid = usable & state.valid
                rgb = torch.where(valid[:, None, None, None], rgb, state.reference)
                feature = torch.where(valid[:, None, None, None], feature, state.features)
                next_state = RXState(rgb, feature, valid, (position + 1) % self.gop_frames)
                outage = ~usable

            if return_outage:
                return rgb, next_state, outage
            return rgb, next_state

        else:
            # Legacy 1D Linear Path
            b = z.shape[0]
            from aetv.ac4 import warp
            if position == 0:
                z_wire = z.reshape(b, 8, 112)
                w_wire = confidence.reshape(b, 8, 112)
                feat_val = self.i_from_wire(z_wire * w_wire).reshape(b, 8, 14, 24)
                feat_w = self.i_from_wire(w_wire).reshape(b, 8, 14, 24)
                prev_feat = state.features if (state is not None and bool(state.valid.any())) else None
                feat_val = self.i_anchor(feat_val, prev_feat)
                rgb_pad = self.i_dec(torch.cat([feat_val, feat_w], dim=1))
                rgb = self._unpad(rgb_pad)
                next_state = RXState(rgb, feat_val, torch.ones(b, device=z.device, dtype=torch.bool), 1)
                if return_outage:
                    return rgb, next_state, torch.zeros(b, device=z.device, dtype=torch.bool)
                return rgb, next_state
            else:
                if state is None:
                    old_rgb = z.new_full((b, 3, self.height, self.width), 0.5)
                    old_feat = z.new_zeros((b, 6, 14, 24))
                    state = RXState(old_rgb, old_feat, torch.zeros(b, device=z.device, dtype=torch.bool), position)

                p_pad = self._pad(state.reference)
                m_z = z[:, :self.motion_values]
                m_w = confidence[:, :self.motion_values]
                fl_val = self.flow_from_wire(m_z.reshape(b, 2, 48) * m_w.reshape(b, 2, 48)).reshape(b, 2, 14, 24)
                fl_w = self.flow_from_wire(m_w.reshape(b, 2, 48)).reshape(b, 2, 14, 24)
                fl = 16.0 * torch.tanh(self.flow_dec(torch.cat([fl_val, fl_w], dim=1)) / 16.0)

                warped = warp(p_pad, fl)
                r_z = z[:, self.motion_values:]
                r_w = confidence[:, self.motion_values:]
                res_val = self.res_from_wire(r_z.reshape(b, 6, 48) * r_w.reshape(b, 6, 48)).reshape(b, 6, 14, 24)
                res_w = self.res_from_wire(r_w.reshape(b, 6, 48)).reshape(b, 6, 14, 24)
                res_out = self.res_dec(torch.cat([res_val, res_w], dim=1))

                mask = torch.sigmoid(res_out[:, 0:1] + 2.0)
                delta = torch.tanh(res_out[:, 1:4])
                rgb_pad = (mask * warped + delta).clamp(0.0, 1.0)
                rgb = self._unpad(rgb_pad)
                next_state = RXState(rgb, res_val, state.valid, (position + 1) % self.gop_frames)
                if return_outage:
                    return rgb, next_state, ~state.valid
                return rgb, next_state

    def encode_gop(
        self,
        video: torch.Tensor,
        state: TXState | None = None,
        retain_state: bool = True,
    ) -> torch.Tensor:
        current_state = self._tx_state if (state is None and retain_state) else state
        payloads = []
        for position in range(self.gop_frames):
            z, current_state = self.encode(video[:, :, position], position, current_state)
            payloads.append(z)
        if retain_state:
            self._tx_state = current_state
        return torch.cat(payloads, 1)

    def decode_gop(
        self,
        payload: torch.Tensor,
        confidence: torch.Tensor | None = None,
        state: RXState | None = None,
        retain_state: bool = True,
        equalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if confidence is None:
            confidence = torch.ones_like(payload)
        current_state = self._rx_state if (state is None and retain_state) else state
        frames = []
        outages = []
        for position, (z, w) in enumerate(
            zip(payload.split(self.frame_sizes, 1), confidence.split(self.frame_sizes, 1))
        ):
            frame, current_state, outage = self.decode(
                ReceivedFrame(position, z, w), current_state, equalize=equalize, return_outage=True
            )
            frames.append(frame)
            outages.append(outage)
        if retain_state:
            self._rx_state = current_state
        return torch.stack(frames, 2), torch.stack(outages, 1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        wire = self.encode_gop(video, retain_state=False)
        recon, _ = self.decode_gop(wire, retain_state=False)
        return recon
