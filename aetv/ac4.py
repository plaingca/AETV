"""AC4: 4x Downsample Video Codec for V8 Resolution and Band W RF Budget.

Engineering design replacing 16x downsampling (84 bottleneck cells) with 4x downsampling:
- Native resolution: 192x108 (16:9 aspect ratio)
- Padded height: 112 (112 / 4 = 28 exact, 192 / 4 = 48 exact)
- Bottleneck spatial resolution: 28x48 = 1,344 spatial cells (16x higher spatial resolution than AC2K!)
- RF Bandwidth budget: Exactly Band W standard channel (2,816 real coordinates per GOP):
    * I-frame: 1,280 coordinates (4 channels x 320 spatial coordinates)
    * P-frame: 512 coordinates:
        - Motion: 128 coordinates (4 channels x 32 spatial coordinates)
        - Residual content: 384 coordinates (2 channels x 192 spatial coordinates)
    * GOP Total: 1,280 + 3 * 512 = 2,816 coordinates (exact on-air Band W budget)
- Loose I-frame context anchoring:
    * The I-frame codec accepts and retains context from the previous GOP's last P-frame.
    * Gated soft moment matching (AdaIN style) and learned residual adaptation loosely
      anchor the new I-frame to the preceding P-frame, eliminating inter-GOP style shifts
      and luminance/color flicker.
    * Standalone decoding is fully preserved when no prior context is available
      (initial tune-in or after channel outages).
    * Scene-change awareness gracefully attenuates the anchor on scene cuts.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "ac4-4x-downsample-v1"

# Wire coordinate allocations (exact Band W on-air budget):
I_VALUES = 1280
MOTION_VALUES = 128
CONTENT_VALUES = 384
P_VALUES = MOTION_VALUES + CONTENT_VALUES  # 512
GOP_VALUES = I_VALUES + 3 * P_VALUES  # 2816 (exact Band W budget)
FRAME_SIZES = (I_VALUES, P_VALUES, P_VALUES, P_VALUES)
P_WEIGHTS = (0.9, 1.2, 0.9)
GOP_FRAMES = 4

# Spatial geometry (4x downsampling):
WIDTH = 192
HEIGHT = 108
PAD_TOP = 2
PAD_BOTTOM = 2
PADDED_HEIGHT = 112  # 108 + 2 + 2 = 112 (112 / 4 = 28 exact integer)
PADDED_WIDTH = 192   # 192 / 4 = 48 exact integer
SPATIAL_SHAPE_4X = (28, 48)
SPATIAL_SIZE_4X = 28 * 48  # 1,344 locations (16x more than AC2K's 84)


def pad_frame(x: torch.Tensor) -> torch.Tensor:
    """Pad 108-height frame to 112 for exact 4x downsampling and upsampling."""
    if x.shape[-2] == HEIGHT:
        return F.pad(x, (0, 0, PAD_TOP, PAD_BOTTOM), mode="replicate")
    return x


def unpad_frame(x: torch.Tensor) -> torch.Tensor:
    """Crop 112-height frame back to original 108-height."""
    if x.shape[-2] == PADDED_HEIGHT:
        return x[..., PAD_TOP : PAD_TOP + HEIGHT, :]
    return x


def normalize_payload(x: torch.Tensor) -> torch.Tensor:
    """Per-source, per-frame/stream RMS normalization; bounded coordinates."""
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
    """Allow a parameter at its floor to move back into the feasible region."""

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
    """All skips are computed at the endpoint executing this module."""

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
    """Nonlinear RGB refinement after propagated decoder features."""

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


class MEMTransform(nn.Sequential):
    """Memory transform signal branch for high-capacity I-frame representations."""

    def __init__(self, channels: int = 64):
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


class FixedAnalysis(nn.Module):
    """Learned exact-dimensional projection for 4x downsampled (28x48) feature planes."""

    def __init__(
        self,
        cin: int,
        channels: int,
        coordinates: int,
        spatial_shape: tuple[int, int] = SPATIAL_SHAPE_4X,
    ):
        super().__init__()
        if coordinates % channels != 0:
            raise ValueError(
                f"Projection dimensions must divide exactly: {coordinates} % {channels} != 0"
            )
        self.spatial_shape = spatial_shape
        self.spatial_size = spatial_shape[0] * spatial_shape[1]
        self.channels = channels
        self.coordinates = coordinates
        self.map = nn.Conv2d(cin, channels, 1)
        self.spatial = nn.Linear(self.spatial_size, coordinates // channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.spatial_shape:
            raise ValueError(
                f"AC4 analysis requires a {self.spatial_shape} feature plane, got {x.shape[-2:]}"
            )
        return self.spatial(self.map(x).flatten(2)).flatten(1)


class FixedSynthesis(nn.Module):
    """Learned exact-dimensional synthesis for 4x downsampled (28x48) feature planes."""

    def __init__(
        self,
        channels: int,
        coordinates: int,
        cout: int,
        spatial_shape: tuple[int, int] = SPATIAL_SHAPE_4X,
    ):
        super().__init__()
        if coordinates % channels != 0:
            raise ValueError(
                f"Synthesis dimensions must divide exactly: {coordinates} % {channels} != 0"
            )
        self.channels = channels
        self.coordinates = coordinates
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


def warp(x: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Backward optical flow in full-raster pixels."""
    b, _, h, w = x.shape
    flow = F.interpolate(motion.float(), (h, w), mode="bilinear", align_corners=False)
    ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) * 2 / h - 1
    xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) * 2 / w - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], -1)[None].expand(b, -1, -1, -1)
    scale = flow.new_tensor([2.0 / w, 2.0 / h])[None, :, None, None]
    grid = grid + (flow * scale).permute(0, 2, 3, 1)
    return F.grid_sample(
        x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).to(x.dtype)


class IAnchorAdapter(nn.Module):
    """Loosely anchors the new I-frame synthesis to the preceding P-frame context.

    Prevents style, luminance, and color shifts across GOP boundaries
    while preserving standalone tune-in decoding and scene cut robustness.
    """

    def __init__(
        self,
        features: int = 16,
        width: int = 32,
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

        # 1. Soft global moment anchor
        alpha = torch.sigmoid(self.style_weight) * self.anchor_style_max
        f_mean = features.mean(dim=(-2, -1), keepdim=True)
        f_var = (features - f_mean).square().mean(dim=(-2, -1), keepdim=True)
        f_std = torch.sqrt(f_var + 1e-6)
        p_mean = prev_feat.mean(dim=(-2, -1), keepdim=True)
        p_var = (prev_feat - p_mean).square().mean(dim=(-2, -1), keepdim=True)
        p_std = torch.sqrt(p_var + 1e-6)

        target_mean = (1 - alpha) * f_mean + alpha * p_mean
        target_std = (1 - alpha) * f_std + alpha * p_std
        f_normalized = (features - f_mean) / f_std * target_std + target_mean

        # 2. Local spatial residual adapter
        rgb_norm = p_rgb * 2 - 1 if p_rgb.min() >= 0 else p_rgb
        x_in = torch.cat([f_normalized, prev_feat, rgb_norm], dim=1)
        delta = self.adapter(x_in)

        # 3. Scene-change attenuation
        feat_diff = (f_mean - p_mean).abs().mean(dim=1, keepdim=True)
        scene_gate = torch.exp(-feat_diff.clamp_min(0.0) / 0.5)

        g = torch.sigmoid(self.gate) * self.anchor_gate_max * scene_gate
        return f_normalized + g * delta


class ICodec(nn.Module):
    """4x Downsampled I-frame analysis and synthesis with exact 1280 wire coordinates."""

    def __init__(
        self,
        width: int = 32,
        features: int = 16,
        wire_channels: int = 4,  # 1280 // 4 = 320 coords per channel
        iframe_mem: bool = True,
        diagonal_gdn: bool = True,
        anchor_gate_max: float = 0.35,
        anchor_style_max: float = 0.30,
        deep_tail: bool = False,
    ):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2  # 32, 48, 64
        gdn_cls = DiagonalGDN if diagonal_gdn else GDN

        # 4x Downsampling Analysis: exactly 2 downsampling stages
        # Stage 1: (112, 192) -> (56, 96)
        # Stage 2: (56, 96) -> (28, 48)
        analysis_layers = [
            Residual(3, a, down=True),
            DepthConv(a),
            gdn_cls(a),
            Residual(a, b, down=True),
            DepthConv(b),
            gdn_cls(b),
            Residual(b, c),
            DepthConv(c),
            gdn_cls(c),
        ]
        if iframe_mem:
            analysis_layers.append(MEMTransform(c))
        self.analysis = nn.Sequential(*analysis_layers)

        self.to_wire = FixedAnalysis(c, wire_channels, I_VALUES, spatial_shape=SPATIAL_SHAPE_4X)
        self.from_wire = FixedSynthesis(wire_channels, I_VALUES, c, spatial_shape=SPATIAL_SHAPE_4X)

        # 4x Upsampling Synthesis: exactly 2 upsampling stages
        # Stage 1: (28, 48) -> (56, 96)
        # Stage 2: (56, 96) -> (112, 192)
        synthesis_layers = []
        if iframe_mem:
            synthesis_layers.append(MEMTransform(c))
        synthesis_layers.extend(
            [
                DepthConv(c),
                Residual(c, b, up=True),
                gdn_cls(b, inverse=True),
                DepthConv(b),
                Residual(b, a, up=True),
                gdn_cls(a, inverse=True),
                DepthConv(a),
                Residual(a, features),
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


class MotionCodec(nn.Module):
    """Learned flow estimator projecting to 128 motion wire coordinates with 4x resolution."""

    def __init__(self, width: int = 32, wire_channels: int = 4):
        super().__init__()
        self.estimate = nn.Sequential(
            nn.Conv2d(6, width, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            Residual(width, width, down=True),
            Residual(width),
            nn.Conv2d(width, 2, 3, padding=1),
        )
        # 2 stages of downsampling: (112, 192) -> (56, 96) -> (28, 48)
        self.analysis = nn.Sequential(
            Residual(2, width, down=True),
            DepthConv(width),
            Residual(width, width, down=True),
            DepthConv(width),
            Residual(width, width),
        )
        self.to_wire = FixedAnalysis(width, wire_channels, MOTION_VALUES, spatial_shape=SPATIAL_SHAPE_4X)
        self.from_wire = FixedSynthesis(wire_channels, MOTION_VALUES, width, spatial_shape=SPATIAL_SHAPE_4X)
        # 2 stages of upsampling: (28, 48) -> (56, 96) -> (112, 192)
        self.synthesis = nn.Sequential(
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            nn.Conv2d(width, 2, 1),
        )

    def encode(self, previous: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_pad = pad_frame(previous)
        curr_pad = pad_frame(current)
        flow = F.interpolate(
            self.estimate(torch.cat([prev_pad, curr_pad], 1) * 2 - 1),
            curr_pad.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        flow = 24 * torch.tanh(flow / 24)
        return normalize_payload(self.to_wire(self.analysis(flow / 24))), flow

    def decode(self, z: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        return 24 * torch.tanh(self.synthesis(self.from_wire(z, confidence)) / 24)


class Context(nn.Module):
    """Learned offset hypotheses, fused at full (112x192), half (56x96), and quarter (28x48) scales."""

    def __init__(self, features: int = 16, width: int = 32, num_offsets: int = 2):
        super().__init__()
        self.features = features
        self.num_offsets = num_offsets
        self.reference = nn.Conv2d(features + 3, features, 3, padding=1)
        self.offsets = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features, 3 * num_offsets, 3, padding=1),
        )
        self.full = Residual(features)
        self.half_scale = Residual(features, width, down=True)
        self.quarter = Residual(width, width * 3 // 2, down=True)

    def forward(
        self, reference: torch.Tensor, features: torch.Tensor, motion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_pad = pad_frame(reference)
        feat_pad = pad_frame(features)
        source = self.reference(torch.cat([ref_pad * 2 - 1, feat_pad], 1))
        initial = warp(source, motion)
        offsets = self.offsets(initial)
        flow_channels = 2 * self.num_offsets
        flow_offsets = offsets[:, :flow_channels]
        weights = offsets[:, flow_channels:].softmax(1)
        aligned = sum(
            warp(source, motion + 2 * torch.tanh(flow_offsets[:, 2 * i : 2 * i + 2]))
            * weights[:, i : i + 1]
            for i in range(self.num_offsets)
        )
        full = self.full(aligned)
        half = self.half_scale(full)
        return full, half, self.quarter(half)


class PAnalysis(nn.Module):
    """4x Downsampled P-frame residual analysis network."""

    def __init__(
        self,
        width: int = 32,
        features: int = 16,
        wire_channels: int = 2,  # 384 // 2 = 192 coords per channel
        diagonal_gdn: bool = True,
    ):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        gdn_cls = DiagonalGDN if diagonal_gdn else GDN
        # Stage 1: (112, 192) -> (56, 96)
        self.first = nn.Sequential(
            nn.Conv2d(3 + features, a, 3, stride=2, padding=1), gdn_cls(a)
        )
        # Stage 2: (56, 96) -> (28, 48)
        self.second = nn.Sequential(
            Residual(a * 2), nn.Conv2d(a * 2, b, 3, stride=2, padding=1), gdn_cls(b)
        )
        # Stage 3: Feature refinement at (28, 48) with quarter-scale context
        self.third = nn.Sequential(
            Residual(b * 2),
            nn.Conv2d(b * 2, c, 3, padding=1),
            gdn_cls(c),
            Residual(c),
        )
        self.to_wire = FixedAnalysis(c, wire_channels, CONTENT_VALUES, spatial_shape=SPATIAL_SHAPE_4X)

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
    """4x Upsampled P-frame residual synthesis network."""

    def __init__(
        self,
        width: int = 32,
        features: int = 16,
        diagonal_gdn: bool = True,
        refine_blocks: int = 2,
    ):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        gdn_cls = DiagonalGDN if diagonal_gdn else GDN
        # Input h is at (28, 48)
        self.first = nn.Sequential(
            Residual(c),
            Residual(c, b),
        )
        # Fuse quarter context at (28, 48), upsample to (56, 96)
        self.second = nn.Sequential(
            Residual(b * 2),
            Residual(b * 2, a, up=True),
            gdn_cls(a, True),
        )
        # Fuse half context at (56, 96), upsample to (112, 192)
        self.third = nn.Sequential(
            Residual(a * 2),
            Residual(a * 2, features, up=True),
        )
        # Fuse full context at (112, 192), refine
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
        h = self.first(h)
        h = self.second(torch.cat([h, quarter], 1))
        h = self.third(torch.cat([h, half], 1))
        return self.refine(torch.cat([h, full], 1))


@dataclass(frozen=True)
class TXState:
    original: torch.Tensor
    features: torch.Tensor
    next_position: int

    def detach(self) -> TXState:
        return TXState(
            self.original.detach(), self.features.detach(), self.next_position
        )


@dataclass(frozen=True)
class RXState:
    reference: torch.Tensor
    features: torch.Tensor
    valid: torch.Tensor
    next_position: int

    def detach(self) -> RXState:
        return RXState(
            self.reference.detach(),
            self.features.detach(),
            self.valid.detach(),
            self.next_position,
        )


@dataclass(frozen=True)
class ReceivedFrame:
    position: int
    payload: torch.Tensor
    confidence: torch.Tensor | None = None
    usable: torch.Tensor | None = None

    @property
    def frame_type(self) -> str:
        return "I" if self.position == 0 else "P"


class AC4Codec(nn.Module):
    """AC4 4x downsample video codec: 192x108 @ 6 fps, 4-frame GOPs, Band W budget (2816)."""

    architecture = ARCHITECTURE

    def __init__(
        self,
        width: int = 32,
        features: int = 16,
        motion_width: int | None = None,
        i_wire_channels: int = 4,
        p_wire_channels: int = 2,
        refine_blocks: int = 2,
        num_offsets: int = 2,
        confidence_threshold: float = 0.02,
        diagonal_gdn: bool = True,
        iframe_mem: bool = True,
        anchor_gate_max: float = 0.35,
        anchor_style_max: float = 0.30,
        deep_tail: bool = False,
    ):
        super().__init__()
        if width % 2 or width < 8 or features < 4:
            raise ValueError("Use even width >=8 and features >=4")
        m_width = motion_width if motion_width is not None else width
        self.config = dict(
            width=width,
            features=features,
            motion_width=m_width,
            i_wire_channels=i_wire_channels,
            p_wire_channels=p_wire_channels,
            refine_blocks=refine_blocks,
            num_offsets=num_offsets,
            confidence_threshold=confidence_threshold,
            diagonal_gdn=diagonal_gdn,
            iframe_mem=iframe_mem,
            anchor_gate_max=anchor_gate_max,
            anchor_style_max=anchor_style_max,
            deep_tail=deep_tail,
        )
        self.i_codec = ICodec(
            width,
            features,
            wire_channels=i_wire_channels,
            iframe_mem=iframe_mem,
            diagonal_gdn=diagonal_gdn,
            anchor_gate_max=anchor_gate_max,
            anchor_style_max=anchor_style_max,
            deep_tail=deep_tail,
        )
        self.motion = MotionCodec(m_width, wire_channels=4)
        self.tx_context = Context(features, width, num_offsets=num_offsets)
        self.rx_context = Context(features, width, num_offsets=num_offsets)
        self.p_analysis = PAnalysis(
            width, features, wire_channels=p_wire_channels, diagonal_gdn=diagonal_gdn
        )
        self.p_from_wire = FixedSynthesis(
            p_wire_channels, CONTENT_VALUES, width * 2, spatial_shape=SPATIAL_SHAPE_4X
        )
        self.tx_projection = PSynthesis(
            width, features, diagonal_gdn=diagonal_gdn, refine_blocks=refine_blocks
        )
        self.p_synthesis = PSynthesis(
            width, features, diagonal_gdn=diagonal_gdn, refine_blocks=refine_blocks
        )
        self.p_rgb = Refine(features, width, deep_tail=deep_tail)
        self.features = features
        self.confidence_threshold = confidence_threshold

        # Streaming memory
        self._tx_state: TXState | None = None
        self._rx_state: RXState | None = None

    def reset(self) -> None:
        """Clear streaming transmitter and receiver histories."""
        self._tx_state = None
        self._rx_state = None

    reset_state = reset

    @staticmethod
    def _position(position: int) -> None:
        if not isinstance(position, int) or not 0 <= position < GOP_FRAMES:
            raise ValueError(f"Scheduled frame position must be 0..{GOP_FRAMES - 1}, got {position}")

    def encode(
        self, frame: torch.Tensor, position: int, state: TXState | None = None
    ) -> tuple[torch.Tensor, TXState]:
        self._position(position)
        if frame.ndim != 4 or frame.shape[1:] != (3, HEIGHT, WIDTH):
            raise ValueError(f"Frame must have shape (B,3,{HEIGHT},{WIDTH}), got {frame.shape}")

        if position == 0:
            prev_context = (state.features, state.original) if state is not None else None
            z, feature = self.i_codec.encode(frame, prev_context=prev_context)
            next_state = TXState(frame, feature, 1)
        else:
            if state is None or state.next_position != position:
                raise ValueError(
                    f"P encoding at pos {position} requires preceding history (got {state})"
                )
            if state.original.shape != frame.shape:
                raise ValueError("TX batch/stream shape changed without reset")
            mz, motion = self.motion.encode(state.original, frame)
            context = self.tx_context(state.original, state.features, motion)
            cz, hidden = self.p_analysis(frame, context)
            feature = self.tx_projection(hidden, context)
            z = torch.cat([mz, cz], 1)
            next_state = TXState(frame, feature, (position + 1) % GOP_FRAMES)
        return z, next_state

    def decode(
        self, received: ReceivedFrame, state: RXState | None = None
    ) -> tuple[torch.Tensor, RXState, torch.Tensor]:
        position, z = received.position, received.payload
        self._position(position)
        if z.ndim != 2 or z.shape[1] != FRAME_SIZES[position]:
            raise ValueError(
                f"Wrong frame payload length: expected {FRAME_SIZES[position]}, got {z.shape[1]}"
            )

        confidence = received.confidence
        if confidence is None:
            confidence = torch.ones_like(z)
        if confidence.shape != z.shape:
            raise ValueError("Confidence must match payload shape")

        finite = torch.isfinite(z) & torch.isfinite(confidence)
        confidence = torch.where(finite, confidence.clamp(0, 1), 0).float()
        z = torch.where(finite, z, 0)
        b = z.shape[0]

        usable = (confidence.mean(1) > self.confidence_threshold) & finite.all(1)
        if position:
            usable = usable & (
                confidence[:, :MOTION_VALUES].mean(1) > self.confidence_threshold
            )
        if received.usable is not None:
            if received.usable.shape != (b,):
                raise ValueError("Usable mask must have shape (B,)")
            usable = usable & received.usable.bool()

        if position == 0:
            prev_context = None
            if state is not None and bool(state.valid.any()):
                prev_context = (state.features, state.reference)

            rgb, features = self.i_codec.decode(z, confidence, prev_context=prev_context)
            old_rgb = torch.full_like(rgb, 0.5)
            old_features = torch.zeros_like(features)
            valid = usable
            next_pos = 1
        else:
            if state is None:
                old_rgb = z.new_full((b, 3, HEIGHT, WIDTH), 0.5)
                old_features = z.new_zeros((b, self.features, PADDED_HEIGHT, PADDED_WIDTH))
                state = RXState(
                    old_rgb,
                    old_features,
                    torch.zeros(b, device=z.device, dtype=torch.bool),
                    position,
                )
            if state.next_position != position or state.reference.shape[0] != b:
                raise ValueError("RX history position or batch changed without refresh")

            old_rgb, old_features = state.reference, state.features
            usable = usable & state.valid
            motion = self.motion.decode(
                z[:, :MOTION_VALUES], confidence[:, :MOTION_VALUES]
            )
            context = self.rx_context(old_rgb, old_features, motion)
            hidden = self.p_from_wire(
                z[:, MOTION_VALUES:], confidence[:, MOTION_VALUES:]
            )
            features = self.p_synthesis(hidden, context)
            rgb = unpad_frame(torch.sigmoid(self.p_rgb(features)))
            valid = state.valid
            next_pos = (position + 1) % GOP_FRAMES

        mask = usable[:, None, None, None]
        rgb = torch.where(mask, rgb, old_rgb)
        features = torch.where(mask, features, old_features)
        return rgb, RXState(rgb, features, valid, next_pos), ~usable

    def encode_gop(
        self,
        video: torch.Tensor,
        state: TXState | None = None,
        retain_state: bool = True,
    ) -> torch.Tensor:
        """Encode a 4-frame GOP (B, 3, 4, 108, 192) to 2,816 real wire values."""
        if video.ndim != 5 or video.shape[1:] != (3, GOP_FRAMES, HEIGHT, WIDTH):
            raise ValueError(
                f"AC4 requires video shape (B, 3, {GOP_FRAMES}, {HEIGHT}, {WIDTH}), got {video.shape}"
            )
        current_state = self._tx_state if (state is None and retain_state) else state
        payloads = []
        for position in range(GOP_FRAMES):
            z, current_state = self.encode(video[:, :, position], position, current_state)
            payloads.append(z)
        if retain_state:
            self._tx_state = current_state
        return torch.cat(payloads, 1)

    def decode_gop(
        self,
        payload: torch.Tensor,
        confidence: torch.Tensor | None = None,
        *,
        usable: torch.Tensor | None = None,
        state: RXState | None = None,
        retain_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode 2,816 wire coordinates to a 4-frame GOP (B, 3, 4, 108, 192) and outages."""
        if payload.ndim != 2 or payload.shape[1] != GOP_VALUES:
            raise ValueError(
                f"AC4 requires exactly {GOP_VALUES} wire coordinates, got {payload.shape}"
            )
        if confidence is None:
            confidence = torch.ones_like(payload)
        if confidence.shape != payload.shape:
            raise ValueError("Confidence must match payload shape")

        current_state = self._rx_state if (state is None and retain_state) else state
        frames, outages = [], []
        for position, (z, w) in enumerate(
            zip(payload.split(FRAME_SIZES, 1), confidence.split(FRAME_SIZES, 1))
        ):
            frame, current_state, outage = self.decode(
                ReceivedFrame(
                    position, z, w, None if usable is None else usable[:, position]
                ),
                current_state,
            )
            frames.append(frame)
            outages.append(outage)
        if retain_state:
            self._rx_state = current_state
        return torch.stack(frames, 2), torch.stack(outages, 1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass for end-to-end training: (B, 3, 4, 108, 192) -> (B, 3, 4, 108, 192)."""
        wire = self.encode_gop(video, retain_state=False)
        recon, _ = self.decode_gop(wire, retain_state=False)
        return recon


# Backwards compatibility alias:
AC4KCodec = AC4Codec
