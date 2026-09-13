"""Fixed-rate asymmetric-context I/P codec; engineering adaptation of 2601.06170v1.

State objects are caller-owned, never registered buffers. The receiver accepts
only received coordinates, receiver-local confidence, and its own history.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "ac16-asymmetric-context-v1"
I_VALUES, MOTION_VALUES, CONTENT_VALUES = 5376, 256, 1280
P_VALUES = MOTION_VALUES + CONTENT_VALUES
GOP_VALUES = I_VALUES + 9 * P_VALUES
P_WEIGHTS = (0.5, 1.2, 0.9, 1.2, 0.5, 1.2, 0.9, 1.2, 0.5)
FRAME_SIZES = (I_VALUES,) + (P_VALUES,) * 9


def normalize_payload(x):
    """Per-source, per-frame/stream RMS; no later frames or channel input."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True).clamp_min(1e-8))


class GDN(nn.Module):
    """Positive cross-channel generalized divisive normalization."""

    def __init__(self, channels, inverse=False):
        super().__init__()
        self.beta = nn.Parameter(torch.full((channels,), 0.54132485))
        self.gamma = nn.Parameter(torch.eye(channels) * 3.0 - 5.0)
        self.inverse = inverse

    def forward(self, x):
        dtype = x.dtype
        # Accumulate in FP32, including under mixed precision.
        with torch.autocast(x.device.type, enabled=False):
            norm = F.conv2d(
                x.float().square(),
                F.softplus(self.gamma)[:, :, None, None],
                F.softplus(self.beta) + 1e-6,
            ).sqrt()
            y = x.float() * norm if self.inverse else x.float() / norm
        return y.to(dtype)


class Residual(nn.Module):
    def __init__(self, cin, cout=None, down=False, up=False):
        super().__init__()
        cout = cout or cin
        if down and up:
            raise ValueError("Residual cannot both downsample and upsample")

        def conv(a, b):
            return (
                nn.Sequential(nn.Conv2d(a, b * 4, 3, padding=1), nn.PixelShuffle(2))
                if up
                else nn.Conv2d(a, b, 3, stride=2 if down else 1, padding=1)
            )

        self.body = nn.Sequential(
            conv(cin, cout), nn.LeakyReLU(0.1), nn.Conv2d(cout, cout, 3, padding=1)
        )
        self.skip = conv(cin, cout) if cin != cout or down or up else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.body(x)


class DepthConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class UNet(nn.Module):
    """All skips are computed at the endpoint executing this module."""

    def __init__(self, channels):
        super().__init__()
        self.first = Residual(channels)
        self.down = Residual(channels, channels * 2, down=True)
        self.mid = nn.Sequential(DepthConv(channels * 2), Residual(channels * 2))
        self.up = Residual(channels * 2, channels, up=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), Residual(channels)
        )

    def forward(self, x):
        skip = self.first(x)
        low = self.up(self.mid(self.down(skip)))
        return x + self.fuse(torch.cat([skip, low], 1))


class FixedAnalysis(nn.Module):
    """Learned exact-dimensional projection; every input location participates."""

    def __init__(self, cin, channels, coordinates):
        super().__init__()
        if coordinates % channels:
            raise ValueError("Projection dimensions must divide exactly")
        self.map = nn.Conv2d(cin, channels, 1)
        self.spatial = nn.Linear(9 * 16, coordinates // channels)

    def forward(self, x):
        if x.shape[-2:] != (9, 16):
            raise ValueError("AC16 analysis requires a 9x16 feature plane")
        return self.spatial(self.map(x).flatten(2)).flatten(1)


class FixedSynthesis(nn.Module):
    def __init__(self, channels, coordinates, cout):
        super().__init__()
        self.channels, self.coordinates = channels, coordinates
        self.spatial = nn.Linear(coordinates // channels, 9 * 16)
        self.map = nn.Conv2d(channels * 2, cout, 1)

    def forward(self, z, confidence):
        b = z.shape[0]
        # Confidence is observed data; use the same learned spatial map for
        # values and confidence, then learn the interpretation jointly.
        a = self.spatial((z * confidence).reshape(b, self.channels, -1))
        w = self.spatial(confidence.reshape(b, self.channels, -1))
        return self.map(torch.cat([a, w], 1).reshape(b, self.channels * 2, 9, 16))


class ICodec(nn.Module):
    def __init__(self, width=32, features=16):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        blocks = []
        previous = 3
        for channels in (a, b, c):
            blocks.extend(
                [
                    Residual(previous, channels, down=True),
                    DepthConv(channels),
                    GDN(channels),
                ]
            )
            previous = channels
        self.analysis = nn.Sequential(*blocks, nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.to_wire = FixedAnalysis(c, 48, I_VALUES)
        self.from_wire = FixedSynthesis(48, I_VALUES, c)
        self.synthesis = nn.Sequential(
            DepthConv(c),
            Residual(c, c, up=True),
            GDN(c, True),
            DepthConv(c),
            Residual(c, b, up=True),
            GDN(b, True),
            DepthConv(b),
            Residual(b, a, up=True),
            GDN(a, True),
            DepthConv(a),
            Residual(a, features, up=True),
            UNet(features),
        )
        self.rgb = nn.Conv2d(features, 3, 3, padding=1)
        self.tx_projection = nn.Conv2d(3, features, 3, padding=1)

    def encode(self, x):
        return normalize_payload(
            self.to_wire(self.analysis(x * 2 - 1))
        ), self.tx_projection(x * 2 - 1)

    def decode(self, z, confidence):
        features = self.synthesis(self.from_wire(z, confidence))
        return torch.sigmoid(self.rgb(features)), features


def warp(x, motion):
    """Backward optical flow in full-raster pixels; grid local to this call."""
    b, _, h, w = x.shape
    mh, mw = motion.shape[-2:]
    flow = F.interpolate(motion.float(), (h, w), mode="bilinear", align_corners=False)
    ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) * 2 / h - 1
    xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) * 2 / w - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], -1)[None].expand(b, -1, -1, -1)
    scale = flow.new_tensor([2 / mw, 2 / mh])[None, :, None, None]
    grid = grid + (flow * scale).permute(0, 2, 3, 1)
    return F.grid_sample(
        x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).to(x.dtype)


class MotionCodec(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        # Small learned pyramid flow estimator, trained only by reconstruction.
        self.estimate = nn.Sequential(
            nn.Conv2d(6, width, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            Residual(width, width, down=True),
            Residual(width),
            nn.Conv2d(width, 2, 3, padding=1),
        )
        self.analysis = nn.Sequential(
            Residual(2, width, down=True),
            DepthConv(width),
            Residual(width, width, down=True),
            DepthConv(width),
            Residual(width, width, down=True),
            DepthConv(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
        )
        self.to_wire = FixedAnalysis(width, 4, MOTION_VALUES)
        self.from_wire = FixedSynthesis(4, MOTION_VALUES, width)
        self.synthesis = nn.Sequential(
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            DepthConv(width),
            Residual(width, width, up=True),
            nn.Conv2d(width, 2, 1),
        )

    def encode(self, previous, current):
        flow = F.interpolate(
            self.estimate(torch.cat([previous, current], 1) * 2 - 1),
            current.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        flow = 24 * torch.tanh(flow / 24)
        return normalize_payload(self.to_wire(self.analysis(flow / 24))), flow

    def decode(self, z, confidence):
        return 24 * torch.tanh(self.synthesis(self.from_wire(z, confidence)) / 24)


class Context(nn.Module):
    """Two learned offset hypotheses, fused at full, half and quarter scales."""

    def __init__(self, features=16, width=32):
        super().__init__()
        self.reference = nn.Conv2d(features + 3, features, 3, padding=1)
        self.offsets = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features, 6, 3, padding=1),
        )
        self.full = Residual(features)
        self.half_scale = Residual(features, width, down=True)
        self.quarter = Residual(width, width * 3 // 2, down=True)

    def forward(self, reference, features, motion):
        source = self.reference(torch.cat([reference * 2 - 1, features], 1))
        initial = warp(source, motion)
        offsets = self.offsets(initial)
        weights = offsets[:, 4:].softmax(1)
        aligned = sum(
            warp(source, motion + 2 * torch.tanh(offsets[:, 2 * i : 2 * i + 2]))
            * weights[:, i : i + 1]
            for i in range(2)
        )
        full = self.full(aligned)
        half = self.half_scale(full)
        return full, half, self.quarter(half)


class PAnalysis(nn.Module):
    def __init__(self, width=32, features=16):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        self.first = nn.Sequential(
            nn.Conv2d(3 + features, a, 3, stride=2, padding=1), GDN(a)
        )
        self.second = nn.Sequential(
            Residual(a * 2), nn.Conv2d(a * 2, b, 3, stride=2, padding=1), GDN(b)
        )
        self.third = nn.Sequential(
            Residual(b * 2),
            nn.Conv2d(b * 2, c, 3, stride=2, padding=1),
            GDN(c),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
        )
        self.to_wire = FixedAnalysis(c, 16, CONTENT_VALUES)

    def forward(self, x, contexts):
        full, half, quarter = contexts
        h = self.first(torch.cat([x * 2 - 1, full], 1))
        h = self.second(torch.cat([h, half], 1))
        h = self.third(torch.cat([h, quarter], 1))
        return normalize_payload(self.to_wire(h)), h


class PSynthesis(nn.Module):
    def __init__(self, width=32, features=16):
        super().__init__()
        a, b, c = width, width * 3 // 2, width * 2
        self.first = nn.Sequential(
            Residual(c, c, up=True), GDN(c, True), Residual(c, b, up=True), GDN(b, True)
        )
        self.second = nn.Sequential(
            Residual(b * 2), Residual(b * 2, a, up=True), GDN(a, True)
        )
        self.third = nn.Sequential(Residual(a * 2), Residual(a * 2, features, up=True))
        self.refine = nn.Sequential(
            nn.Conv2d(features * 2, features, 3, padding=1),
            DepthConv(features),
            UNet(features),
            UNet(features),
        )

    def forward(self, h, contexts):
        full, half, quarter = contexts
        h = self.second(torch.cat([self.first(h), quarter], 1))
        h = self.third(torch.cat([h, half], 1))
        return self.refine(torch.cat([h, full], 1))


@dataclass(frozen=True)
class TXState:
    original: torch.Tensor
    features: torch.Tensor
    next_position: int

    def detach(self):
        return TXState(
            self.original.detach(), self.features.detach(), self.next_position
        )


@dataclass(frozen=True)
class RXState:
    reference: torch.Tensor
    features: torch.Tensor
    valid: torch.Tensor
    next_position: int

    def detach(self):
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
    def frame_type(self):
        return "I" if self.position == 0 else "P"


class AsymmetricContextCodec(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, width=32, features=16, confidence_threshold=0.02):
        super().__init__()
        if width % 2 or width < 8 or features < 4:
            raise ValueError("Use even width >=8 and features >=4")
        self.config = dict(
            width=width, features=features, confidence_threshold=confidence_threshold
        )
        self.i_codec = ICodec(width, features)
        self.motion = MotionCodec(width)
        self.tx_context = Context(features, width)
        self.rx_context = Context(features, width)
        self.p_analysis = PAnalysis(width, features)
        self.p_from_wire = FixedSynthesis(16, CONTENT_VALUES, width * 2)
        self.tx_projection = PSynthesis(width, features)
        self.p_synthesis = PSynthesis(width, features)
        self.p_rgb = nn.Conv2d(features, 3, 3, padding=1)
        self.features = features
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def reset():
        """A caller resets either endpoint by discarding its state."""
        return None

    @staticmethod
    def _position(position):
        if not isinstance(position, int) or not 0 <= position < 10:
            raise ValueError("Scheduled frame position must be 0..9")

    def encode(self, frame, position, state=None):
        self._position(position)
        if frame.ndim != 4 or frame.shape[1:] != (3, 144, 256):
            raise ValueError("Frame must have shape (B,3,144,256)")
        if position == 0:
            z, feature = self.i_codec.encode(frame)
        else:
            if state is None or state.next_position != position:
                raise ValueError("P encoding requires the preceding TX history")
            if state.original.shape != frame.shape:
                raise ValueError("TX batch/stream shape changed without reset")
            mz, motion = self.motion.encode(state.original, frame)
            context = self.tx_context(state.original, state.features, motion)
            cz, hidden = self.p_analysis(frame, context)
            feature = self.tx_projection(hidden, context)
            z = torch.cat([mz, cz], 1)
        return z, TXState(frame, feature, position + 1)

    def decode(self, received: ReceivedFrame, state=None):
        position, z = received.position, received.payload
        self._position(position)
        if z.ndim != 2 or z.shape[1] != FRAME_SIZES[position]:
            raise ValueError("Wrong fixed frame payload length")
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
            # Motion cannot be reconstructed from a wholly unusable motion substream.
            usable = usable & (
                confidence[:, :MOTION_VALUES].mean(1) > self.confidence_threshold
            )
        if received.usable is not None:
            if received.usable.shape != (b,):
                raise ValueError("Usable mask must have shape (B,)")
            usable = usable & received.usable.bool()
        if position == 0:
            # Refresh disregards ALL previous receiver state, including shape/validity.
            rgb, features = self.i_codec.decode(z, confidence)
            old_rgb, old_features = (
                torch.full_like(rgb, 0.5),
                torch.zeros_like(features),
            )
            valid = usable
        else:
            if state is None:
                old_rgb = z.new_full((b, 3, 144, 256), 0.5)
                old_features = z.new_zeros((b, self.features, 144, 256))
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
            rgb = torch.sigmoid(self.p_rgb(features))
            valid = state.valid
        mask = usable[:, None, None, None]
        rgb = torch.where(mask, rgb, old_rgb)
        features = torch.where(mask, features, old_features)
        return rgb, RXState(rgb, features, valid, position + 1), ~usable

    def encode_gop(self, video):
        if video.ndim != 5 or video.shape[1:] != (3, 10, 144, 256):
            raise ValueError("AC16 requires (B,3,10,144,256)")
        state, payloads = None, []
        for position in range(10):
            z, state = self.encode(video[:, :, position], position, state)
            payloads.append(z)
        return torch.cat(payloads, 1)

    def decode_gop(self, payload, confidence=None, *, usable=None):
        if payload.ndim != 2 or payload.shape[1] != GOP_VALUES:
            raise ValueError("AC16 requires exactly 19200 coordinates")
        if confidence is None:
            confidence = torch.ones_like(payload)
        if confidence.shape != payload.shape:
            raise ValueError("Confidence must match payload")
        state, frames, outages = None, [], []
        for position, (z, w) in enumerate(
            zip(payload.split(FRAME_SIZES, 1), confidence.split(FRAME_SIZES, 1))
        ):
            frame, state, outage = self.decode(
                ReceivedFrame(
                    position, z, w, None if usable is None else usable[:, position]
                ),
                state,
            )
            frames.append(frame)
            outages.append(outage)
        return torch.stack(frames, 2), torch.stack(outages, 1)

    def forward(self, video):
        return self.decode_gop(self.encode_gop(video))[0]


class Refine(nn.Module):
    """Separate nonlinear RGB refinement after propagated decoder features.

    Equation7 describes multiple convolutional layers; exact block count is an
    AETV choice. Keeping refinement separate avoids using RGB logits as cache.
    """

    def __init__(self, features, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(features, width, 3, padding=1),
            nn.LeakyReLU(0.1),
            Residual(width),
            Residual(width),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class AsymmetricContextCodecV2(AsymmetricContextCodec):
    """Corrected synthesis normalization and explicit Refine, diagnostic until gated.

    V1 remains loadable without changing how its saved checkpoints decode.
    Figures3/5 label synthesis normalization GDN, not inverse GDN. No author
    code establishes that distinction; this implements the figures literally.
    Losses, exact payloads, and state ownership are unchanged.
    """

    architecture = "ac16-asymmetric-context-v2"

    def __init__(self, width=32, features=16, confidence_threshold=0.02):
        super().__init__(width, features, confidence_threshold)
        for module in self.modules():
            if isinstance(module, GDN):
                module.inverse = False
        self.i_codec.rgb = Refine(features, width)
        self.p_rgb = Refine(features, width)


class Figure3ICodec(ICodec):
    """I-frame transform with the supplied diagram's fixed channel dimensions.

    Analysis: 128/192/256, then stride-2 Conv(256). Synthesis:
    256/192/128/16, then U-Net and a separate RGB Refine. Decoder GDN is
    divisive as labeled. FixedAnalysis/FixedSynthesis retain AC16's analog
    MEM replacement and its exact 5376-coordinate I-frame payload.
    """

    def __init__(self):
        super().__init__(width=128, features=16)
        for module in self.synthesis.modules():
            if isinstance(module, GDN):
                module.inverse = False
        self.rgb = Refine(16, 128)


class AsymmetricContextCodecV3(AsymmetricContextCodecV2):
    """Diagram-sized I transform; width controls only the remaining P path.

    Legacy V1/V2 weights keep their original shapes and decoding behavior.
    V3 needs fresh training and keeps V2's P-frame normalization/refinement.
    """

    architecture = "ac16-asymmetric-context-v3"

    def __init__(self, width=32, features=16, confidence_threshold=0.02):
        if features != 16:
            raise ValueError("AC16 V3 requires the diagram's 16 propagated features")
        super().__init__(width, features, confidence_threshold)
        self.i_codec = Figure3ICodec()


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
    """GDN with diagonal initialization and bounded square parameterization.

    Uses the nonnegative parameterization described by CompressAI's GDN:
    beta=1, gamma=0.1*identity, pedestal=(2**-18)**2, beta floor 1e-6.
    This is a parameterization AND initialization intervention, not just a
    different random seed. All accumulation still takes place in FP32.
    """

    pedestal = (2**-18) ** 2

    def __init__(self, channels, inverse=False):
        super().__init__(channels, inverse)
        self.beta = nn.Parameter(torch.full((channels,), (1 + self.pedestal) ** 0.5))
        self.gamma = nn.Parameter((0.1 * torch.eye(channels) + self.pedestal).sqrt())

    def forward(self, x):
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


class MEMTransform(nn.Sequential):
    """Figure 6 signal branch without the entropy policy or selection mask.

    Two Resblock/Conv/GDN/LeakyReLU groups, followed by an output Conv.
    All stages keep the 9x16 plane and 256 channels; AC16's existing exact
    projection follows this encoder and precedes this decoder. This placement
    is an explicit fixed-rate adaptation, not recovered author source code.
    """

    def __init__(self, channels=256):
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


class AsymmetricContextCodecV4(AsymmetricContextCodecV3):
    """Opt-in I-only MEM/GDN ablations; P modules remain exactly V3.

    Building V3 before the extra modules keeps shared convolution/projection
    initialization identical for a matched seed. Flags are saved in config.
    """

    architecture = "ac16-asymmetric-context-v4"

    def __init__(
        self,
        width=32,
        features=16,
        confidence_threshold=0.02,
        iframe_mem=True,
        diagonal_gdn=True,
    ):
        super().__init__(width, features, confidence_threshold)
        self.config.update(iframe_mem=iframe_mem, diagonal_gdn=diagonal_gdn)
        if iframe_mem:
            # Preserve names/initial weights of every original transform layer.
            self.i_codec.analysis.add_module("mem", MEMTransform())
            self.i_codec.synthesis = nn.Sequential(
                OrderedDict(
                    [("mem", MEMTransform()), *self.i_codec.synthesis.named_children()]
                )
            )
        if diagonal_gdn:

            def replace(parent):
                for name, child in list(parent.named_children()):
                    if isinstance(child, GDN):
                        setattr(
                            parent, name, DiagonalGDN(len(child.beta), child.inverse)
                        )
                    else:
                        replace(child)

            replace(self.i_codec)


CANDIDATE_VERSIONS = {
    1: AsymmetricContextCodec,
    2: AsymmetricContextCodecV2,
    3: AsymmetricContextCodecV3,
    4: AsymmetricContextCodecV4,
}
CANDIDATE_ARCHITECTURES = {cls.architecture: cls for cls in CANDIDATE_VERSIONS.values()}
