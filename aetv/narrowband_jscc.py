"""Joint spatiotemporal JSCC autoencoder for the existing narrowband budgets.

The wire length is the on-air latent count already used by the modem. It is
not increased to buy PSNR:

- 2.2 kHz: 2,816 real coordinates (band W, the V8 / standard-channel GOP)
- 8 kHz: 10,112 real coordinates (band U, the V7 GOP)
- 16 kHz: 19,200 real coordinates (band A, the AC16 GOP)

A 6-frame 192x108 GOP is encoded as one vector. A channel-wise spacetime mix
gives every coordinate a full-GOP receptive field, and a block-diagonal linear
map realizes a full-rank projection onto that exact budget. Unit-RMS
normalization matches the analog modem. The decoder sees the noisy coordinates
together with a per-coordinate Wiener confidence.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import BAND_A, LATENTS_PER_GOP_U, LATENTS_PER_GOP_W

ARCHITECTURE = "narrowband-jscc-v1"

HEIGHT = 108
WIDTH = 192
GOP_FRAMES = 6
PAD_TOP = 2
PADDED_HEIGHT = 112
TOKENS = GOP_FRAMES * 7 * 12  # 16x spatial grid of the padded frame

# (kHz, latent budget, modem mode whose GOP carries exactly that many reals)
BANDWIDTHS: tuple[tuple[float, int, str], ...] = (
    (2.2, LATENTS_PER_GOP_W, "V8"),
    (8.0, LATENTS_PER_GOP_U, "V7"),
    (16.0, BAND_A.latents_per_gop, "AC16"),
)


def resolve_bandwidth(khz: float) -> tuple[float, int, str]:
    """Return the existing (kHz, latent budget, modem mode) for a request."""
    for value, budget, mode in BANDWIDTHS:
        if abs(value - float(khz)) < 0.05:
            return value, budget, mode
    known = ", ".join(str(item[0]) for item in BANDWIDTHS)
    raise ValueError(f"bandwidth {khz} kHz is outside the existing bands ({known})")


def pad_gop(video: torch.Tensor) -> torch.Tensor:
    """Pad 108-row frames to 112 so four stride-2 stages land on 7x12."""
    if video.shape[-2] == HEIGHT:
        return F.pad(video, (0, 0, PAD_TOP, PAD_TOP, 0, 0), mode="replicate")
    return video


def unpad_gop(video: torch.Tensor) -> torch.Tensor:
    if video.shape[-2] == PADDED_HEIGHT:
        return video[..., PAD_TOP : PAD_TOP + HEIGHT, :]
    return video


def unit_rms(wire: torch.Tensor) -> torch.Tensor:
    power = wire.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return wire / power.sqrt().to(wire.dtype)


def impair_wire(
    wire: torch.Tensor, snr_db: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """AWGN at the given SNR for a unit-RMS coordinate vector.

    Matches the AC6 evaluation convention: noise sigma is ``10^(-snr/20)`` and
    confidence is the scalar Wiener gain ``snr / (snr + 1)``.
    """
    if not torch.is_tensor(snr_db):
        snr_db = torch.full((wire.shape[0],), float(snr_db), device=wire.device, dtype=torch.float32)
    else:
        snr_db = snr_db.to(device=wire.device, dtype=torch.float32).reshape(-1)
    snr_linear = torch.pow(10.0, snr_db / 10.0).reshape(-1, 1)
    noise = torch.randn_like(wire) / snr_linear.sqrt().to(wire.dtype)
    confidence = (snr_linear / (snr_linear + 1.0)).to(wire.dtype).expand_as(wire)
    return wire + noise, confidence


def psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    mse = (reference.float() - reconstruction.float()).square().mean().item()
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def global_ssim(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """Per-frame global SSIM, the statistic reported by the AC6 runs."""
    c1, c2 = 0.01**2, 0.03**2
    dims = (-2, -1)
    mu_x = reference.mean(dim=dims, keepdim=True)
    mu_y = reconstruction.mean(dim=dims, keepdim=True)
    var_x = reference.square().mean(dim=dims, keepdim=True) - mu_x.square()
    var_y = reconstruction.square().mean(dim=dims, keepdim=True) - mu_y.square()
    cov = (reference * reconstruction).mean(dim=dims, keepdim=True) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x.clamp_min(0) + var_y.clamp_min(0) + c2)
    )
    return float(value.clamp(0, 1).mean().item())


def _even_splits(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def choose_blocks(in_features: int, out_features: int, target: int = 96) -> int:
    """Pick a block count whose block-diagonal map keeps rank min(in, out)."""
    if in_features < 1 or out_features < 1:
        raise ValueError("projection sizes must be positive")
    large = max(in_features, out_features)
    small = min(in_features, out_features)
    blocks = max(1, small // target)
    while blocks > 1:
        if large // blocks >= math.ceil(small / blocks):
            break
        blocks -= 1
    return blocks


class MixedProjection(nn.Module):
    """Block-diagonal map plus a low-rank path over the whole feature vector.

    The low-rank branch starts at zero so early steps follow the local map.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 384):
        super().__init__()
        self.local = BlockLinear(in_features, out_features)
        hidden = min(rank, in_features, out_features)
        self.down = nn.Linear(in_features, hidden)
        self.up = nn.Linear(hidden, out_features)
        nn.init.xavier_uniform_(self.down.weight, gain=0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        global_term = self.up(F.gelu(self.down(value)))
        return self.local(value) + global_term


class BlockLinear(nn.Module):
    """Block-diagonal linear map.

    Blocks partition both sides. When every block is at least square along the
    smaller side, the map has rank equal to ``min(in_features, out_features)``,
    so a wire budget is fully used without a dense ``in * out`` matrix.
    """

    def __init__(self, in_features: int, out_features: int, target: int = 96):
        super().__init__()
        blocks = choose_blocks(in_features, out_features, target)
        self.in_splits = _even_splits(in_features, blocks)
        self.out_splits = _even_splits(out_features, blocks)
        reducing = in_features >= out_features
        for src, dst in zip(self.in_splits, self.out_splits):
            if reducing and src < dst:
                raise RuntimeError("reducing block is rank-deficient")
            if not reducing and dst < src:
                raise RuntimeError("expanding block is rank-deficient")
        self.blocks = nn.ModuleList(
            nn.Linear(src, dst) for src, dst in zip(self.in_splits, self.out_splits)
        )
        for layer in self.blocks:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        parts = value.split(self.in_splits, dim=-1)
        return torch.cat(
            [layer(part) for layer, part in zip(self.blocks, parts)], dim=-1
        )


class SpaceTimeMix(nn.Module):
    """Shared full-GOP mix inside each feature channel. Identity at init."""

    def __init__(self, tokens: int = TOKENS):
        super().__init__()
        self.proj = nn.Linear(tokens, tokens)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        flat = value.flatten(2)
        mixed = flat + self.proj(flat)
        return mixed.reshape(batch, channels, frames, height, width)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        nn.init.normal_(self.body[-1].weight, std=0.02)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class SpatialDown(nn.Module):
    def __init__(self, cin: int, cout: int, first: bool = False):
        super().__init__()
        kernel = (3, 5, 5) if first else (3, 3, 3)
        padding = (1, 2, 2) if first else (1, 1, 1)
        self.conv = nn.Conv3d(cin, cout, kernel, stride=(1, 2, 2), padding=padding)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(value)))


class SpatialUp(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout * 4, 3, padding=1)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = value.shape
        lifted = self.conv(value)
        channels = lifted.shape[1]
        lifted = lifted.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        lifted = F.pixel_shuffle(lifted, 2)
        out_channels = lifted.shape[1]
        lifted = lifted.reshape(batch, frames, out_channels, height * 2, width * 2)
        lifted = lifted.permute(0, 2, 1, 3, 4).contiguous()
        return self.act(self.norm(lifted))


class NarrowbandJSCC(nn.Module):
    """6-frame 192x108 autoencoder with an exact existing RF latent budget."""

    architecture = ARCHITECTURE

    def __init__(self, bandwidth_khz: float = 2.2, width: int = 64):
        super().__init__()
        khz, budget, mode = resolve_bandwidth(bandwidth_khz)
        self.bandwidth_khz = khz
        self.budget = budget
        self.modem_mode = mode
        self.width = width
        if width * TOKENS < budget:
            raise ValueError(
                f"encoder width {width} yields {width * TOKENS} features, "
                f"below the {budget}-coordinate {khz} kHz budget"
            )
        channels = (32, 48, width, width)
        if channels[-1] != width:
            raise ValueError("bottleneck width must match the last encoder stage")

        self.down = nn.ModuleList(
            [
                SpatialDown(3, channels[0], first=True),
                SpatialDown(channels[0], channels[1]),
                SpatialDown(channels[1], channels[2]),
                SpatialDown(channels[2], channels[3]),
            ]
        )
        self.enc_res = nn.ModuleList(ResBlock3d(c) for c in channels)
        self.channel_mix = nn.Conv3d(width, width, 1)
        self.spacetime = SpaceTimeMix(TOKENS)
        feature_dim = width * TOKENS
        mix_rank = min(2048, budget)
        self.to_wire = MixedProjection(feature_dim, budget, rank=mix_rank)
        # Per-coordinate confidence gates the wire before the inverse map.
        self.from_wire = MixedProjection(budget, feature_dim, rank=mix_rank)
        self.unmix = SpaceTimeMix(TOKENS)
        self.restore = nn.Conv3d(width, width, 1)
        self.dec_res = ResBlock3d(width)
        self.up = nn.ModuleList(
            [
                SpatialUp(width, width),
                SpatialUp(width, 48),
                SpatialUp(48, 32),
                SpatialUp(32, 16),
            ]
        )
        self.up_res = nn.ModuleList(
            [ResBlock3d(width), ResBlock3d(48), ResBlock3d(32), ResBlock3d(16)]
        )
        self.coarse = nn.Conv3d(width, 3, 1)
        self.head = nn.Conv3d(16, 3, 3, padding=1)
        nn.init.normal_(self.coarse.weight, std=0.01)
        nn.init.zeros_(self.coarse.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.config = {
            "bandwidth_khz": khz,
            "budget": budget,
            "modem_mode": mode,
            "width": width,
            "gop_frames": GOP_FRAMES,
            "height": HEIGHT,
            "frame_width": WIDTH,
        }

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, 6, 108, 192)`` in ``[0, 1]`` to a unit-RMS wire vector."""
        if video.shape[2:] != (GOP_FRAMES, HEIGHT, WIDTH):
            raise ValueError(
                f"expected (B, 3, {GOP_FRAMES}, {HEIGHT}, {WIDTH}), got {tuple(video.shape)}"
            )
        value = pad_gop(video).mul(2).sub(1)
        for down, residual in zip(self.down, self.enc_res):
            value = residual(down(value))
        value = self.spacetime(self.channel_mix(value))
        flat = value.flatten(1)
        return unit_rms(self.to_wire(flat))

    def decode(self, wire: torch.Tensor, confidence: torch.Tensor | None = None) -> torch.Tensor:
        """Decode a wire vector to ``(B, 3, 6, 108, 192)`` in ``[0, 1]``."""
        if wire.shape[-1] != self.budget:
            raise ValueError(f"wire length {wire.shape[-1]} != budget {self.budget}")
        if confidence is None:
            confidence = torch.ones_like(wire)
        features = self.from_wire(wire * confidence)
        value = features.reshape(wire.shape[0], self.width, GOP_FRAMES, 7, 12)
        value = self.dec_res(self.restore(self.unmix(value)))
        coarse = F.interpolate(
            self.coarse(value),
            size=(GOP_FRAMES, PADDED_HEIGHT, WIDTH),
            mode="trilinear",
            align_corners=False,
        )
        for up, residual in zip(self.up, self.up_res):
            value = residual(up(value))
        logits = coarse + self.head(value)
        return unpad_gop(torch.sigmoid(logits))

    def forward(
        self,
        video: torch.Tensor,
        snr_db: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        wire = self.encode(video)
        confidence = None
        if snr_db is not None:
            wire, confidence = impair_wire(wire, snr_db)
        return self.decode(wire, confidence)
