"""LTX-Video latent transport over the fixed V8 2,816-value RF budget.

The released LTX VAE maps a six-frame 192x108 V8 GOP (tail padded to the
VAE's valid nine-frame temporal extent) to a 128x2x4x6 latent tensor.  This
module learns a compact, reliability-aware joint source/channel mapping from
those 6,144 values to the 2,816 real values carried by one V8 RF GOP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LTX_REPO = "Lightricks/LTX-Video"
V8_VIDEO_SHAPE = (6, 108, 192)
LTX_PADDED_FRAMES = 9
LTX_LATENT_SHAPE = (128, 2, 4, 6)
V8_CHANNEL_VALUES = 2816


def prepare_ltx_video(video: torch.Tensor) -> torch.Tensor:
    """Convert a [0,1] V8 GOP to LTX's [-1,1] nine-frame input."""
    if video.ndim != 5 or tuple(video.shape[1:]) != (3, *V8_VIDEO_SHAPE):
        raise ValueError(
            f"expected (B,3,{V8_VIDEO_SHAPE[0]},{V8_VIDEO_SHAPE[1]},"
            f"{V8_VIDEO_SHAPE[2]}), got {tuple(video.shape)}"
        )
    tail = video[:, :, -1:].expand(-1, -1, LTX_PADDED_FRAMES - video.shape[2], -1, -1)
    return torch.cat((video, tail), dim=2).mul(2).sub(1)


def finish_ltx_video(decoded: torch.Tensor) -> torch.Tensor:
    """Crop LTX's padded output back to the exact V8 GOP and [0,1]."""
    frames, height, width = V8_VIDEO_SHAPE
    return decoded[:, :, :frames, :height, :width].float().add(1).mul(0.5).clamp(0, 1)


class ChannelResidual(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(self.norm(value))


@dataclass(frozen=True)
class LTXChannelGeometry:
    latent_shape: tuple[int, int, int, int] = LTX_LATENT_SHAPE
    channel_values: int = V8_CHANNEL_VALUES
    # Preserve LTX's complete 128-channel semantic basis and collapse the two
    # temporal slices instead. Halving the channel basis produced low latent
    # MSE but catastrophic decoded pictures because LTX's decoder is extremely
    # sensitive to cross-channel errors.
    bottleneck_channels: int = 128
    bottleneck_frames: int = 1

    @property
    def bottleneck_values(self) -> int:
        return self.bottleneck_channels * self.bottleneck_frames * self.latent_shape[2] * self.latent_shape[3]


class LTXV8ChannelAdapter(nn.Module):
    """Reliability-aware 6,144 <-> 2,816 LTX latent transport adapter."""

    def __init__(self, geometry: LTXChannelGeometry | None = None):
        super().__init__()
        self.geometry = geometry or LTXChannelGeometry()
        channels = self.geometry.latent_shape[0]
        native_values = math.prod(self.geometry.latent_shape)

        self.encoder_residual = ChannelResidual(channels)
        # A global projection is intentional. Local channel/time factorizations
        # achieved low latent MSE while destroying decoded images: important LTX
        # directions span channels, time and space. This layer can learn the
        # actual decoder-sensitive rank-2,816 subspace.
        self.rate_encode = nn.Linear(native_values, self.geometry.channel_values)

        # Confidence conditions each received coordinate without a dense second
        # confidence projection. At confidence=1 this is exactly the received
        # symbol; low-confidence coordinates learn whether to attenuate, retain,
        # or replace the observation before the shared inverse projection.
        self.confidence_gain = nn.Parameter(torch.zeros(self.geometry.channel_values))
        self.missing_symbol = nn.Parameter(torch.zeros(self.geometry.channel_values))
        self.rate_decode = nn.Linear(self.geometry.channel_values, native_values)
        self.decoder_residual = ChannelResidual(channels)

        self.register_buffer("latent_mean", torch.zeros(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, channels, 1, 1, 1))
        self.register_buffer("stats_ready", torch.tensor(False))
        self._reset_rate_layers()

    def _reset_rate_layers(self) -> None:
        with torch.no_grad():
            nn.init.normal_(
                self.rate_encode.weight,
                std=1.0 / math.sqrt(self.rate_encode.in_features),
            )
            self.rate_decode.weight.copy_(self.rate_encode.weight.transpose(0, 1))
            self.rate_encode.bias.zero_()
            self.rate_decode.bias.zero_()

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (self.geometry.latent_shape[0],)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError(f"expected per-channel statistics shaped {expected}")
        self.latent_mean.copy_(mean.float().view(1, -1, 1, 1, 1))
        self.latent_std.copy_(std.float().clamp_min(1e-4).view(1, -1, 1, 1, 1))
        self.stats_ready.fill_(True)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean.to(latent)) / self.latent_std.to(latent)

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std.to(latent) + self.latent_mean.to(latent)

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape[1:]) != self.geometry.latent_shape:
            raise ValueError(f"expected latent shape {self.geometry.latent_shape}, got {tuple(latent.shape[1:])}")
        value = self.encoder_residual(self.normalize(latent))
        symbols = self.rate_encode(value.flatten(1))
        # The analog modem contract is one unit-RMS real vector per RF GOP.
        return symbols / symbols.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()

    def decode(self, received: torch.Tensor, confidence: torch.Tensor | None = None) -> torch.Tensor:
        if confidence is None:
            confidence = torch.ones_like(received)
        confidence = confidence.to(received).clamp(0, 1)
        gain = 1 + torch.tanh(self.confidence_gain).to(received) * (1 - confidence)
        conditioned = received * gain + self.missing_symbol.to(received) * (1 - confidence)
        value = self.rate_decode(conditioned).reshape(
            received.shape[0], *self.geometry.latent_shape
        )
        return self.denormalize(self.decoder_residual(value))

    def forward(
        self, latent: torch.Tensor, received: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        symbols = self.encode(latent)
        if received is None:
            received = symbols
        return self.decode(received, confidence), symbols


def latent_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable source-referenced loss for the adapter pretraining stages."""
    huber = F.smooth_l1_loss(reconstructed.float(), target.float(), beta=0.1)
    cosine = 1 - F.cosine_similarity(
        reconstructed.float().flatten(1), target.float().flatten(1), dim=1
    ).mean()
    return huber + 0.1 * cosine
