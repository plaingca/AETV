"""Receiver for the frozen 2.2 kHz AC2K wire after a Watterson fade.

The temporally common residual is worth about two decibels on a real MPP
decode only when that residual arrives exactly, on top of a full AC2K picture.
Packing it into the 2,816 reals — quantization dither, a patch code in place
of AC2K coordinates, or a half-rate repetition — loses once the channel fades,
because the detail is smaller than the damage of giving up coordinates. A
network that invents the erased coordinates does not generalize: trusting a
wrong fill lowers PSNR.

Two receiver corrections do survive, and neither one rewrites the on-air
vector. Modem confidence on this multipath profile is slightly pessimistic, so
carrier-varying weights are scaled (flat AWGN weights are left alone). A
zero-init convolutional head then adds the part of the temporal-mean residual
it can see from the four decoded frames. ``CarrierFill`` is available as a
carrier-wise residual and stays at its zero initialization, where it is the
identity.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .ac2k_psnr import load_psnr_checkpoint
from .framing import GOP_DEINTERLEAVER_W

GOP_VALUES = 2816
LATENT_CARRIERS = 44
COORDS_PER_CARRIER = GOP_VALUES // LATENT_CARRIERS
CARRIER_HZ = 50.0
MPP_DELAY_S = 0.002


def carrier_index() -> torch.Tensor:
    """OFDM latent-carrier index of each of the 2,816 real coordinates."""
    raw = torch.as_tensor(GOP_DEINTERLEAVER_W.copy(), dtype=torch.long)
    if raw.numel() != GOP_VALUES:
        raise RuntimeError(f"interleaver length {raw.numel()} != {GOP_VALUES}")
    return (raw // 2) % LATENT_CARRIERS


def carrier_layout(index: torch.Tensor | None = None) -> torch.Tensor:
    """Coordinate indices grouped as ``(44, 64)``, one row per latent carrier."""
    carriers = carrier_index() if index is None else index
    rows = []
    for carrier in range(LATENT_CARRIERS):
        coords = torch.nonzero(carriers == carrier, as_tuple=False).flatten()
        if coords.numel() != COORDS_PER_CARRIER:
            raise RuntimeError(
                f"carrier {carrier} has {coords.numel()} coordinates, expected {COORDS_PER_CARRIER}"
            )
        rows.append(coords)
    return torch.stack(rows)


def apply_carrier_fade(
    wire: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable stand-in for an equalized two-path fade, used as augmentation.

    A carrier of magnitude ``|H|`` is returned as ``x + N(0, sigma/|H|)`` with
    confidence ``|H|^2 / (|H|^2 + sigma^2)``. This is not the headline channel.
    Scoring uses ``hfchannel.emulate`` on the V8 waveform.
    """
    if wire.shape[-1] != GOP_VALUES:
        raise ValueError(f"wire length {wire.shape[-1]} != {GOP_VALUES}")
    batch = wire.shape[0]
    device = wire.device
    g1 = torch.randn(batch, 1, device=device, generator=generator) + 1j * torch.randn(
        batch, 1, device=device, generator=generator
    )
    g2 = torch.randn(batch, 1, device=device, generator=generator) + 1j * torch.randn(
        batch, 1, device=device, generator=generator
    )
    tones = torch.arange(LATENT_CARRIERS, device=device, dtype=torch.float32)
    phase = torch.exp(-2j * math.pi * tones * CARRIER_HZ * MPP_DELAY_S)
    response = (g1 + g2 * phase) / math.sqrt(2)
    power = response.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-8)
    magnitude = (response / power.sqrt()).abs()
    carriers = carrier_index().to(device)
    tone = magnitude[:, carriers].clamp_min(0.08)
    noise = torch.randn(wire.shape, device=device, dtype=wire.dtype, generator=generator)
    received = wire + noise * (sigma / tone)
    strength = tone.square()
    confidence = (strength / (strength + sigma * sigma)).to(wire.dtype)
    return received, confidence


class CarrierFill(nn.Module):
    """Predict faded coordinates from the carriers the fade left intact."""

    def __init__(self, hidden: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.05):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden {hidden} must be divisible by heads {heads}")
        layout = carrier_layout()
        inverse = torch.empty(GOP_VALUES, dtype=torch.long)
        inverse[layout.reshape(-1)] = torch.arange(GOP_VALUES)
        self.register_buffer("layout", layout)
        self.register_buffer("inverse", inverse)
        self.position = nn.Parameter(torch.randn(1, LATENT_CARRIERS, hidden) * 0.02)
        self.input = nn.Linear(COORDS_PER_CARRIER * 2, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.output = nn.Linear(hidden, COORDS_PER_CARRIER)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, received: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        if received.shape[-1] != GOP_VALUES or confidence.shape != received.shape:
            raise ValueError(
                f"expected matching (B, {GOP_VALUES}) tensors, got {tuple(received.shape)} and {tuple(confidence.shape)}"
            )
        weight = confidence.clamp(0, 1)
        grouped_y = received[:, self.layout]
        grouped_w = weight[:, self.layout]
        # Clamp only the network input. The skip path keeps the modem's value
        # so a confident coordinate is bitwise unchanged.
        grouped_in = grouped_y.clamp(-6, 6)
        hidden = self.encoder(self.input(torch.cat([grouped_in, grouped_w], dim=-1)) + self.position)
        # Low confidence is where the equalizer amplified noise. High confidence
        # coordinates already match the transmitted wire and must not move.
        correction = (1.0 - grouped_w) * self.output(hidden)
        restored = (grouped_y + correction).reshape(received.shape[0], GOP_VALUES)
        return restored[:, self.inverse]


class CommonHead(nn.Module):
    """Add one residual plane, shared by the four frames, on top of the decode.

    The plane is the part of the temporal-mean residual that can be estimated
    from the reconstruction itself. It is zero-initialized, so an untrained
    head reproduces AC2K. It is not a second description on the wire: every
    explicit packing of that residual into the 2,816 reals lost more under
    multipath than the residual returned.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.collapse = nn.Conv3d(3, channels, kernel_size=(4, 3, 3), padding=(0, 1, 1))
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        shared = self.output(self.refine(F.gelu(self.collapse(video).squeeze(2))))
        return (video + shared.unsqueeze(2)).clamp(0, 1)


class MultipathFill(nn.Module):
    architecture = "multipath-fill-v1"

    def __init__(
        self,
        ac2k_path: str = "models/ac2k-psnr-2.2khz-best.pt",
        hidden: int = 128,
        layers: int = 3,
    ):
        super().__init__()
        self.ac2k_path = str(ac2k_path)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.base, _ = load_psnr_checkpoint(ac2k_path, "cpu")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.fill = CarrierFill(hidden=hidden, layers=layers)
        self.pixels = CommonHead()
        # Modem weights are a bit pessimistic on MPP: scaling them by ~1.5
        # raised PSNR on two disjoint clip sets. Flat AWGN has no spread, so
        # the scale is not applied there and the AWGN table stays on AC2K.
        self.confidence_gain = 1.5

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def reset(self) -> None:
        self.base.reset()

    def encode_gop(self, video: torch.Tensor, retain_state: bool = True, **kwargs) -> torch.Tensor:
        del kwargs
        return self.base.encode_gop(video, retain_state=retain_state)

    def decode_gop(
        self,
        wire: torch.Tensor,
        confidence: torch.Tensor | None = None,
        retain_state: bool = True,
        **kwargs,
    ):
        del kwargs
        if wire.shape[-1] != GOP_VALUES:
            raise ValueError(f"wire length {wire.shape[-1]} != {GOP_VALUES}")
        if confidence is None:
            confidence = torch.ones_like(wire)
        confidence = self._scale_confidence(confidence)
        restored = self.fill(wire, confidence)
        recon, outage = self.base.decode_gop(restored, confidence=confidence, retain_state=retain_state)
        return self.pixels(recon), outage

    def _scale_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        spread = confidence.amax(dim=-1, keepdim=True) - confidence.amin(dim=-1, keepdim=True)
        scaled = (confidence * self.confidence_gain).clamp(0, 1)
        return torch.where(spread > 0.05, scaled, confidence)


def load_multipath_fill(
    path: str | Path, device: torch.device | str = "cpu", ac2k_path: str | None = None
) -> tuple[MultipathFill, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MultipathFill(
        ac2k_path=ac2k_path or payload.get("ac2k_checkpoint", "models/ac2k-psnr-2.2khz-best.pt"),
        hidden=int(payload.get("hidden", 128)),
        layers=int(payload.get("layers", 3)),
    )
    model.fill.load_state_dict(payload["fill"])
    model.pixels.load_state_dict(payload["pixels"])
    model.confidence_gain = float(payload.get("confidence_gain", 1.5))
    model.to(device).eval()
    return model, payload
