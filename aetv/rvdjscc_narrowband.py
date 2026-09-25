"""Narrowband adaptation of RVDJSCC (arXiv:2601.01729).

The paper sends a 4-frame GOP over OFDM: the last frame is a key frame with
three times the symbols of each interpolation frame (packet counts 6, 2, 2, 2),
interpolation frames are conditioned on multi-scale Gaussian-warped context,
and a lightweight SNR-gated denoiser cleans the latent before the frame decoder.

This port keeps those three pieces and the paper's loss (frame MSE plus 0.7
times latent denoising MSE). It does not use the paper's 256-subcarrier modem.
The wire is the existing real coordinate budget, unit-RMS, with the project's
AWGN. 108 is not divisible by 16, so the strided towers use two stride-2 stages
instead of the paper's four. Channel width is 32 rather than 256 so the model
fits the 2,816-coordinate bottleneck on this GPU.

Budgets, from the 6:2:2:2 split with any remainder given to the key frame:

- 2.2 kHz, 2,816 coordinates: key 1,412, interpolation 468
- 8 kHz, 10,112 coordinates: key 5,060, interpolation 1,684
- 16 kHz, 19,200 coordinates: key 9,600, interpolation 3,200
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .narrowband_jscc import resolve_bandwidth, unit_rms

HEIGHT = 108
WIDTH = 192
GOP_FRAMES = 4
NOMINAL_SNR_DB = 15.0
CLEAN_SNR_DB = 40.0
FEAT_H = 27
FEAT_W = 48
REDUCED = 8
FLAT = FEAT_H * FEAT_W * REDUCED


def packet_split(budget: int) -> tuple[int, int]:
    """Return (key_length, interpolation_length) for a 6:2:2:2 GOP split."""
    if budget < 12:
        raise ValueError(f"budget {budget} is smaller than one symbol per packet share")
    base = budget // 12
    key = 6 * base + (budget % 12)
    interp = 2 * base
    if key + 3 * interp != budget:
        raise RuntimeError(f"split of {budget} did not sum to the budget")
    return key, interp


class AFModule(nn.Module):
    """Channel gate conditioned on a scalar SNR, as in the paper's AF module."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels + 1, channels),
            nn.LeakyReLU(0.2),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        context = value.mean(dim=tuple(range(2, value.ndim)))
        snr = (snr_db.reshape(-1, 1).to(dtype=value.dtype) / 20.0)
        if snr.shape[0] == 1 and context.shape[0] != 1:
            snr = snr.expand(context.shape[0], -1)
        scale = self.gate(torch.cat([context, snr], dim=1))
        view = (value.shape[0], value.shape[1], *([1] * (value.ndim - 2)))
        return value * scale.view(view)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels: int, kernel_size: int, sigma: float):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.padding = kernel_size // 2
        self.groups = channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(value, self.weight, padding=self.padding, groups=self.groups)


def ss_warp(volume: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp a scale-space volume. ``flow`` is dx, dy in pixels and scale in [0, 1]."""
    batch, _, _, height, width = volume.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=volume.device, dtype=volume.dtype),
        torch.arange(width, device=volume.device, dtype=volume.dtype),
        indexing="ij",
    )
    x = 2.0 * (xx + flow[:, 0]) / max(width - 1, 1) - 1.0
    y = 2.0 * (yy + flow[:, 1]) / max(height - 1, 1) - 1.0
    z = 2.0 * flow[:, 2] - 1.0
    grid = torch.stack([x, y, z], dim=-1).unsqueeze(1)
    warped = F.grid_sample(volume, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return warped.squeeze(2)


class ScaleSpaceFlow(nn.Module):
    """Two-level flow tower. 108 pixels is not divisible by 16, so this is /4 not /16."""

    def __init__(self, width: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(6, width, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(width, width, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, 3, 4, stride=2, padding=1),
        )
        nn.init.normal_(self.up[-1].weight, std=0.01)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(pair))


class LatentDenoiser(nn.Module):
    """SNR-gated residual denoiser. Zero-init output so it starts as the identity."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.entry = nn.Conv1d(1, width, 3, padding=1)
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1),
        )
        self.af = AFModule(width)
        self.exit = nn.Conv1d(width, 1, 3, padding=1)
        nn.init.zeros_(self.exit.weight)
        nn.init.zeros_(self.exit.bias)

    def forward(self, code: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        hidden = self.entry(code.unsqueeze(1))
        hidden = hidden + self.block(hidden)
        hidden = self.af(hidden, snr_db)
        return code + self.exit(hidden).squeeze(1)


class FrameCodec(nn.Module):
    def __init__(self, c_in: int, c_out: int, latent: int, width: int):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, width, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(width, width, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.af = AFModule(width)
        self.reduce = nn.Conv2d(width, REDUCED, 1)
        self.to_wire = nn.Linear(FLAT, latent)
        self.from_wire = nn.Linear(latent, FLAT)
        self.expand = nn.Conv2d(REDUCED, width, 1)
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, c_out, 4, stride=2, padding=1),
        )
        nn.init.normal_(self.dec[-1].weight, std=0.01)
        nn.init.zeros_(self.dec[-1].bias)

    def encode(self, frame: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        feat = self.reduce(self.af(self.enc(frame), snr_db))
        if feat.shape[-2:] != (FEAT_H, FEAT_W):
            raise RuntimeError(f"encoder spatial shape {tuple(feat.shape[-2:])} != {(FEAT_H, FEAT_W)}")
        return self.to_wire(feat.flatten(1))

    def decode(self, code: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        feat = self.from_wire(code).view(code.shape[0], REDUCED, FEAT_H, FEAT_W)
        hidden = self.expand(feat)
        hidden = self.af(hidden, snr_db)
        return self.dec(hidden)


class RVDJSCCNarrowband(nn.Module):
    architecture = "rvdjscc-narrowband-v1"

    def __init__(self, bandwidth_khz: float = 2.2, width: int = 32, context_channels: int = 32):
        super().__init__()
        khz, budget, mode = resolve_bandwidth(bandwidth_khz)
        key_len, interp_len = packet_split(budget)
        self.bandwidth_khz = khz
        self.budget = budget
        self.modem_mode = mode
        self.key_len = key_len
        self.interp_len = interp_len
        self.width = width
        self.context_channels = context_channels
        self.nominal_snr_db = NOMINAL_SNR_DB
        self.clean_snr_db = CLEAN_SNR_DB

        self.key_codec = FrameCodec(3, 3, key_len, width)
        self.interp_codec = FrameCodec(3 + 2 * context_channels, 9, interp_len, width)
        self.denoiser = LatentDenoiser(width=64)
        self.ssf = ScaleSpaceFlow(width)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, context_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(context_channels, context_channels, 3, padding=1),
        )
        self.gaussians = nn.ModuleList(
            [GaussianSmoothing(context_channels, kernel_size=7, sigma=(2**level) * 0.5) for level in range(3)]
        )
        self.flow_refine = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.flow_refine[-1].weight)
        nn.init.zeros_(self.flow_refine[-1].bias)
        self.context_refine = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(context_channels, context_channels, 3, padding=1),
        )
        self.contextual = nn.Sequential(
            nn.Conv2d(3 + 2 * context_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.normal_(self.contextual[-1].weight, std=0.01)
        nn.init.zeros_(self.contextual[-1].bias)
        self.prev_key: torch.Tensor | None = None
        self.config = {
            "bandwidth_khz": khz,
            "budget": budget,
            "modem_mode": mode,
            "key_len": key_len,
            "interp_len": interp_len,
            "width": width,
            "context_channels": context_channels,
        }

    def reset(self) -> None:
        self.prev_key = None

    def _snr(self, batch: int, value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((batch, 1), value, device=device, dtype=dtype)

    def _reference(self, key_hat: torch.Tensor) -> torch.Tensor:
        prev = self.prev_key
        if prev is None or prev.shape[0] != key_hat.shape[0] or prev.shape[-2:] != key_hat.shape[-2:]:
            return key_hat
        return prev

    def scale_volume(self, feat: torch.Tensor) -> torch.Tensor:
        levels = [feat]
        for kernel in self.gaussians:
            levels.append(kernel(feat))
        return torch.stack(levels, dim=2)

    def context_from(self, flow: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        refined = flow + self.flow_refine(torch.cat([flow, reference], dim=1))
        volume = self.scale_volume(self.feature_extract(reference))
        warped = ss_warp(volume, refined)
        return self.context_refine(warped)

    def _denoise_frame(self, code: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        return self.denoiser(code, snr_db)

    def _decode_key(self, code: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.key_codec.decode(code, snr_db))

    def _decode_interp(
        self,
        code: torch.Tensor,
        ref_a: torch.Tensor,
        ref_b: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.interp_codec.decode(code, snr_db)
        detail, flow_a, flow_b = raw.split(3, dim=1)
        ctx_a = self.context_from(flow_a, ref_a)
        ctx_b = self.context_from(flow_b, ref_b)
        logits = self.contextual(torch.cat([detail, ctx_a, ctx_b], dim=1))
        return torch.sigmoid(logits)

    def _encode_interp(
        self,
        target: torch.Tensor,
        ref_a: torch.Tensor,
        ref_b: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        flow_a = self.ssf(torch.cat([target, ref_a], dim=1))
        flow_b = self.ssf(torch.cat([target, ref_b], dim=1))
        ctx_a = self.context_from(flow_a, ref_a)
        ctx_b = self.context_from(flow_b, ref_b)
        code = self.interp_codec.encode(torch.cat([target, ctx_a, ctx_b], dim=1), snr_db)
        return unit_rms(code)

    def encode_gop(self, video: torch.Tensor, retain_state: bool = True) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1:] != (3, GOP_FRAMES, HEIGHT, WIDTH):
            raise ValueError(
                f"expected (B, 3, {GOP_FRAMES}, {HEIGHT}, {WIDTH}), got {tuple(video.shape)}"
            )
        snr = self._snr(video.shape[0], self.nominal_snr_db, video.device, video.dtype)
        clean = self._snr(video.shape[0], self.clean_snr_db, video.device, video.dtype)
        frame0, frame1, frame2, key = (video[:, :, index] for index in range(GOP_FRAMES))
        key_code = unit_rms(self.key_codec.encode(key, snr))
        key_hat = self._decode_key(self._denoise_frame(key_code, clean), clean)
        prev = self._reference(key_hat)
        code1 = self._encode_interp(frame1, prev, key_hat, snr)
        hat1 = self._decode_interp(self._denoise_frame(code1, clean), prev, key_hat, clean)
        code0 = self._encode_interp(frame0, prev, hat1, snr)
        code2 = self._encode_interp(frame2, hat1, key_hat, snr)
        if retain_state:
            self.prev_key = key_hat.detach()
        return torch.cat([code0, code1, code2, key_code], dim=1)

    def _slices(self, wire: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if wire.shape[1] != self.budget:
            raise ValueError(f"wire length {wire.shape[1]} != budget {self.budget}")
        return tuple(wire.split([self.interp_len, self.interp_len, self.interp_len, self.key_len], dim=1))

    def _snr_from_confidence(self, confidence: torch.Tensor | None, batch: int, device, dtype) -> torch.Tensor:
        if confidence is None:
            return self._snr(batch, self.clean_snr_db, device, torch.float32)
        level = confidence[:, :1].float().clamp(1e-4, 1.0 - 1e-4)
        snr_lin = level / (1.0 - level)
        return (10.0 * torch.log10(snr_lin)).to(device=device)

    def decode_gop(
        self,
        wire: torch.Tensor,
        confidence: torch.Tensor | None = None,
        retain_state: bool = True,
        *,
        snr_db: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code0, code1, code2, key_code = self._slices(wire)
        if snr_db is None:
            snr_db = self._snr_from_confidence(confidence, wire.shape[0], wire.device, wire.dtype)
        elif snr_db.ndim == 0:
            snr_db = self._snr(wire.shape[0], float(snr_db), wire.device, torch.float32)
        d0 = self._denoise_frame(code0, snr_db)
        d1 = self._denoise_frame(code1, snr_db)
        d2 = self._denoise_frame(code2, snr_db)
        dkey = self._denoise_frame(key_code, snr_db)
        key_hat = self._decode_key(dkey, snr_db)
        prev = self._reference(key_hat)
        hat1 = self._decode_interp(d1, prev, key_hat, snr_db)
        hat0 = self._decode_interp(d0, prev, hat1, snr_db)
        hat2 = self._decode_interp(d2, hat1, key_hat, snr_db)
        recon = torch.stack([hat0, hat1, hat2, key_hat], dim=2)
        if retain_state:
            self.prev_key = key_hat.detach()
        outage = wire.new_zeros(wire.shape[0], GOP_FRAMES)
        return recon, outage

    def denoising_loss(
        self,
        clean_wire: torch.Tensor,
        received_wire: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        """Paper loss term: MSE between the sent code and the denoised received code."""
        losses = [
            F.mse_loss(self._denoise_frame(received, snr_db), clean)
            for clean, received in zip(self._slices(clean_wire), self._slices(received_wire), strict=True)
        ]
        return torch.stack(losses).mean()


def load_rvdjscc(path: str, device: torch.device | str = "cpu") -> tuple[RVDJSCCNarrowband, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("model_config") or payload.get("config") or {}
    model = RVDJSCCNarrowband(
        bandwidth_khz=payload.get("bandwidth_khz", config.get("bandwidth_khz", 2.2)),
        width=config.get("width", 32),
        context_channels=config.get("context_channels", 32),
    )
    state = {
        key: value.float() if torch.is_floating_point(value) else value
        for key, value in payload["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.to(device).eval()
    return model, payload
