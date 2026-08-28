"""Differentiable sequence channel curriculum for recurrent GOP training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiGOPChannelCurriculum(nn.Module):
    """Correlated clean/AWGN/fade/missing patterns over complete GOP streams.

    This is a training surrogate, not promotion evidence.  Promotion uses the
    stateful continuous waveform runtime.  All corruptions preserve the V8
    tensor/wire geometry and retain gradients to every non-missing TX value.
    """

    def __init__(
        self,
        *,
        missing_probability: float = 0.08,
        fade_probability: float = 0.65,
    ):
        super().__init__()
        self.missing_probability = missing_probability
        self.fade_probability = fade_probability

    def forward(
        self, values: torch.Tensor, *, progress: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if values.ndim != 3 or values.shape[-1] != 2816:
            raise ValueError("multi-GOP channel expects B,G,2816")
        batch, count, length = values.shape
        device, dtype = values.device, values.dtype
        progress = float(min(max(progress, 0.0), 1.0))

        # Clean-heavy warmup transitions into the full 6 dB / fading mix.
        hard = max(0.0, (progress - 0.15) / 0.85)
        base_snr = 18.0 - 12.0 * hard
        snr = base_snr + 3.0 * (2.0 * torch.rand(batch, count, device=device) - 1.0)
        gain = torch.ones(batch, count, device=device, dtype=dtype)
        fade = torch.zeros(batch, count, dtype=torch.bool, device=device)

        # Each selected stream contains a contiguous good -> fade -> good
        # episode.  Fade depth and smooth spectral notches vary by example.
        if count >= 5 and hard > 0:
            for item in range(batch):
                if torch.rand((), device=device) < self.fade_probability * hard:
                    start = int(torch.randint(1, max(2, count - 2), (), device=device))
                    maximum = max(1, min(count - start - 1, max(2, count // 3)))
                    duration = int(torch.randint(1, maximum + 1, (), device=device))
                    stop = min(count - 1, start + duration)
                    fade[item, start:stop] = True
                    gain[item, start:stop] = 0.18 + 0.42 * torch.rand((), device=device)
                    snr[item, start:stop] = -1.0 + 8.0 * torch.rand((), device=device)

        coordinate = torch.linspace(0, 2 * math.pi, length, device=device, dtype=dtype)
        phase = 2 * math.pi * torch.rand(batch, count, 1, device=device, dtype=dtype)
        notch = 0.72 + 0.28 * torch.cos(3.0 * coordinate.view(1, 1, -1) + phase)
        spectral_gain = torch.where(fade[..., None], notch, torch.ones_like(notch))
        effective_gain = gain[..., None] * spectral_gain
        sigma = torch.pow(values.new_tensor(10.0), -snr[..., None] / 20.0)
        noise = torch.randn_like(values) * sigma
        received = values * effective_gain + noise
        confidence = effective_gain.square() / (
            effective_gain.square() + sigma.square().clamp_min(1e-8)
        )

        missing = torch.zeros(batch, count, dtype=torch.bool, device=device)
        if hard > 0:
            draw = torch.rand(batch, count, device=device)
            missing = draw < (self.missing_probability * hard)
            # Preserve the first and last good reference GOP and avoid two
            # adjacent erasures in this curriculum.
            missing[:, 0] = False
            missing[:, -1] = False
            if count > 1:
                missing[:, 1:] &= ~missing[:, :-1]
        received = received.masked_fill(missing[..., None], 0.0)
        confidence = confidence.masked_fill(missing[..., None], 0.0)

        fade_end = torch.zeros_like(fade)
        if count > 1:
            fade_end[:, 1:] = fade[:, :-1] & ~fade[:, 1:]
        return received, confidence, {
            "missing": missing,
            "fade": fade,
            "fade_end": fade_end,
            "snr_db": snr,
            "gain": gain,
        }

