"""Differentiable PyTorch simulation and training channels for AETV.

Provides:
1. AETVLatentChannel (Stage 1): fast differentiable latent corruption with AWGN,
   progressive 3-group truncation, and time-correlated erasure bursts.
2. AETVWaveformChannel (Stage 2): end-to-end differentiable OFDM channel modeling
   exact 8 kHz AETV numerology, Watterson fading across the 1.0s GOP, envelope
   clipping at 0.5 dB headroom, and confidence-weighted demodulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

from .config import (
    BANDS,
    CLIP_HEADROOM_DB,
    DATA_SYMS_PER_FRAME,
    DEMOD_BACKOFF,
    FRAMES_PER_GOP,
    FS,
    M,
    NCP,
    NSYM,
    RS,
    SYMS_PER_FRAME,
    reference_noise_bandwidth_scale,
)
from .framing import GOP_INTERLEAVER_N, GOP_INTERLEAVER_W, GOP_INTERLEAVER_U
from .ofdm import pilot_sequence


@dataclass
class AETVChannelConfig:
    snr_db_range: tuple[float, float] = (-2.0, 10.0)  # low-SNR focused regime
    snr_focus_range: tuple[float, float] | None = None
    p_snr_focus: float = 0.0
    p_fading: float = 0.70  # 70% probability of Watterson multipath fading
    doppler_range_hz: tuple[float, float] = (0.1, 2.0)
    delay_range_ms: tuple[float, float] = (0.5, 4.0)
    clip_headroom_db: float = CLIP_HEADROOM_DB
    p_truncate: float = 0.0
    erasure_rate_max: float = 0.0


def _sample_snr_db(
    cfg: AETVChannelConfig,
    batch: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample the broad channel range with an optional OTA-focused mixture."""
    lo, hi = cfg.snr_db_range
    broad = lo + torch.rand(batch, 1, device=device, generator=generator) * (hi - lo)
    if cfg.snr_focus_range is None or cfg.p_snr_focus <= 0.0:
        return broad
    focus_lo, focus_hi = cfg.snr_focus_range
    focused = focus_lo + torch.rand(batch, 1, device=device, generator=generator) * (
        focus_hi - focus_lo
    )
    choose_focus = (
        torch.rand(batch, 1, device=device, generator=generator) < cfg.p_snr_focus
    )
    return torch.where(choose_focus, focused, broad)


class AETVLatentChannel(nn.Module):
    """Stage 1 fast differentiable latent corruption channel."""

    def __init__(self, cfg: AETVChannelConfig | None = None):
        super().__init__()
        self.cfg = cfg or AETVChannelConfig()

    def forward(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, N_LATENTS) -> (noisy_z, weights)"""
        b, n_latents = z.shape
        device = z.device
        weights = torch.ones_like(z)

        # Progressive 3-group truncation
        if self.cfg.p_truncate > 0:
            p = torch.rand(b, 1, device=device, generator=generator)
            g1_mask = p < (self.cfg.p_truncate * 0.33)
            g2_mask = (p >= (self.cfg.p_truncate * 0.33)) & (p < self.cfg.p_truncate)

            g1_cutoff = int(n_latents * (1.0 / 3.0))
            g2_cutoff = int(n_latents * (2.0 / 3.0))

            mask = torch.ones_like(z)
            mask[g1_mask.squeeze(1), g1_cutoff:] = 0.0
            mask[g2_mask.squeeze(1), g2_cutoff:] = 0.0
            weights = weights * mask

        # Random erasure bursts
        if self.cfg.erasure_rate_max > 0:
            rate = torch.rand(b, 1, device=device, generator=generator) * self.cfg.erasure_rate_max
            erasure_mask = torch.rand(b, n_latents, device=device, generator=generator) >= rate
            weights = weights * erasure_mask.float()

        # Additive white Gaussian noise (AWGN)
        snr_db = _sample_snr_db(self.cfg, b, device, generator)
        sigma = 10.0 ** (-snr_db / 20.0)
        noise = torch.randn(z.shape, device=device, generator=generator) * sigma

        noisy_z = (z + noise) * weights
        return noisy_z, weights


class AETVWaveformChannel(nn.Module):
    """Stage 2 end-to-end differentiable AETV OFDM simulation with symbol-domain Watterson fading."""

    def __init__(
        self,
        band: str = "W",
        cfg: AETVChannelConfig | None = None,
        interleave: bool = True,
    ):
        super().__init__()
        self.band = band
        self.geometry = BANDS[band]
        self.cfg = cfg or AETVChannelConfig()
        self.interleave = interleave

        g = self.geometry
        self.FS = g.fs
        self.M = self.FS // RS
        self.NCP = self.M // 4
        self.NSYM = self.M + self.NCP

        self.N_FRAMES = FRAMES_PER_GOP  # 8
        self.N_SYMS = self.N_FRAMES * SYMS_PER_FRAME  # 40 symbols / GOP
        self.N_LATENTS = self.geometry.latents_per_gop

        # Precompute modulation & demodulation matrices
        carrier_freqs = g.carrier0_hz + RS * np.arange(g.carriers)
        n_symbol = np.arange(self.NSYM) - self.NCP
        mod = np.exp(2j * np.pi * (np.outer(n_symbol, carrier_freqs) % self.FS) / self.FS)

        n_useful = np.arange(self.M)
        baseband_freqs = carrier_freqs - g.fcenter_hz
        demod = np.exp(-2j * np.pi * (np.outer(baseband_freqs, n_useful) % self.FS) / self.FS)

        self.register_buffer("mod_mat", torch.from_numpy(mod).to(torch.complex64))
        self.register_buffer("demod_mat", torch.from_numpy(demod).to(torch.complex64))
        self.register_buffer("pilot", torch.from_numpy(pilot_sequence(band)).to(torch.complex64))
        self.register_buffer("carrier_freqs", torch.from_numpy(carrier_freqs.astype(np.float32)))

        # Interleaver permutations
        if band == "N":
            perm = GOP_INTERLEAVER_N
        elif band == "W":
            perm = GOP_INTERLEAVER_W
        else:
            perm = GOP_INTERLEAVER_U
        self.register_buffer("tx_perm", torch.from_numpy(perm).long())
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        self.register_buffer("rx_perm", torch.from_numpy(inv_perm).long())

        # Bandpass filter taps
        taps = firwin(129, g.tx_bandpass, fs=self.FS, pass_zero=False)
        self.register_buffer("bp_taps", torch.from_numpy(taps).float()[None, None, :])

    def _to_symbols(self, latents: torch.Tensor) -> torch.Tensor:
        g = self.geometry
        b = latents.shape[0]

        if self.interleave:
            tx_lat = latents[:, self.tx_perm]
        else:
            tx_lat = latents

        pairs = tx_lat.view(b, self.N_FRAMES, DATA_SYMS_PER_FRAME, g.latent_carriers, 2)
        data = torch.complex(pairs[..., 0], pairs[..., 1]).to(torch.complex64)

        symbols = torch.empty(
            b, self.N_FRAMES, SYMS_PER_FRAME, g.carriers, dtype=torch.complex64, device=latents.device
        )
        symbols[:, :, 0] = self.pilot
        symbols[:, :, 1:, : g.latent_carriers] = data

        # Beacon chips
        beacon_bits = (
            torch.randint(0, 2, (b, self.N_FRAMES, DATA_SYMS_PER_FRAME), device=latents.device) * 2 - 1
        ).to(torch.complex64)
        symbols[:, :, 1:, g.beacon_carrier] = beacon_bits
        return symbols.view(b, self.N_SYMS, g.carriers)

    def _smooth_gains(self, b: int, doppler: torch.Tensor, device: torch.device) -> torch.Tensor:
        """(b, N_SYMS) complex tap gains, ~Gaussian Doppler spectrum."""
        sym_rate = self.FS / self.NSYM
        g = torch.complex(
            torch.randn(b, self.N_SYMS, device=device),
            torch.randn(b, self.N_SYMS, device=device),
        )
        sigma_syms = (sym_rate / (2 * np.pi * doppler)).clamp(1.0, max(2.0, self.N_SYMS / 4))
        out = torch.empty_like(g)
        half = min(int(3 * sigma_syms.max().item()), self.N_SYMS - 1)
        half = max(1, half)
        t = torch.arange(-half, half + 1, device=device).float()
        for i in range(b):
            k = torch.exp(-0.5 * (t / sigma_syms[i]) ** 2)
            k = (k / k.sum())[None, None, :]
            gr = F.conv1d(g[i].real[None, None, :], k, padding=half)[0, 0, : self.N_SYMS]
            gi = F.conv1d(g[i].imag[None, None, :], k, padding=half)[0, 0, : self.N_SYMS]
            out[i] = torch.complex(gr, gi)
        return out / out.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-9)

    def _fading(self, b: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
        """(b, N_SYMS, NC) complex channel transfer matrix with 2-ray Watterson frequency-selective fading."""
        cfg = self.cfg
        lo, hi = cfg.doppler_range_hz
        doppler = lo + torch.rand(b, device=device, generator=generator) * (hi - lo)
        g1 = self._smooth_gains(b, doppler, device)
        g2 = self._smooth_gains(b, doppler, device)
        dlo, dhi = cfg.delay_range_ms
        tau_s = (dlo + torch.rand(b, device=device, generator=generator) * (dhi - dlo)) * 1e-3
        phase = -2 * np.pi * tau_s[:, None] * self.carrier_freqs[None, :]
        rot = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
        h = (g1[:, :, None] + g2[:, :, None] * rot[:, None, :]) / np.sqrt(2)
        flat = torch.rand(b, device=device, generator=generator) >= cfg.p_fading
        h[flat] = 1.0 + 0j
        return h

    def _synthesize(self, symbols: torch.Tensor) -> torch.Tensor:
        samples = torch.einsum("bsc,nc->bsn", symbols, self.mod_mat).real
        return samples.flatten(1)

    @staticmethod
    def _analytic(samples: torch.Tensor) -> torch.Tensor:
        n = samples.shape[-1]
        spectrum = torch.fft.fft(samples.float(), dim=-1)
        mask = torch.zeros(n, device=samples.device)
        mask[0] = 1
        mask[1 : (n + 1) // 2] = 2
        if n % 2 == 0:
            mask[n // 2] = 1
        return torch.fft.ifft(spectrum * mask, dim=-1)

    def _clip_filter(self, samples: torch.Tensor) -> torch.Tensor:
        g = self.geometry
        threshold = (
            torch.sqrt(2 * samples.square().mean(dim=1, keepdim=True))
            * 10 ** (self.cfg.clip_headroom_db / 20)
        )
        padding = (len(self.bp_taps[0, 0]) - 1) // 2
        for _ in range(2):
            analytic = self._analytic(samples)
            gain = torch.clamp(threshold / (analytic.abs().clamp_min(1e-9)), max=1.0)
            samples = (analytic * gain).real
            samples = F.conv1d(samples[:, None], self.bp_taps, padding=padding)[:, 0]
        return samples

    def _to_baseband(self, samples: torch.Tensor) -> torch.Tensor:
        b, n = samples.shape
        t = torch.arange(n, device=samples.device, dtype=torch.float32)
        het = torch.exp(-2j * np.pi * self.geometry.fcenter_hz * t / self.FS)
        return samples * het

    def _demodulate(self, samples: torch.Tensor) -> torch.Tensor:
        g = self.geometry
        b = samples.shape[0]
        z = self._to_baseband(samples)
        windows = z.view(b, self.N_SYMS, self.NSYM)
        start = self.NCP
        useful = windows[:, :, start : start + self.M].to(torch.complex64)
        return (2.0 / self.M) * torch.einsum("bsn,cn->bsc", useful, self.demod_mat)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        latents: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """latents: (B, N_LATENTS) -> (noisy_latents, weights)"""
        latents = latents.float()
        b = latents.shape[0]
        device = latents.device
        g = self.geometry

        # Modulate to passband symbols
        symbols = self._to_symbols(latents)


        # Apply symbol-domain 2-ray Watterson frequency-selective fading
        h_channel = self._fading(b, device, generator=generator)
        faded_symbols = symbols * h_channel

        # Synthesize audio waveform and apply clip-and-filter
        samples = self._synthesize(faded_symbols)
        tx_audio = self._clip_filter(samples)

        # Channel simulation: AWGN over reference bandwidth in target SNR range
        snr_db = _sample_snr_db(self.cfg, b, device, generator)
        snr_linear = 10.0 ** (snr_db / 10.0)
        signal_pwr = tx_audio.square().mean(dim=1, keepdim=True)
        # ``snr_db`` is referenced to noise in SNR_REF_BW_HZ. The generated
        # real white noise spans the full Nyquist bandwidth, so expand its
        # total variance by (FS/2)/reference_bandwidth. The old expression
        # divided by occupied_signal_bandwidth/reference_bandwidth instead,
        # making labelled conditions 11.86 dB too optimistic in V7/U.
        noise_std = torch.sqrt(
            signal_pwr
            * reference_noise_bandwidth_scale(self.FS)
            / snr_linear.clamp_min(1e-12)
        )
        rx_audio = tx_audio + torch.randn_like(tx_audio) * noise_std

        # Demodulate
        rx_syms = self._demodulate(rx_audio)  # (B, N_SYMS, NC)
        rx_frames = rx_syms.view(b, self.N_FRAMES, SYMS_PER_FRAME, g.carriers)

        # Equalization from pilots (sym 0 in each frame)
        pilot_rx = rx_frames[:, :, 0, :]  # (B, N_FRAMES, NC)
        h_est = pilot_rx / (self.pilot[None, None, :] + 1e-9)

        # Equalize data symbols (syms 1..4). A complete GOP is available to
        # both training and production decode, so interpolate toward the next
        # frame pilot instead of using a channel estimate up to 100 ms old.
        data_syms_rx = rx_frames[:, :, 1:, : g.latent_carriers]  # (B, N_FRAMES, 4, NC_LATENT)
        h_current = h_est[:, :, : g.latent_carriers]
        h_next = torch.cat([h_current[:, 1:], h_current[:, -1:]], dim=1)
        fractions = (
            torch.arange(1, DATA_SYMS_PER_FRAME + 1, device=device).float()
            / SYMS_PER_FRAME
        )[None, None, :, None]
        h_latent = (
            (1.0 - fractions) * h_current[:, :, None, :]
            + fractions * h_next[:, :, None, :]
        )

        differences = h_current[:, 1:] - h_current[:, :-1]
        noise_variance = 0.5 * differences.abs().square().mean(dim=(1, 2))
        total_power = h_current.abs().square().mean(dim=(1, 2))
        noise_variance = torch.maximum(
            noise_variance, total_power.clamp_min(1e-18) * 1e-9
        )[:, None, None, None]
        channel_power = h_latent.abs().square()
        denom = channel_power.clamp_min(noise_variance * 1e-6)
        eq_data = data_syms_rx * torch.conj(h_latent) / denom
        eq_weights = torch.clamp(
            channel_power / (channel_power + noise_variance), 0.0, 1.0
        )

        # Unpack I and Q
        flat_eq = eq_data.reshape(b, -1)
        flat_w = eq_weights.reshape(b, -1)

        raw_latents = torch.empty(b, self.N_LATENTS, dtype=latents.dtype, device=device)
        raw_weights = torch.empty(b, self.N_LATENTS, dtype=latents.dtype, device=device)

        raw_latents[:, 0::2] = flat_eq.real
        raw_latents[:, 1::2] = flat_eq.imag

        raw_weights[:, 0::2] = flat_w.real
        raw_weights[:, 1::2] = flat_w.real

        # De-interleave
        if self.interleave:
            out_latents = raw_latents[:, self.rx_perm]
            out_weights = raw_weights[:, self.rx_perm]
        else:
            out_latents = raw_latents
            out_weights = raw_weights

        return out_latents, out_weights
