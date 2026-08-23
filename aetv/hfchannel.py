"""NumPy HF channel simulator: AWGN, Watterson fading, frequency offset.

SNR is signal power relative to the noise power falling in
``SNR_REF_BW_HZ`` (2500 Hz), matching the AETV modem's pilot estimator.
Pass ``fs`` from the mode geometry — Flex-8k (band U) is 24 kHz, not 8 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import FS, reference_noise_bandwidth_scale


@dataclass(frozen=True)
class FadingPreset:
    name: str
    doppler_hz: float
    delay_ms: float


FADING_PRESETS = {
    "mpg": FadingPreset("mpg", 0.1, 0.5),
    "mpp": FadingPreset("mpp", 1.0, 2.0),
    "mpd": FadingPreset("mpd", 2.0, 4.0),
}


@dataclass(frozen=True)
class ChannelProfile:
    """Named, repeatable operator profile for local modem loopback tests."""

    key: str
    label: str
    snr_db: float | None = None
    fading: str = "none"
    description: str = ""


CHANNEL_PROFILES = {
    "clean": ChannelProfile("clean", "Clean", description="No channel impairment"),
    "awgn12": ChannelProfile("awgn12", "12 dB", 12.0, description="AWGN at 12 dB SNR"),
    "awgn6": ChannelProfile("awgn6", "6 dB", 6.0, description="AWGN at 6 dB SNR"),
    "awgn0": ChannelProfile("awgn0", "0 dB", 0.0, description="AWGN at 0 dB SNR"),
    "mpp12": ChannelProfile("mpp12", "MPP 12", 12.0, "mpp", "MPP fading at 12 dB SNR"),
    "mpp6": ChannelProfile("mpp6", "MPP 6", 6.0, "mpp", "MPP fading at 6 dB SNR"),
    "mpp0": ChannelProfile("mpp0", "MPP 0", 0.0, "mpp", "MPP fading at 0 dB SNR"),
}


def _analytic(x: np.ndarray) -> np.ndarray:
    return signal.hilbert(x)


def wrap_cycles(cycles: np.ndarray) -> np.ndarray:
    return np.mod(cycles, 1.0)


def freq_shift(x: np.ndarray, df_hz: float, fs: int = FS) -> np.ndarray:
    n = np.arange(len(x))
    return np.real(_analytic(x) * np.exp(2j * np.pi * wrap_cycles(df_hz * n / fs)))


def _rayleigh_taps(
    n: int, doppler_hz: float, rng: np.random.Generator, fs: int
) -> np.ndarray:
    lowrate = max(8 * doppler_hz, 1.0)
    n_low = int(np.ceil(n * lowrate / fs)) + 8
    g = rng.normal(size=n_low) + 1j * rng.normal(size=n_low)
    b, a = signal.butter(2, min(doppler_hz / (lowrate / 2), 0.99))
    g = signal.lfilter(b, a, g)
    g = g[4:]
    t_low = np.arange(len(g)) * (fs / lowrate)
    t = np.arange(n)
    tap = np.interp(t, t_low, g.real) + 1j * np.interp(t, t_low, g.imag)
    return tap / np.sqrt(np.mean(np.abs(tap) ** 2))


def fading(
    x: np.ndarray,
    preset: str | FadingPreset,
    seed: int = 0,
    fs: int = FS,
) -> np.ndarray:
    """Two independent equal-power Rayleigh paths (Watterson model)."""
    p = FADING_PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    z = _analytic(x)
    delay = int(round(p.delay_ms * 1e-3 * fs))
    g1 = _rayleigh_taps(len(z), p.doppler_hz, rng, fs)
    g2 = _rayleigh_taps(len(z), p.doppler_hz, rng, fs)
    z2 = np.concatenate([np.zeros(delay, dtype=complex), z[: len(z) - delay]])
    return np.real((z * g1 + z2 * g2) / np.sqrt(2))


def awgn(x: np.ndarray, snr_db: float, seed: int = 0, fs: int = FS) -> np.ndarray:
    """Add white noise for the given SNR in a ``SNR_REF_BW_HZ`` bandwidth."""
    rng = np.random.default_rng(seed)
    env = np.abs(_analytic(x))
    active = env > 0.1 * np.sqrt(np.mean(x**2))
    s_power = np.mean(x[active] ** 2) if active.any() else np.mean(x**2)
    sigma2 = (
        s_power * reference_noise_bandwidth_scale(fs) / 10 ** (snr_db / 10)
    )
    return x + rng.normal(scale=np.sqrt(sigma2), size=len(x))


def emulate(
    x: np.ndarray,
    profile: str | ChannelProfile,
    seed: int = 42,
    fs: int = FS,
) -> np.ndarray:
    """Apply a named local-test profile to one complete transmit waveform.

    Applying the profile to the complete waveform keeps the Watterson taps
    continuous across GOP boundaries. The fixed default seed makes visual
    comparisons and bug reproduction deterministic.
    """
    selected = CHANNEL_PROFILES[profile] if isinstance(profile, str) else profile
    impaired = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if selected.fading != "none":
        impaired = fading(impaired, preset=selected.fading, seed=seed, fs=fs)
    if selected.snr_db is not None:
        impaired = awgn(impaired, snr_db=selected.snr_db, seed=seed, fs=fs)
    return impaired.astype(np.float32)


class StreamingChannelEmulator:
    """Stateful channel used by the GUI's incremental modem loopback.

    Watterson path gains use a deterministic sum-of-sinusoids Rayleigh model,
    which keeps Doppler phase and delayed-path history continuous across GOP
    chunks. Noise generation is likewise continuous for repeatable debugging.
    """

    def __init__(
        self,
        profile: str | ChannelProfile,
        seed: int = 42,
        fs: int = FS,
    ):
        self.profile = CHANNEL_PROFILES[profile] if isinstance(profile, str) else profile
        self.fs = int(fs)
        self._sample = 0
        self._noise_rng = np.random.default_rng(seed + 1)
        self._delay_line = np.zeros(0, dtype=np.complex128)
        self._doppler_freqs = np.zeros(0)
        self._phases1 = np.zeros(0)
        self._phases2 = np.zeros(0)
        if self.profile.fading != "none":
            preset = FADING_PRESETS[self.profile.fading]
            oscillators = 24
            angles = np.pi * (np.arange(oscillators) + 0.5) / oscillators
            self._doppler_freqs = preset.doppler_hz * np.cos(angles)
            phase_rng = np.random.default_rng(seed)
            self._phases1 = phase_rng.uniform(0.0, 2.0 * np.pi, oscillators)
            self._phases2 = phase_rng.uniform(0.0, 2.0 * np.pi, oscillators)
            delay = int(round(preset.delay_ms * 1e-3 * self.fs))
            self._delay_line = np.zeros(delay, dtype=np.complex128)

    def _tap(self, samples: np.ndarray, phases: np.ndarray) -> np.ndarray:
        tap = np.zeros(len(samples), dtype=np.complex128)
        time_s = samples / self.fs
        for frequency, phase in zip(self._doppler_freqs, phases):
            tap += np.exp(1j * (2.0 * np.pi * frequency * time_s + phase))
        return tap / np.sqrt(max(1, len(phases)))

    def process(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return values.astype(np.float32)
        impaired = values.copy()
        if self.profile.fading != "none":
            analytic = _analytic(values)
            samples = self._sample + np.arange(len(values), dtype=np.float64)
            first = self._tap(samples, self._phases1)
            second = self._tap(samples, self._phases2)
            delay = len(self._delay_line)
            if delay:
                delayed = np.concatenate([self._delay_line, analytic])[: len(analytic)]
                self._delay_line = np.concatenate([self._delay_line, analytic])[-delay:]
            else:
                delayed = analytic
            impaired = np.real((analytic * first + delayed * second) / np.sqrt(2.0))
        if self.profile.snr_db is not None:
            envelope = np.abs(_analytic(impaired))
            threshold = 0.1 * np.sqrt(np.mean(impaired**2))
            active = envelope > threshold
            power = np.mean(impaired[active] ** 2) if active.any() else np.mean(impaired**2)
            variance = (
                power * reference_noise_bandwidth_scale(self.fs)
                / 10 ** (self.profile.snr_db / 10)
            )
            impaired = impaired + self._noise_rng.normal(
                scale=np.sqrt(variance), size=len(impaired)
            )
        self._sample += len(values)
        return impaired.astype(np.float32)
