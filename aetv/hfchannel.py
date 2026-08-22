"""NumPy HF channel simulator: AWGN, Watterson fading, frequency offset.

SNR is signal power relative to the noise power falling in
``SNR_REF_BW_HZ`` (2500 Hz), matching the AETV modem's pilot estimator.
Pass ``fs`` from the mode geometry — Flex-8k (band U) is 24 kHz, not 8 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import FS, SNR_REF_BW_HZ


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
    sigma2 = s_power * (fs / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10)
    return x + rng.normal(scale=np.sqrt(sigma2), size=len(x))
