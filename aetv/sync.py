"""Acquisition and synchronization for AETV continuous streams.

Supports fast preamble acquisition and blind pilot-based acquisition for join-in-progress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import (
    ACQUIRE_MAX_BINS,
    BANDS,
    FS,
    M,
    NCP,
    PREAMBLE_CORR_WINDOW,
    PREAMBLE_CP,
    PREAMBLE_REPEATS,
    PREAMBLE_SAMPLES,
    PREAMBLE_THRESHOLD,
    RS,
    TEMPLATE_SCORE_THRESHOLD,
)
from .ofdm import preamble_template


def freq_correct(z: np.ndarray, f_hz: float, fs: int = FS) -> np.ndarray:
    n = np.arange(len(z))
    cycles = (f_hz * n / fs) % 1.0
    return z * np.exp(-2j * np.pi * cycles)


def sync_lowpass(z: np.ndarray, cutoff_hz: float = 1200.0, fs: int = FS) -> np.ndarray:
    taps = signal.firwin(65, min(cutoff_hz, 0.45 * fs), fs=fs)
    return signal.convolve(z, taps, mode="same")



class SyncError(Exception):
    pass


@dataclass
class Acquisition:
    preamble_start: int  # index of first preamble sample
    freq_offset: float  # Hz
    metric: float  # correlation confidence 0..1


def _autocorr_metric(z: np.ndarray, m: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding lag-M autocorrelation over window W."""
    if len(z) < m + w:
        return np.array([]), np.array([])
    prod = z[m:] * np.conj(z[:-m])
    power = np.abs(z) ** 2
    kernel = np.ones(w)
    a = signal.fftconvolve(prod, kernel, mode="valid")
    e1 = signal.fftconvolve(power[:-m], kernel, mode="valid")
    e2 = signal.fftconvolve(power[m:], kernel, mode="valid")
    floor = 1e-3 * w * (np.mean(power) + 1e-12)
    energy = np.sqrt(np.maximum(e1, floor) * np.maximum(e2, floor)) + 1e-12
    return np.abs(a) / energy, a


def acquire(
    z: np.ndarray,
    band: str = "W",
    threshold: float = PREAMBLE_THRESHOLD,
    max_bins: int = ACQUIRE_MAX_BINS,
    search: tuple[int, int] | None = None,
) -> Acquisition:
    """Find preamble in analytic baseband signal z."""
    geom = BANDS[band]
    fs = geom.fs
    m = fs // RS
    w = (PREAMBLE_REPEATS - 1) * m
    ncp = m // 4
    preamble_cp = 2 * ncp
    preamble_samples = preamble_cp + PREAMBLE_REPEATS * m

    z_filt = sync_lowpass(z, cutoff_hz=min(0.45 * fs, (geom.carriers * RS) / 2 + 500), fs=fs)
    metric, a = _autocorr_metric(z_filt, m=m, w=w)

    if len(metric) == 0:
        raise SyncError("signal too short for acquisition")

    if search is not None:
        s0, s1 = search
        metric_slice = metric[s0:s1]
        if len(metric_slice) == 0:
            raise SyncError("search window out of range")
        peak_rel = int(np.argmax(metric_slice))
        peak_idx = s0 + peak_rel
    else:
        peak_idx = int(np.argmax(metric))

    peak_val = float(metric[peak_idx])
    if peak_val < threshold:
        raise SyncError(f"no preamble detected (peak metric {peak_val:.3f} < {threshold:.3f})")

    # Fractional CFO from angle of lag-M autocorrelation
    frac_cfo = float(np.angle(a[peak_idx]) / (2 * np.pi * (m / fs)))

    # Integer bin search using preamble matched filter
    tmpl = preamble_template(band)
    best_bin = 0
    best_score = -1.0
    best_offset = 0

    # Test candidate integer carrier offsets
    candidate_bins = range(-max_bins, max_bins + 1)
    for k in candidate_bins:
        cfo_cand = frac_cfo + k * RS
        z_cand = freq_correct(z_filt, cfo_cand, fs=fs)
        
        # Cross-correlate around peak_idx
        win_start = max(0, peak_idx - m)
        win_end = min(len(z_cand), peak_idx + preamble_samples + m)
        if win_end - win_start < len(tmpl):
            continue
        corr = signal.correlate(z_cand[win_start:win_end], tmpl, mode="valid")
        if len(corr) == 0:
            continue
        corr_peak = np.max(np.abs(corr))
        tmpl_energy = np.sum(np.abs(tmpl) ** 2)
        norm_score = corr_peak / (tmpl_energy + 1e-12)
        if norm_score > best_score:
            best_score = norm_score
            best_bin = k
            best_offset = win_start + int(np.argmax(np.abs(corr)))

    total_cfo = frac_cfo + best_bin * RS
    return Acquisition(
        preamble_start=best_offset,
        freq_offset=total_cfo,
        metric=peak_val,
    )
