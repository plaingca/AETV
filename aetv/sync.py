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

    # Fractional CFO from angle of lag-M autocorrelation
    frac_cfo = float(np.angle(a[peak_idx]) / (2 * np.pi * (m / fs)))

    # Search the full candidate interval with a normalized preamble matched
    # filter. At severe SNR the repeated-symbol autocorrelation still gives a
    # useful fractional-CFO estimate, but its largest timing peak is often a
    # noise sample. The full known preamble has much greater processing gain.
    tmpl = preamble_template(band)
    best_bin = 0
    best_score = -1.0
    best_offset = 0
    best_noncoherent = -1.0
    best_noncoherent_bin = 0
    best_noncoherent_offset = 0
    template_energy = float(np.sum(np.abs(tmpl) ** 2))
    window_energy = signal.fftconvolve(
        np.abs(z_filt) ** 2, np.ones(len(tmpl)), mode="valid"
    )
    energy_floor = (
        1e-3 * len(tmpl) * (float(np.mean(np.abs(z_filt) ** 2)) + 1e-12)
    )
    single_template = tmpl[preamble_cp : preamble_cp + m]
    single_energy = float(np.sum(np.abs(single_template) ** 2))
    single_window_energy = signal.fftconvolve(
        np.abs(z_filt) ** 2, np.ones(m), mode="valid"
    )
    single_floor = 1e-3 * m * (float(np.mean(np.abs(z_filt) ** 2)) + 1e-12)
    candidate_count = max(0, len(z_filt) - preamble_samples + 1)

    candidate_bins = range(-max_bins, max_bins + 1)
    for k in candidate_bins:
        cfo_cand = frac_cfo + k * RS
        z_cand = freq_correct(z_filt, cfo_cand, fs=fs)
        corr = signal.fftconvolve(z_cand, np.conj(tmpl[::-1]), mode="valid")
        scores = np.abs(corr) / np.sqrt(
            np.maximum(window_energy, energy_floor) * template_energy
        )
        if search is None:
            score_start, score_end = 0, len(scores)
        else:
            score_start = max(0, int(search[0]))
            score_end = min(len(scores), int(search[1]))
        if score_end <= score_start:
            continue
        relative = int(np.argmax(scores[score_start:score_end]))
        offset = score_start + relative
        norm_score = float(scores[offset])
        if norm_score > best_score:
            best_score = norm_score
            best_bin = k
            best_offset = offset

        # Noncoherently combine the repeated known symbol. This retains
        # processing gain through MPP fades whose phase rotates enough to
        # cancel a single coherent correlation over the whole preamble.
        single_corr = signal.fftconvolve(
            z_cand, np.conj(single_template[::-1]), mode="valid"
        )
        single_scores = np.abs(single_corr) / np.sqrt(
            np.maximum(single_window_energy, single_floor) * single_energy
        )
        if candidate_count:
            repeated_scores = np.zeros(candidate_count, dtype=np.float64)
            for repeat_index in range(PREAMBLE_REPEATS):
                begin = preamble_cp + repeat_index * m
                repeated_scores += single_scores[begin : begin + candidate_count]
            repeated_scores /= PREAMBLE_REPEATS
            if search is None:
                repeat_start, repeat_end = 0, len(repeated_scores)
            else:
                repeat_start = max(0, int(search[0]))
                repeat_end = min(len(repeated_scores), int(search[1]))
            if repeat_end > repeat_start:
                relative = int(
                    np.argmax(repeated_scores[repeat_start:repeat_end])
                )
                repeat_offset = repeat_start + relative
                repeat_score = float(repeated_scores[repeat_offset])
                if repeat_score > best_noncoherent:
                    best_noncoherent = repeat_score
                    best_noncoherent_bin = k
                    best_noncoherent_offset = repeat_offset

    if best_noncoherent > best_score:
        best_offset = best_noncoherent_offset
        best_bin = best_noncoherent_bin
    noncoherent_score = max(0.0, best_noncoherent)
    coarse_cfo = frac_cfo + best_bin * RS

    if max(best_score, noncoherent_score) < TEMPLATE_SCORE_THRESHOLD:
        raise SyncError(
            "no preamble detected "
            f"(template scores coherent={best_score:.3f}, "
            f"noncoherent={noncoherent_score:.3f} < "
            f"{TEMPLATE_SCORE_THRESHOLD:.3f}; "
            f"repeat metric {peak_val:.3f})"
        )

    # Re-estimate fractional CFO at the matched timing using all adjacent
    # preamble repetitions. The initial autocorrelation peak may be a nearby
    # noise maximum; its phase is adequate for the template bank but leaves
    # enough residual rotation to cancel a repeated low-SNR header.
    useful_start = best_offset + preamble_cp
    useful_end = useful_start + PREAMBLE_REPEATS * m
    repeated = z_filt[useful_start:useful_end]
    if len(repeated) >= 2 * m:
        adjacent = repeated[m:] * np.conj(repeated[:-m])
        refined_frac = float(
            np.angle(np.sum(adjacent)) / (2 * np.pi * (m / fs))
        )
        integer_bins = int(round((coarse_cfo - refined_frac) / RS))
        total_cfo = refined_frac + integer_bins * RS
    else:
        total_cfo = coarse_cfo
    return Acquisition(
        preamble_start=best_offset,
        freq_offset=total_cfo,
        metric=max(best_score, noncoherent_score),
    )
