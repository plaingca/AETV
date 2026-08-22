"""DFT-matrix OFDM for AETV: complex carrier amplitudes <-> waveform samples.

Supports:
- Variant N (24 carriers, 1.2 kHz @ 8 kHz FS)
- Variant W (45 carriers, 2.25 kHz @ 8 kHz FS)
- Variant U (160 carriers, 8.0 kHz @ 24 kHz FS)
under the display-friendly 8 frames/s numerology (RS=50, 5.0 ms CP, 25.0 ms symbol).
"""

from __future__ import annotations

import numpy as np

from .config import (
    BANDS,
    DEMOD_BACKOFF,
    FS,
    M,
    NCP,
    NSYM,
    PREAMBLE_CP,
    PREAMBLE_REPEATS,
    PREAMBLE_SAMPLES,
    RS,
)


def _get_numerology(band: str) -> tuple[int, int, int, int]:
    geom = BANDS[band]
    fs = geom.fs
    m = fs // RS
    ncp = m // 4  # 5.0 ms CP (40 @ 8 kHz, 120 @ 24 kHz)
    nsym = m + ncp  # 25.0 ms full symbol (200 @ 8 kHz, 600 @ 24 kHz)
    return fs, m, ncp, nsym


def _phasor(cycles_num: np.ndarray, fs: int = FS, sign: int = 1) -> np.ndarray:
    """exp(sign * 2j*pi * cycles_num / fs) for integer cycles_num."""
    return np.exp(sign * 2j * np.pi * (np.asarray(cycles_num) % fs) / fs)


def carrier_frequencies(band: str = "W") -> np.ndarray:
    geometry = BANDS[band]
    return geometry.carrier0_hz + RS * np.arange(geometry.carriers)


def baseband_frequencies(band: str = "W") -> np.ndarray:
    geometry = BANDS[band]
    return carrier_frequencies(band) - geometry.fcenter_hz


def mod_matrix(band: str = "W") -> np.ndarray:
    fs, m, ncp, nsym = _get_numerology(band)
    n_sym = np.arange(nsym) - ncp
    return _phasor(np.outer(n_sym, carrier_frequencies(band)), fs=fs)  # (nsym, NC)


def demod_matrix(band: str = "W") -> np.ndarray:
    fs, m, ncp, nsym = _get_numerology(band)
    n_use = np.arange(m)
    return _phasor(np.outer(baseband_frequencies(band), n_use), fs=fs, sign=-1)  # (NC, m)


# Precomputed matrices for N, W, and U
_MOD_MATRICES = {b: mod_matrix(b) for b in BANDS}
_DEMOD_MATRICES = {b: demod_matrix(b) for b in BANDS}


def modulate_symbols(symbols: np.ndarray, band: str = "W") -> np.ndarray:
    """(n_sym, NC) complex symbols -> real passband waveform (n_sym * NSYM,)."""
    m = _MOD_MATRICES[band]
    x = np.real(m @ symbols.T)  # (NSYM, n_sym)
    return x.T.reshape(-1)


def demod_window(
    z: np.ndarray, start: int, band: str = "W", backoff: int = DEMOD_BACKOFF
) -> np.ndarray:
    """Demodulate one useful window of baseband signal starting at `start`.
    `backoff` shifts earlier into the cyclic prefix.
    Factor 2 undoes the amplitude halving of the analytic signal.
    """
    fs, m, ncp, nsym = _get_numerology(band)
    s = start - backoff
    win = z[s : s + m]
    if len(win) < m:
        win = np.pad(win, (0, m - len(win)))
    d = _DEMOD_MATRICES[band]
    return (2.0 / m) * (d @ win)


def pilot_sequence(band: str = "W") -> np.ndarray:
    """Fixed unit-magnitude QPSK pilot sequence for carrier count NC."""
    quads = np.asarray(BANDS[band].pilot_quadrants)
    phases = np.pi / 4 + np.pi / 2 * quads
    return np.exp(1j * phases)


def preamble_waveform(band: str = "W") -> np.ndarray:
    """Real passband preamble: pilot symbol repeated with double-length CP."""
    fs, m, ncp, nsym = _get_numerology(band)
    p = pilot_sequence(band)
    preamble_cp = 2 * ncp
    preamble_samples = preamble_cp + PREAMBLE_REPEATS * m
    n = np.arange(preamble_samples) - preamble_cp
    e = _phasor(np.outer(n, carrier_frequencies(band)), fs=fs)
    return np.real(e @ p)


def preamble_template(band: str = "W") -> np.ndarray:
    """Complex baseband replica of the preamble for timing correlation."""
    fs, m, ncp, nsym = _get_numerology(band)
    p = pilot_sequence(band)
    preamble_cp = 2 * ncp
    preamble_samples = preamble_cp + PREAMBLE_REPEATS * m
    n = np.arange(preamble_samples) - preamble_cp
    e = _phasor(np.outer(n, baseband_frequencies(band)), fs=fs)
    return e @ p
