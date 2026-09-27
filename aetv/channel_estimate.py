"""2-D LMMSE pilot channel estimation and demod-window placement.

Adapted from SSTVAE's "Receiver improvements ported from Data2G"
(https://github.com/arodland/SSTVAE/pull/58, Artistic License 2.0). Across
carriers the pilots are projected onto the measured delay support; across time
a Wiener interpolator with a Gaussian (ITU-R F.1487) Doppler model predicts the
channel at every data symbol. Unlike SSTVAE, which measures everything from a
single transmission's pilots, the delay profile and Doppler correlation here
are accumulated across GOPs, and the receiver falls back to its original
estimator when those statistics are too noisy to trust.

Delays are in samples and are measured relative to the start of the unshifted
FFT window. A path with delay ``d`` is free of inter-symbol interference when
``0 <= d - window_shift <= ncp``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import BANDS, DATA_SYMS_PER_FRAME, DEMOD_BACKOFF, FRAMES_PER_GOP, RS, SYMS_PER_FRAME

FRAME_S = SYMS_PER_FRAME * 1.25 / RS
PEAK_GATE = 10 ** (-15 / 10)
# SSTVAE gates the delay support at 2x the median floor. With AETV's 8-pilot
# GOPs that admits noise peaks below about 3 dB pilot SNR, widening the
# projection until LMMSE loses to the old estimator on mpp0/mpp3; 3x-8x
# measured the same, and high-SNR channels are gated by the -15 dB rule.
SUPPORT_FLOOR = 4.0
PLACEMENT_FLOOR = 4.0
# A noise-only profile averaged over 8+ pilots has peak/median near 2; the
# weakest usable signals measured (awgn -3, mpp0) stay above 6.
MIN_PROFILE_CONTRAST = 4.0
MIN_PILOT_SNR = 10 ** (-6 / 10)
MIN_PILOTS = 8
MIN_SPREAD_HZ = 0.02
MAX_SPREAD_HZ = 4.0
PLACEMENT_MARGIN = 2
CONTEXT_FRAMES = 2


def band_numerology(band: str) -> tuple[int, int, int]:
    """Return (fs, useful symbol length, cyclic prefix) for ``band``."""
    fs = BANDS[band].fs
    m = fs // RS
    return fs, m, m // 4


def shift_limits(band: str) -> tuple[int, int]:
    """Window shifts that keep every GOP symbol inside a one-GOP slice."""
    _fs, _m, ncp = band_numerology(band)
    return -(ncp - DEMOD_BACKOFF), DEMOD_BACKOFF


def _gaussian_correlation(dt_frames: np.ndarray, spread_hz: float) -> np.ndarray:
    return np.exp(-2.0 * (np.pi * spread_hz / 2.0 * dt_frames * FRAME_S) ** 2)


def _spread_from_correlation(rho: float) -> float:
    rho = float(np.clip(rho, 1e-3, 0.9999))
    spread = 2.0 * math.sqrt(-math.log(rho) / 2.0) / (math.pi * FRAME_S)
    return float(np.clip(spread, MIN_SPREAD_HZ, MAX_SPREAD_HZ))


@dataclass
class Support:
    first: int
    last: int
    contrast: float


class ChannelStatistics:
    """Delay profile and pilot correlation accumulated across GOPs.

    Each update is normalized by its own pilot power, so GOPs demodulated at
    different receive gains (the tracked path normalizes every GOP by its
    peak) contribute on the same scale. ``memory_gops`` sets the exponential
    forgetting horizon.
    """

    def __init__(self, band: str, memory_gops: float = 16.0):
        geom = BANDS[band]
        self.band = band
        self.fs, self.m, self.ncp = band_numerology(band)
        self.carriers = geom.carriers
        self.frequencies = geom.carrier0_hz + RS * np.arange(geom.carriers) - geom.fcenter_hz
        self.delays = np.arange(-2 * self.ncp, 2 * self.ncp + 1)
        self.taper = np.hanning(geom.carriers + 2)[1:-1]
        self.margin = int(math.ceil(self.fs / (geom.carriers * RS)))
        self.decay_per_pilot = math.exp(-1.0 / (memory_gops * FRAMES_PER_GOP))
        self.window_shift = 0
        self.reset()

    def reset(self) -> None:
        self.profile = np.zeros(len(self.delays))
        self.profile_weight = 0.0
        self.lag = 0j
        self.lag_power = 0.0
        self.lag_weight = 0.0
        self.window_shift = 0

    def steering(self, delays: np.ndarray) -> np.ndarray:
        return np.exp(-2j * np.pi * np.outer(self.frequencies, delays) / self.fs)

    def update_profile(self, pilots: np.ndarray, window_shift: int = 0, weight: float | None = None) -> None:
        """Add pilot observations taken with the window moved by ``window_shift``."""
        h = np.atleast_2d(np.asarray(pilots))
        scale = float(np.mean(np.abs(h) ** 2))
        if scale <= 1e-18 or not np.isfinite(scale):
            return
        steer = self.steering(self.delays - window_shift)
        power = np.abs((h * self.taper) @ np.conj(steer)) ** 2
        count = float(len(h) if weight is None else weight)
        decay = self.decay_per_pilot ** count
        self.profile = decay * self.profile + count * np.mean(power, axis=0) / scale
        self.profile_weight = decay * self.profile_weight + count

    def support(self, floor: float = SUPPORT_FLOOR) -> Support | None:
        """First and last profile peak above max(peak - 15 dB, floor x median)."""
        if self.profile_weight <= 0:
            return None
        prof = self.profile / self.profile_weight
        median = float(np.median(prof))
        peak = float(prof.max())
        if median <= 0:
            return None
        threshold = max(peak * PEAK_GATE, floor * median)
        interior = prof[1:-1]
        is_peak = (interior >= threshold) & (interior >= prof[:-2]) & (interior >= prof[2:])
        index = np.flatnonzero(is_peak) + 1
        if not index.size:
            index = np.array([int(np.argmax(prof))])
        return Support(int(self.delays[index[0]]), int(self.delays[index[-1]]), peak / median)

    def update_correlation(self, lag: complex, power: float, pairs: float) -> None:
        decay = self.decay_per_pilot ** pairs
        self.lag = decay * self.lag + lag
        self.lag_power = decay * self.lag_power + power
        self.lag_weight = decay * self.lag_weight + pairs

    def doppler_spread(self) -> float | None:
        if self.lag_weight < FRAMES_PER_GOP - 2 or self.lag_power <= 0:
            return None
        return _spread_from_correlation(abs(self.lag) / self.lag_power)


def placement_shift(stats: ChannelStatistics, current: int = 0) -> int:
    """Return the window shift for the accumulated delay profile.

    The window moves only when a path above the placement gate lies outside
    the cyclic prefix at ``current``; then it moves just far enough to bring
    it inside with a small margin. Blind centring cost 0.4-0.7 dB on the
    40 m OTA-fitted channel, where the paths already fit.
    """
    low, high = shift_limits(stats.band)
    if stats.profile_weight < MIN_PILOTS:
        return current
    support = stats.support(PLACEMENT_FLOOR)
    if support is None or support.contrast < MIN_PROFILE_CONTRAST:
        return current
    d0, d1, ncp = support.first, support.last, stats.ncp
    if d0 - current >= 0 and d1 - current <= ncp:
        return current
    margin = PLACEMENT_MARGIN
    if d1 - d0 > ncp - 2 * margin:
        shift = int(round((d0 + d1 - ncp) / 2))
    elif d0 - current < 0:
        shift = d0 - margin
    else:
        shift = d1 - ncp + margin
    return int(np.clip(shift, low, high))


def _projection(stats: ChannelStatistics, support: Support, window_shift: int) -> tuple[np.ndarray, int]:
    first = support.first - window_shift - stats.margin
    last = support.last - window_shift + stats.margin
    basis = stats.steering(np.arange(first, last + 1))
    lam, vectors = np.linalg.eigh(basis @ basis.conj().T)
    u = vectors[:, lam > lam[-1] * 1e-4]
    return u @ u.conj().T, u.shape[1]


def lmmse_payload_channels(
    pilots: np.ndarray,
    stats: ChannelStatistics,
    window_shift: int = 0,
    context: int = CONTEXT_FRAMES,
) -> list[np.ndarray | None]:
    """Channel at every data symbol, one ``(8, 4, carriers)`` array per GOP.

    ``pilots`` holds whole GOPs of raw pilot estimates ``received / pilot``.
    Each GOP is interpolated from its own pilots plus up to ``context``
    neighbouring pilots on each side, so its last frame is no longer
    equalized with a stale pilot. The profile must already include these
    pilots (``stats.update_profile``); their temporal correlation is added
    here. A ``None`` entry means the statistics fail the noise gates and the
    caller must use its fallback estimator for that GOP.
    """
    h = np.asarray(pilots)
    n_frames, nc = h.shape
    n_gops = n_frames // FRAMES_PER_GOP
    support = stats.support(SUPPORT_FLOOR)
    if (
        n_gops == 0
        or support is None
        or stats.profile_weight < MIN_PILOTS
        or support.contrast < MIN_PROFILE_CONTRAST
    ):
        return [None] * n_gops
    projector, rank = _projection(stats, support, window_shift)
    hs_all = h @ projector.T
    residual_scale = nc / max(nc - rank, 1)

    blocks = []
    for g in range(n_gops):
        lo = max(0, g * FRAMES_PER_GOP - context)
        hi = min(n_frames, (g + 1) * FRAMES_PER_GOP + context)
        hs = hs_all[lo:hi]
        n0 = float(np.mean(np.abs(h[lo:hi] - hs) ** 2)) * residual_scale
        n0_projected = n0 * rank / nc
        p_sig = float(np.mean(np.abs(hs) ** 2)) - n0_projected
        lag = complex(np.mean(hs[1:] * np.conj(hs[:-1])))
        blocks.append((lo, hi, hs, n0, n0_projected, p_sig, lag))
        own = hs_all[g * FRAMES_PER_GOP : (g + 1) * FRAMES_PER_GOP]
        own_power = float(np.mean(np.abs(own) ** 2))
        if own_power > 1e-18:
            own_lag = complex(np.mean(own[1:] * np.conj(own[:-1])))
            pairs = float(FRAMES_PER_GOP - 1)
            stats.update_correlation(
                own_lag * pairs / own_power, max(own_power - n0_projected, 0.0) * pairs / own_power, pairs
            )
    spread = stats.doppler_spread()
    stats.pilot_snr = [max(b[5], 0.0) / max(b[3], 1e-18) for b in blocks]

    out: list[np.ndarray | None] = []
    offsets = np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME
    for g, (lo, hi, hs, n0, n0_projected, p_sig, lag) in enumerate(blocks):
        if spread is None or p_sig <= MIN_PILOT_SNR * n0:
            out.append(None)
            continue
        rotation = float(np.angle(lag)) / (2 * np.pi)
        times = np.arange(lo, hi, dtype=np.float64)
        derotated = hs * np.exp(-2j * np.pi * rotation * times)[:, None]
        rpp = p_sig * _gaussian_correlation(times[:, None] - times[None, :], spread)
        rpp += n0_projected * np.eye(len(times))
        frames = np.arange(g * FRAMES_PER_GOP, (g + 1) * FRAMES_PER_GOP, dtype=np.float64)
        targets = (frames[:, None] + offsets[None, :]).reshape(-1)
        rdp = p_sig * _gaussian_correlation(targets[:, None] - times[None, :], spread)
        weights = np.linalg.solve(rpp, rdp.T).T
        estimate = (weights @ derotated) * np.exp(2j * np.pi * rotation * targets)[:, None]
        out.append(estimate.reshape(FRAMES_PER_GOP, DATA_SYMS_PER_FRAME, nc))
    return out
