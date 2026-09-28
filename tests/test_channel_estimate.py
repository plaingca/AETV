"""LMMSE channel estimation, window placement, and Gaussian fading taps."""

import copy

import numpy as np
import pytest
from scipy import signal

from aetv import modem
from aetv.beacon import generate_beacon_chips
from aetv.channel_estimate import (
    ChannelStatistics,
    lmmse_payload_channels,
    placement_shift,
    shift_limits,
)
from aetv.config import AETV_MODES, BANDS, FRAMES_PER_GOP, RS, SYMS_PER_FRAME
from aetv.hfchannel import RESEARCH_PROFILES, _gaussian_taps, _rayleigh_taps, awgn, emulate


@pytest.fixture
def receiver(monkeypatch):
    def configure(estimator: str, placement: bool) -> None:
        monkeypatch.setattr(modem, "CHANNEL_ESTIMATOR", estimator)
        monkeypatch.setattr(modem, "WINDOW_PLACEMENT", placement)

    return configure


def _frequencies(band: str) -> np.ndarray:
    geom = BANDS[band]
    return geom.carrier0_hz + RS * np.arange(geom.carriers) - geom.fcenter_hz


def _paths(band: str, delays, gains) -> np.ndarray:
    """Frequency response (frames, carriers) of fixed or time-varying paths."""
    f = _frequencies(band)
    fs = BANDS[band].fs
    gains = np.atleast_2d(np.asarray(gains))
    steer = np.exp(-2j * np.pi * np.outer(f, delays) / fs)
    return gains @ steer.T


def _noise(rng, shape, variance):
    return np.sqrt(variance / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _effective_snr(x, y):
    rho2 = min(float(np.corrcoef(x, y)[0, 1]) ** 2, 0.999999)
    return 10 * np.log10(rho2 / (1 - rho2))


def test_delay_support_finds_both_paths():
    stats = ChannelStatistics("W")
    rng = np.random.default_rng(1)
    h = _paths("W", [-8, 8], np.ones((8, 2))) + _noise(rng, (8, 45), 0.05)
    stats.update_profile(h)
    support = stats.support()
    assert abs(support.first + 8) <= 1 and abs(support.last - 8) <= 1
    assert support.contrast > 50


def test_profile_is_invariant_to_receive_gain():
    rng = np.random.default_rng(2)
    h = _paths("W", [8], np.ones((8, 1))) + _noise(rng, (8, 45), 0.1)
    quiet, loud = ChannelStatistics("W"), ChannelStatistics("W")
    quiet.update_profile(h)
    loud.update_profile(1000 * h)
    assert np.allclose(quiet.profile, loud.profile)


def test_profile_accumulates_across_gops_with_forgetting():
    stats = ChannelStatistics("W", memory_gops=4)
    h = _paths("W", [8], np.ones((8, 1)))
    for _ in range(40):
        stats.update_profile(h)
    assert stats.profile_weight == pytest.approx(FRAMES_PER_GOP * 4, rel=0.15)


def test_profile_tracks_window_shift_in_absolute_delay():
    stats = ChannelStatistics("W")
    # Measured through a window moved 10 samples earlier, a path at absolute
    # delay -6 appears at +4.
    stats.update_profile(_paths("W", [4], np.ones((8, 1))), window_shift=-10)
    support = stats.support()
    assert support.first == support.last == -6


@pytest.mark.parametrize("band", ["W", "M", "U"])
def test_placement_moves_only_when_a_path_is_outside_the_cp(band):
    fs = BANDS[band].fs
    ncp = fs // RS // 4
    rng = np.random.default_rng(3)

    def shift_for(delays):
        stats = ChannelStatistics(band)
        h = _paths(band, delays, np.ones((8, len(delays)))) + _noise(rng, (8, BANDS[band].carriers), 0.02)
        stats.update_profile(h)
        return placement_shift(stats, 0)

    # AWGN-like single path at the nominal backoff, and a path late in the CP.
    assert shift_for([8]) == 0
    assert shift_for([8, ncp - 6]) == 0
    # Earlier path 2 ms before the later one: the window moves just far
    # enough to include it, and no further.
    early = 8 - int(0.002 * fs)
    moved = shift_for([early, 8])
    assert moved == early - 2
    assert early - moved >= 0 and 8 - moved <= ncp
    assert shift_limits(band)[0] <= moved


def test_placement_ignores_a_noise_only_profile():
    rng = np.random.default_rng(4)
    stats = ChannelStatistics("W")
    stats.update_profile(_noise(rng, (16, 45), 1.0))
    assert placement_shift(stats, 0) == 0
    assert placement_shift(stats, -7) == -7


def _pilot_error(estimate, truth):
    return float(np.mean(np.abs(estimate - truth) ** 2))


def _fading_scene(rng, spread_hz=0.5, n_gops=1, noise=0.05):
    """Two paths with smooth fading: pilots, data-symbol truth, and raw pilots."""
    band = "W"
    frames = n_gops * FRAMES_PER_GOP
    t = np.arange(frames * SYMS_PER_FRAME) / SYMS_PER_FRAME * 0.125
    gains = np.stack(
        [np.exp(1j * (2 * np.pi * spread_hz * t * k + phase)) * (0.8 + 0.3 * np.cos(2 * np.pi * 0.3 * t + k))
         for k, phase in ((1, 0.3), (-1, 1.9))],
        axis=1,
    )
    truth = _paths(band, [-4, 12], gains).reshape(frames, SYMS_PER_FRAME, -1)
    pilots = truth[:, 0] + _noise(rng, truth[:, 0].shape, noise)
    return pilots, truth[:, 1:], noise


def test_lmmse_beats_the_pilot_estimator_and_fixes_the_stale_last_frame():
    rng = np.random.default_rng(5)
    pilots, truth, noise = _fading_scene(rng)
    stats = ChannelStatistics("W")
    stats.update_profile(pilots)
    (estimate,) = lmmse_payload_channels(pilots, stats)
    baseline = modem._pilot_gop_channels(pilots, noise)
    assert _pilot_error(estimate, truth) < 0.5 * _pilot_error(baseline, truth)
    assert _pilot_error(estimate[-1], truth[-1]) < 0.5 * _pilot_error(baseline[-1], truth[-1])


def test_lmmse_uses_neighbouring_gops_at_the_boundary():
    rng = np.random.default_rng(6)
    pilots, truth, _noise_var = _fading_scene(rng, n_gops=3, noise=0.2)
    stats = ChannelStatistics("W")
    stats.update_profile(pilots)
    with_context = lmmse_payload_channels(pilots, copy.deepcopy(stats))
    alone = lmmse_payload_channels(pilots, copy.deepcopy(stats), context=0)
    boundary = slice(FRAMES_PER_GOP - 1, FRAMES_PER_GOP)
    assert _pilot_error(with_context[0][boundary], truth[boundary]) < _pilot_error(alone[0][boundary], truth[boundary])


def test_lmmse_falls_back_when_the_profile_is_noise():
    rng = np.random.default_rng(7)
    pilots = _noise(rng, (FRAMES_PER_GOP, 45), 1.0)
    stats = ChannelStatistics("W")
    stats.update_profile(pilots)
    assert lmmse_payload_channels(pilots, stats) == [None]
    channels, used = modem._payload_channels(pilots, np.array([1.0]), "W", stats, 0)
    assert used == 0
    assert np.array_equal(channels, modem._pilot_gop_channels(pilots, 1.0))


def test_doppler_spread_is_measured_from_accumulated_pilots():
    slow, fast = ChannelStatistics("W"), ChannelStatistics("W")
    for stats, spread in ((slow, 0.1), (fast, 2.0)):
        rng = np.random.default_rng(8)
        for _ in range(6):
            pilots, _truth, _n = _fading_scene(rng, spread_hz=spread / 2, noise=0.01)
            stats.update_profile(pilots)
            lmmse_payload_channels(pilots, stats)
    assert slow.doppler_spread() < 0.5 < fast.doppler_spread()


def _two_path_capture(mode_name, seed=5, snr_db=25.0):
    mode = AETV_MODES[mode_name]
    fs = mode.geometry.fs
    rng = np.random.default_rng(seed)
    latents = np.tanh(rng.normal(size=mode.geometry.latents_per_gop)).astype(np.float32)
    tx = modem.modulate_gop_stream([latents], mode_name=mode_name)
    z = signal.hilbert(tx)
    delay = int(0.002 * fs)
    # The stronger path arrives 2 ms late, so acquisition locks onto it and
    # the earlier path falls before the FFT window.
    rx = 0.6 * z + np.exp(1j) * np.concatenate([np.zeros(delay), z[:-delay]])
    return latents, awgn(np.real(rx), snr_db, seed=1, fs=fs)


@pytest.mark.parametrize("mode_name", ["V8", "V9", "V7"])
def test_placement_and_lmmse_each_raise_latent_snr_on_a_late_lock(receiver, mode_name):
    latents, rx = _two_path_capture(mode_name)
    band = AETV_MODES[mode_name].band
    snr = {}
    for estimator, placement in (("pilot", False), ("pilot", True), ("lmmse", False), ("lmmse", True)):
        receiver(estimator, placement)
        result = modem.demodulate_gop_stream(rx, band=band)
        snr[estimator, placement] = _effective_snr(latents, result.gops_latents[0] * result.gops_weights[0])
        assert (result.window_shift < 0) == placement
    assert snr["pilot", True] > snr["pilot", False] + 1.5
    assert snr["lmmse", False] > snr["pilot", False] + 0.5
    assert snr["lmmse", True] > snr["pilot", True] + 0.5


def test_pilot_arm_keeps_the_original_receiver(receiver):
    receiver("pilot", False)
    latents, rx = _two_path_capture("V8")
    result = modem.demodulate_gop_stream(rx, band="W")
    assert result.channel_stats is None and result.window_shift == 0 and result.lmmse_gops == 0


def test_tracked_gop_does_not_mutate_the_callers_statistics(receiver):
    receiver("lmmse", True)
    mode = AETV_MODES["V8"]
    rng = np.random.default_rng(9)
    latents = rng.standard_normal(mode.latents_per_gop).astype(np.float32)
    chips = generate_beacon_chips(n_frames=FRAMES_PER_GOP, callsign="N0CALL", mode_index=mode.index)
    payload = awgn(modem._payload_wave(latents, chips, mode, interleave=True), 25.0, seed=2)
    stats = ChannelStatistics("W")
    result = modem.demodulate_tracked_gop(payload, mode, channel_stats=stats)
    assert stats.profile_weight == 0
    assert result.channel_stats.profile_weight == FRAMES_PER_GOP
    assert result.lmmse_gops == 1
    assert np.corrcoef(latents, result.gops_latents[0])[0, 1] > 0.97


def test_noiseless_pilots_keep_the_exact_pilot_estimate(receiver):
    receiver("lmmse", True)
    mode = AETV_MODES["V8"]
    rng = np.random.default_rng(11)
    latents = rng.standard_normal(mode.latents_per_gop).astype(np.float32)
    chips = generate_beacon_chips(n_frames=FRAMES_PER_GOP, callsign="N0CALL", mode_index=mode.index)
    payload = modem._payload_wave(latents, chips, mode, interleave=True)
    result = modem.demodulate_tracked_gop(payload, mode)
    assert result.lmmse_gops == 0


def test_streaming_receiver_accumulates_statistics_across_gops(receiver):
    receiver("lmmse", True)
    mode = AETV_MODES["V8"]
    rng = np.random.default_rng(10)
    gops = [rng.standard_normal(mode.latents_per_gop).astype(np.float32) for _ in range(5)]
    tx = np.concatenate(list(modem.modulate_continuous_chunks(gops, "V8", "N0CALL")))
    rx = emulate(tx, "awgn12", seed=3, fs=mode.geometry.fs)
    demod = modem.StreamingDemodulator("W", continuous=True, mode_name="V8", boundary_tracking=True)
    results = []
    for start in range(0, len(rx), 800):
        results.extend(demod.feed(rx[start:start + 800]))
    results.extend(demod.feed(np.zeros(2 * mode.geometry.fs, dtype=np.float32)))
    assert len(results) == 5
    weights = [r.channel_stats.profile_weight for r in results]
    assert all(b > a for a, b in zip(weights, weights[1:]))
    assert demod._channel_stats is results[-1].channel_stats
    assert all(r.lmmse_gops == 1 and r.window_shift == 0 for r in results)


def test_gaussian_taps_follow_the_f1487_correlation_and_mpp12_is_unchanged():
    fs = 8000
    tap = _gaussian_taps(fs * 400, 1.0, np.random.default_rng(1), fs)
    assert np.mean(np.abs(tap) ** 2) == pytest.approx(1.0)
    for lag in (0.125, 0.25, 0.5):
        k = int(lag * fs)
        rho = abs(np.mean(tap[k:] * np.conj(tap[:-k])))
        assert rho == pytest.approx(np.exp(-2 * (np.pi * 0.5 * lag) ** 2), abs=0.05)
    # mpp12 keeps the historical Butterworth generator (its score bar).
    butter = _rayleigh_taps(fs * 400, 1.0, np.random.default_rng(1), fs)
    assert abs(np.mean(butter[fs // 2:] * np.conj(butter[: -fs // 2]))) < 0.15
    tone = np.sin(2 * np.pi * 1000 * np.arange(fs * 2) / fs).astype(np.float32)
    assert not np.array_equal(emulate(tone, "mpp12", seed=4), emulate(tone, RESEARCH_PROFILES["mpp12-gauss"], seed=4))
    assert np.array_equal(emulate(tone, "mpp12-gauss", seed=4), emulate(tone, RESEARCH_PROFILES["mpp12-gauss"], seed=4))
