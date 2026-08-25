"""Analog-value invariants for the OFDM modem and GUI channel emulator."""

import numpy as np

from aetv.analog_validation import (
    aggregate_roundtrips,
    gui_channel_roundtrip,
    ideal_ofdm_roundtrip,
)
from aetv.config import reference_noise_bandwidth_scale
from aetv.hfchannel import (
    ChannelProfile,
    StreamingChannelEmulator,
    emulate,
    fading,
)


def test_linear_real_audio_ofdm_is_numerically_lossless_in_every_band():
    for mode_name in ("V0", "V1", "V7"):
        result = ideal_ofdm_roundtrip(mode_name, seed=20260825)
        assert result.decoded
        assert result.nmse < 1e-10
        assert result.correlation > 1.0 - 1e-10
        np.testing.assert_allclose(result.gain, 1.0, rtol=1e-6, atol=1e-6)


def test_gui_clean_profile_exposes_tx_conditioner_baseline_distortion():
    for mode_name in ("V0", "V1", "V7"):
        ideal = ideal_ofdm_roundtrip(mode_name, seed=41)
        clean = gui_channel_roundtrip(mode_name, "clean", seed=41)
        assert clean.decoded
        assert ideal.nmse < 1e-10
        # The clean channel itself is exact, but the production 0.5 dB
        # clip/filter TX conditioner is intentionally nonlinear.  Keep this
        # visible instead of describing the full GUI path as lossless.
        assert 0.05 < clean.nmse < 0.13
        assert clean.correlation > 0.95


def test_gui_awgn_conditions_order_latent_fidelity_and_confidence():
    summaries = {}
    for profile in ("clean", "awgn12", "awgn6", "awgn0"):
        rows = [gui_channel_roundtrip("V1", profile, seed) for seed in range(4)]
        summaries[profile] = aggregate_roundtrips(rows)
        assert summaries[profile]["decode_rate"] == 1.0

    ordered = [summaries[name] for name in ("clean", "awgn12", "awgn6", "awgn0")]
    assert all(a["mean_nmse"] < b["mean_nmse"] for a, b in zip(ordered, ordered[1:]))
    assert all(
        a["mean_correlation"] > b["mean_correlation"]
        for a, b in zip(ordered, ordered[1:])
    )
    assert all(
        a["mean_confidence"] > b["mean_confidence"]
        for a, b in zip(ordered, ordered[1:])
    )


def test_gui_multipath_damage_is_selective_and_reflected_in_confidence():
    rows = [gui_channel_roundtrip("V1", "mpp12", seed) for seed in range(6)]
    summary = aggregate_roundtrips(rows)
    clean = aggregate_roundtrips(
        [gui_channel_roundtrip("V1", "clean", seed) for seed in range(6)]
    )
    awgn = aggregate_roundtrips(
        [gui_channel_roundtrip("V1", "awgn12", seed) for seed in range(6)]
    )
    assert summary["decode_rate"] == 1.0
    assert summary["mean_nmse"] > awgn["mean_nmse"] > clean["mean_nmse"]
    assert summary["mean_confidence"] < awgn["mean_confidence"]
    # Deep time/frequency fades should receive lower confidence than the
    # better-preserved latent slots, hence error^2 correlates with 1-weight.
    assert summary["mean_damage_confidence_correlation"] > 0.20


def test_streaming_fading_is_invariant_to_process_chunk_boundaries():
    fs = 8_000
    time = np.arange(3 * fs) / fs
    audio = (
        np.cos(2 * np.pi * 700.0 * time)
        + 0.4 * np.sin(2 * np.pi * 1_900.0 * time)
    ).astype(np.float32)
    profile = ChannelProfile("fading-only", "Fading only", fading="mpp")
    whole = StreamingChannelEmulator(profile, seed=17, fs=fs).process(audio)
    chunked_channel = StreamingChannelEmulator(profile, seed=17, fs=fs)
    chunked = np.concatenate(
        [
            chunked_channel.process(audio[start : start + 733])
            for start in range(0, len(audio), 733)
        ]
    )
    np.testing.assert_allclose(chunked, whole, rtol=0.0, atol=1e-6)


def test_faded_profile_awgn_keeps_transmit_referenced_noise_floor():
    fs = 8_000
    time = np.arange(30 * fs) / fs
    audio = np.sin(2 * np.pi * 1_000.0 * time)
    profile = ChannelProfile("mpp-test", "MPP test", snr_db=12.0, fading="mpp")
    impaired = emulate(audio, profile, seed=23, fs=fs).astype(np.float64)
    faded = fading(audio, "mpp", seed=23, fs=fs)
    measured_noise_power = float(np.var(impaired - faded))
    expected_noise_power = (
        np.mean(audio**2)
        * reference_noise_bandwidth_scale(fs)
        / 10 ** (profile.snr_db / 10.0)
    )
    np.testing.assert_allclose(measured_noise_power, expected_noise_power, rtol=0.02)
