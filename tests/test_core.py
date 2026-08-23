"""Comprehensive test suite for AETV (Autoencoder Television)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aetv import (
    AETV_MODES,
    AETV_MODES_BY_INDEX,
    BAND_N,
    BAND_U,
    BAND_W,
    DATA_SYMS_PER_FRAME,
    FRAME_SAMPLES,
    FRAMES_PER_GOP,
    FS,
    LATENTS_PER_GOP_N,
    LATENTS_PER_GOP_U,
    LATENTS_PER_GOP_W,
    M,
    NCP,
    NSYM,
    RS,
    SYMS_PER_FRAME,
    AETVAutoencoder,
    AETVChannelConfig,
    AETVLatentChannel,
    AETVSyntheticVideoDataset,
    AETVWaveformChannel,
    demodulate_gop_stream,
    modulate_gop_stream,
)
from aetv import beacon, framing, ofdm
from aetv.config import PROTOCOL_VERSION, reference_noise_bandwidth_scale
from aetv.hfchannel import (
    CHANNEL_PROFILES,
    StreamingChannelEmulator,
    awgn,
    emulate,
    freq_shift,
)
from aetv.modem import (
    StreamingDemodulator,
    modulate_continuous_chunks,
    modulate_gop_chunks,
)
from aetv.modem import (
    _equalize_payload_symbol,
    _estimate_snr_db,
    _header_candidates,
    _header_carriers,
    _pilot_coherence,
    decode_header,
    encode_header,
)


def test_operator_channel_profiles_are_repeatable_and_clean_is_exact():
    tone = np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000).astype(np.float32)
    assert np.array_equal(emulate(tone, "clean", fs=8000), tone)
    first = emulate(tone, "awgn6", fs=8000)
    second = emulate(tone, CHANNEL_PROFILES["awgn6"], fs=8000)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, tone)


def test_reference_snr_uses_real_nyquist_noise_bandwidth():
    assert reference_noise_bandwidth_scale(8000) == pytest.approx(1.6)
    assert reference_noise_bandwidth_scale(24000) == pytest.approx(4.8)

    tone = np.sin(2 * np.pi * 1000 * np.arange(240000) / 24000)
    noisy = emulate(tone, "awgn12", seed=7, fs=24000)
    noise = noisy - tone
    expected = (
        np.mean(tone**2)
        * reference_noise_bandwidth_scale(24000)
        / 10 ** (12.0 / 10.0)
    )
    assert np.var(noise) == pytest.approx(expected, rel=0.02)


def test_streaming_channel_profiles_are_repeatable_across_gop_chunks():
    tone = np.sin(2 * np.pi * 1000 * np.arange(16000) / 8000).astype(np.float32)
    for profile in ("clean", "awgn6", "mpp6"):
        first = StreamingChannelEmulator(profile, fs=8000)
        second = StreamingChannelEmulator(profile, fs=8000)
        got = np.concatenate([first.process(tone[:8000]), first.process(tone[8000:])])
        repeated = np.concatenate(
            [second.process(tone[:8000]), second.process(tone[8000:])]
        )
        assert np.array_equal(got, repeated)
        assert np.isfinite(got).all()
    clean = StreamingChannelEmulator("clean", fs=8000)
    assert np.array_equal(clean.process(tone), tone)


def test_aetv_numerology():
    assert FS == 8000
    assert RS == 50
    assert M == 160
    assert NCP == 40
    assert NSYM == 200
    assert SYMS_PER_FRAME == 5
    assert DATA_SYMS_PER_FRAME == 4
    assert FRAME_SAMPLES == 1000  # 125 ms = 8 frames/s
    assert FRAMES_PER_GOP == 8  # exactly 1.000 s per GOP


def test_aetv_band_geometries():
    # Narrow band (24 carriers)
    assert BAND_N.carriers == 24
    assert BAND_N.carrier0_hz == 950
    assert BAND_N.latent_carriers == 23
    assert BAND_N.beacon_carrier == 23
    assert BAND_N.latents_per_frame == 23 * 4 * 2  # 184
    assert BAND_N.latents_per_gop == 184 * 8  # 1472
    assert LATENTS_PER_GOP_N == 1472

    # Wide band (45 carriers)
    assert BAND_W.carriers == 45
    assert BAND_W.carrier0_hz == 450
    assert BAND_W.latent_carriers == 44
    assert BAND_W.beacon_carrier == 44
    assert BAND_W.latents_per_frame == 44 * 4 * 2  # 352
    assert BAND_W.latents_per_gop == 352 * 8  # 2816
    assert LATENTS_PER_GOP_W == 2816

    # Flex-8k ultra-wide (160 carriers @ 24 kHz)
    assert BAND_U.carriers == 160
    assert BAND_U.carrier0_hz == 1000
    assert BAND_U.latent_carriers == 158
    assert BAND_U.fs == 24000
    assert BAND_U.latents_per_gop == 10112
    assert LATENTS_PER_GOP_U == 10112


def test_aetv_modes_specs():
    for name in ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]:
        mode = AETV_MODES[name]
        assert mode.name == name
        assert mode.index in AETV_MODES_BY_INDEX
        assert mode.latents_per_gop in (1472, 2816, 10112)
        assert mode.gop_frames in (1, 6, 10, 12)
        assert mode.pixels_per_latent > 10.0
    assert AETV_MODES["V7"].band == "U"
    assert AETV_MODES["V7"].width == 256
    assert AETV_MODES["V7"].height == 144
    assert AETV_MODES["V7"].fps == 12.0


def test_pilot_sequence_and_papr():
    for band, geom in [("N", BAND_N), ("W", BAND_W), ("U", BAND_U)]:
        p = ofdm.pilot_sequence(band)
        assert len(p) == geom.carriers
        assert np.allclose(np.abs(p), 1.0)


def test_beacon_encode_decode_superframe():
    frame_counter = 42
    callsign = "W1AW/7"
    mode_idx = 2
    chips = beacon.encode_superframe(frame_counter, callsign, mode_idx)
    assert len(chips) == beacon.SUPERFRAME_LEN

    # Soft decode
    decoded = beacon.decode_superframe(chips[beacon.SYNC_LEN :])
    assert decoded is not None
    dec_counter, dec_callsign, dec_mode = decoded
    assert dec_counter == frame_counter
    assert dec_callsign == "W1AW/7"
    assert dec_mode == mode_idx


def test_beacon_sync_is_gain_independent_and_accepts_inverted_iq():
    chips = beacon.encode_superframe(17, "VE7TEST", 7)
    padded = np.concatenate([np.zeros(9), -0.12 * chips, np.zeros(4)])
    decoded = beacon.find_beacon_superframe(padded)
    assert decoded is not None
    assert decoded.callsign == "VE7TEST"


def test_streaming_modem_allows_late_entry_and_recovers_beacon():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(12)]
    chunks = list(modulate_gop_chunks(iter(gops), "V7", "VE7TEST"))
    receiver = StreamingDemodulator(mode.band)
    decoded = []
    # Tune in partway through a GOP and use deliberately awkward callback sizes.
    audio = np.concatenate(chunks[1:])[9000:]
    for start in range(0, len(audio), 733):
        decoded.extend(receiver.feed(audio[start : start + 733]))
    assert len(decoded) >= 6
    assert decoded[-1].callsign == "VE7TEST"


def test_streaming_modem_rejects_noise_false_triggers():
    receiver = StreamingDemodulator("U")
    noise = np.random.default_rng(2026).standard_normal(120000).astype(np.float32)
    decoded = []
    for start in range(0, len(noise), 733):
        decoded.extend(receiver.feed(noise[start : start + 733]))
    assert decoded == []


def test_v7_repeated_beacon_identifies_within_seven_gops():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(7)]
    receiver = StreamingDemodulator(mode.band)
    decoded = []
    for chunk in modulate_gop_chunks(iter(gops), "V7", "VE7TEST"):
        decoded.extend(receiver.feed(chunk))
    assert decoded[-1].callsign == "VE7TEST"


def test_streaming_demodulator_is_invariant_to_low_kiwi_level():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(7)]
    audio = np.concatenate(list(modulate_gop_chunks(gops, "V7", "VE7TEST")))
    receiver = StreamingDemodulator(mode.band)
    decoded = []
    for start in range(0, len(audio), 733):
        decoded.extend(receiver.feed(0.001 * audio[start : start + 733]))
    assert len(decoded) == 7
    assert decoded[-1].callsign == "VE7TEST"


def test_streaming_demodulator_consumes_large_backlog_in_order():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(7)]
    audio = np.concatenate(list(modulate_gop_chunks(gops, "V7", "VE7TEST")))
    decoded = StreamingDemodulator(mode.band).feed(audio)
    assert len(decoded) == 7
    assert decoded[-1].callsign == "VE7TEST"


def test_continuous_v7_stream_has_exact_one_second_steady_state_gops():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(7)]
    chunks = list(modulate_continuous_chunks(gops, "V7", "VE7TEST"))
    assert [len(chunk) for chunk in chunks] == [37200, *([24000] * 5), 26400]
    assert sum(map(len, chunks)) == 7 * 24000 + 15600

    receiver = StreamingDemodulator(mode.band, continuous=True)
    decoded = []
    audio = np.concatenate(chunks)
    for start in range(0, len(audio), 733):
        decoded.extend(receiver.feed(audio[start : start + 733]))
    assert len(decoded) == 7
    assert decoded[-1].callsign == "VE7TEST"


def test_continuous_tx_clipping_matches_checkpoint_batch_contract():
    """The stateful FIR may add delay, but must not add extra clipping."""
    rng = np.random.default_rng(20260824)
    mode = AETV_MODES["V7"]
    latents = rng.standard_normal(mode.latents_per_gop).astype(np.float32)
    latents /= np.sqrt(np.mean(latents**2))

    batch_audio = modulate_gop_stream([latents], mode_name="V7")
    live_audio = np.concatenate(list(modulate_continuous_chunks([latents], "V7")))
    batch_latents = demodulate_gop_stream(batch_audio, band="U").gops_latents[0]
    live_latents = demodulate_gop_stream(live_audio, band="U").gops_latents[0]

    # Causal filtering shifts the preamble by its group delay; after acquisition
    # the payload distortion should be numerically the same as centered batch TX.
    assert np.mean((live_latents - batch_latents) ** 2) < 1e-8


def test_payload_equalization_is_invariant_to_receive_level():
    rng = np.random.default_rng(77)
    mode = AETV_MODES["V7"]
    latents = rng.standard_normal(mode.latents_per_gop).astype(np.float32)
    audio = modulate_gop_stream([latents], mode_name="V7")
    nominal = demodulate_gop_stream(audio, band="U")
    quiet = demodulate_gop_stream(0.001 * audio, band="U")
    assert np.allclose(nominal.gops_latents[0], quiet.gops_latents[0], atol=2e-5)
    assert np.allclose(nominal.gops_weights[0], quiet.gops_weights[0], atol=2e-5)


def test_v7_acquires_calibrated_zero_db_and_corrects_cfo():
    rng = np.random.default_rng(20260825)
    mode = AETV_MODES["V7"]
    latents = rng.standard_normal(mode.latents_per_gop).astype(np.float32)
    audio = np.concatenate(list(modulate_continuous_chunks([latents], "V7")))

    for seed in range(5):
        impaired = awgn(audio, 0.0, seed=seed, fs=mode.geometry.fs)
        results = StreamingDemodulator(
            "U", continuous=True, mode_name="V7"
        ).feed(impaired)
        assert len(results) == 1
        assert results[0].pilot_coherence > 0.09

    for offset in (-20.0, 20.0):
        shifted = freq_shift(audio, offset, fs=mode.geometry.fs)
        results = StreamingDemodulator(
            "U", continuous=True, mode_name="V7"
        ).feed(shifted)
        assert len(results) == 1
        assert results[0].freq_offset == pytest.approx(offset, abs=0.1)


def test_pilot_structure_rejects_unstructured_noise():
    rng = np.random.default_rng(9)
    noise = rng.standard_normal((8, 160)) + 1j * rng.standard_normal((8, 160))
    smooth = np.ones((8, 160), dtype=np.complex128)
    assert _pilot_coherence(noise) < 0.09
    assert _pilot_coherence(smooth) > 0.99


def test_pilot_snr_estimator_subtracts_noise_from_total_power():
    rng = np.random.default_rng(20260823)
    # U-band physical reference SNR is per-carrier SNR times 8000/2500.
    target_db = 0.0
    carrier_snr = 10 ** (target_db / 10.0) / (8000.0 / 2500.0)
    noise_power = 1.0 / carrier_snr
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((256, 160))
        + 1j * rng.standard_normal((256, 160))
    )
    estimate = _estimate_snr_db(1.0 + noise, band="U")
    assert estimate == pytest.approx(target_db, abs=0.7)


def test_payload_confidence_tracks_pilot_noise_power():
    received = np.ones(16, dtype=np.complex128)
    channel = np.ones(16, dtype=np.complex128)
    _, clean_weights = _equalize_payload_symbol(received, channel, 0.01)
    _, noisy_weights = _equalize_payload_symbol(received, channel, 10.0)
    assert np.mean(clean_weights) > 0.98
    assert np.mean(noisy_weights) < 0.10


def test_continuous_v7_receiver_can_join_after_initial_header():
    mode = AETV_MODES["V7"]
    rng = np.random.default_rng(20260826)
    # Real encoder latents occupy the payload carriers. All-zero payloads form
    # an artificial sequence of mostly empty OFDM symbols that can correlate
    # with a repeated pilot and are not representative of a transmitted GOP.
    gops = [
        rng.standard_normal(mode.latents_per_gop).astype(np.float32)
        for _ in range(18)
    ]
    audio = np.concatenate(list(modulate_continuous_chunks(gops, "V7", "VE7TEST")))
    # Enter at an arbitrary point well after the only RF preamble/header.
    audio = audio[42000:]
    events = []
    receiver = StreamingDemodulator(mode.band, continuous=True, on_debug=events.append)
    decoded = []
    for start in range(0, len(audio), 733):
        decoded.extend(receiver.feed(audio[start : start + 733]))
    assert decoded
    assert decoded[-1].callsign == "VE7TEST"
    assert any(
        event["event"] in {"blind_acquired", "preamble_candidate"}
        for event in events
    )


def test_continuous_v7_receiver_keeps_looking_when_started_before_tx():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(7)]
    transmission = np.concatenate(
        list(modulate_continuous_chunks(gops, "V7", "VE7TEST"))
    )
    # Starting reception in quiet/noisy spectrum used to put the state machine
    # permanently into blind-acquisition mode before the one-time preamble
    # arrived. Keep the prefix below the 12-second late-join observation time
    # so this specifically exercises continued startup-preamble searching.
    audio = np.concatenate(
        [np.zeros(2 * mode.geometry.fs, dtype=np.float32), transmission]
    )
    events = []
    receiver = StreamingDemodulator(
        mode.band, continuous=True, on_debug=events.append
    )
    decoded = []
    # Match the GUI's default two-second polling cadence. Acquisition must be
    # invariant to a preamble landing anywhere inside a large callback batch.
    batch = 2 * mode.geometry.fs
    for start in range(0, len(audio), batch):
        decoded.extend(receiver.feed(audio[start : start + batch]))

    assert len(decoded) == 7
    assert decoded[-1].callsign == "VE7TEST"
    assert not any(event["event"] == "blind_acquired" for event in events)


def test_continuous_v7_tracking_stops_at_post_transmission_noise():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(7)]
    transmission = np.concatenate(
        list(modulate_continuous_chunks(gops, "V7", "VE7TEST"))
    )
    noise = np.random.default_rng(7).standard_normal(
        3 * mode.geometry.fs
    ).astype(np.float32)
    events = []
    receiver = StreamingDemodulator(
        mode.band, continuous=True, on_debug=events.append
    )
    decoded = []
    audio = np.concatenate([transmission, noise])
    for start in range(0, len(audio), 4096):
        decoded.extend(receiver.feed(audio[start : start + 4096]))

    assert len(decoded) == 7
    assert decoded[-1].callsign == "VE7TEST"
    assert any(event["event"] == "tracking_lost" for event in events)


def test_v7_header_repetition_survives_loss_of_legacy_carriers():
    chips = encode_header(AETV_MODES["V7"].index)
    carriers = _header_carriers(chips, AETV_MODES["V7"].geometry.carriers)
    # Simulate a deep fade over the legacy first copy. The copies elsewhere in
    # the 8 kHz channel still identify the protocol and mode.
    carriers[:24] = 0.0
    legacy, combined = _header_candidates(carriers)
    assert decode_header(legacy) != (AETV_MODES["V7"].index, PROTOCOL_VERSION)
    assert decode_header(combined) == (AETV_MODES["V7"].index, PROTOCOL_VERSION)


def test_gop_framing_and_interleaving():
    for band, geom in [("N", BAND_N), ("W", BAND_W), ("U", BAND_U)]:
        latents = np.random.randn(geom.latents_per_gop).astype(np.float32)
        beacon_chips = np.ones(32, dtype=np.float32)

        packed = framing.pack_gop_symbols(latents, beacon_chips, band=band, interleave=True)
        assert packed.shape == (32, geom.carriers)
        if band == "U":
            assert np.all(packed[:, geom.latent_carriers] == 1.0 + 1.0j)
            assert np.all(packed[:, geom.beacon_carrier] == 1.0 + 1.0j)

        unpacked_lat, weights = framing.unpack_gop_symbols(
            packed, np.ones((32, geom.carriers), dtype=np.float32), band=band, interleave=True
        )
        assert np.allclose(unpacked_lat, latents, atol=1e-5)


@pytest.mark.parametrize("mode_name", ["V0", "V1", "V2", "V3", "V4", "V5"])
def test_aetv_autoencoder_forward_backward(mode_name):
    mode_spec = AETV_MODES[mode_name]
    model = AETVAutoencoder(mode=mode_spec, width=32, causal=mode_spec.causal)
    
    # Input video tensor (B, 3, T, H, W)
    b = 1
    video = torch.rand(b, 3, mode_spec.gop_frames, mode_spec.height, mode_spec.width, requires_grad=True)
    
    # Encoder produces unit-RMS latents
    latents = model.encoder(video)
    assert latents.shape == (b, mode_spec.latents_per_gop)
    rms = latents.square().mean().sqrt().item()
    assert rms == pytest.approx(1.0, abs=1e-2)

    # Decoder reconstructs video
    recon = model.decoder(latents, torch.ones_like(latents), output_shape=(mode_spec.gop_frames, mode_spec.height, mode_spec.width))
    assert recon.shape == video.shape
    
    # Backpropagation gradient flow
    loss = F.mse_loss(recon, video)
    loss.backward()
    assert video.grad is not None
    assert torch.isfinite(video.grad).all()


def test_aetv_latent_channel_stage1():
    channel = AETVLatentChannel()
    z = torch.randn(2, 2816, requires_grad=True)
    noisy_z, weights = channel(z)
    assert noisy_z.shape == z.shape
    assert weights.shape == z.shape
    loss = (noisy_z * weights).sum()
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_aetv_channel_ota_focus_mixture_can_override_broad_range():
    generator = torch.Generator().manual_seed(7)
    channel = AETVLatentChannel(
        AETVChannelConfig(
            snr_db_range=(-20.0, -20.0),
            snr_focus_range=(20.0, 20.0),
            p_snr_focus=1.0,
        )
    )
    z = torch.zeros(8, 4096)
    noisy, _ = channel(z, generator=generator)
    # 20 dB amplitude SNR produces sigma=0.1 and variance ~=0.01. Sampling
    # the broad -20 dB range instead would produce variance ~=100.
    assert noisy.square().mean().item() == pytest.approx(0.01, rel=0.08)


def test_aetv_waveform_channel_stage2():
    torch.manual_seed(0)
    for band in ["N", "W"]:
        channel = AETVWaveformChannel(
            band=band,
            cfg=AETVChannelConfig(snr_db_range=(30.0, 30.0), p_fading=0.0),
        )
        n_latents = channel.N_LATENTS
        z = torch.randn(1, n_latents, requires_grad=True)
        noisy_z, weights = channel(z)
        assert noisy_z.shape == z.shape
        assert weights.shape == z.shape
        
        # High SNR should give low error
        err = F.mse_loss(noisy_z, z)
        assert err.item() < 0.1
        
        # Gradient flow through differentiable OFDM waveform
        err.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()


def test_aetv_end_to_end_modem_clean_loopback():
    rng = np.random.default_rng(0)
    for mode_name, band, geom in [
        ("V0", "N", BAND_N),
        ("V1", "W", BAND_W),
        ("V7", "U", BAND_U),
    ]:
        gop_lat = rng.standard_normal(geom.latents_per_gop).astype(np.float32)
        audio = modulate_gop_stream([gop_lat], mode_name=mode_name, callsign="N0CALL")
        assert len(audio) > 0
        demod_res = demodulate_gop_stream(audio, band=band, drift_track="off")
        assert demod_res.frames_received >= 8
        assert len(demod_res.gops_latents) >= 1
        recovered_lat = demod_res.gops_latents[0]
        corr = np.corrcoef(gop_lat, recovered_lat)[0, 1]
        assert corr > 0.95, f"{mode_name}: expected high correlation on clean loopback, got {corr:.3f}"


def test_v7_tracked_gops_match_continuous_batch_equalization():
    """Later streaming GOPs must not be level-shrunk by synthetic sync audio."""
    rng = np.random.default_rng(20260823)
    originals = [
        rng.standard_normal(BAND_U.latents_per_gop).astype(np.float32)
        for _ in range(2)
    ]
    audio = np.concatenate(
        list(modulate_continuous_chunks(originals, mode_name="V7", callsign="N0CALL"))
    )
    batch = demodulate_gop_stream(audio, band="U", drift_track="off")
    streaming = StreamingDemodulator("U", continuous=True, mode_name="V7")
    tracked = []
    for start in range(0, len(audio), BAND_U.fs // 10):
        for result in streaming.feed(audio[start : start + BAND_U.fs // 10]):
            tracked.extend(result.gops_latents)

    assert len(tracked) == len(batch.gops_latents) == 2
    for original, expected, recovered in zip(originals, batch.gops_latents, tracked):
        relative_difference = np.mean((recovered - expected) ** 2) / np.mean(original**2)
        assert relative_difference < 1e-4
