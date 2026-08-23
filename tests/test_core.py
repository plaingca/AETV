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
from aetv.config import PROTOCOL_VERSION
from aetv.modem import (
    StreamingDemodulator,
    modulate_continuous_chunks,
    modulate_gop_chunks,
)
from aetv.modem import _header_candidates, _header_carriers, decode_header, encode_header


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
    assert [len(chunk) for chunk in chunks] == [29760, *([24000] * 5), 26400]
    assert sum(map(len, chunks)) == 7 * 24000 + int(0.34 * 24000)

    receiver = StreamingDemodulator(mode.band, continuous=True)
    decoded = []
    audio = np.concatenate(chunks)
    for start in range(0, len(audio), 733):
        decoded.extend(receiver.feed(audio[start : start + 733]))
    assert len(decoded) == 7
    assert decoded[-1].callsign == "VE7TEST"


def test_continuous_v7_receiver_can_join_after_initial_header():
    mode = AETV_MODES["V7"]
    gops = [np.zeros(mode.latents_per_gop, dtype=np.float32) for _ in range(18)]
    audio = np.concatenate(list(modulate_continuous_chunks(gops, "V7", "VE7TEST")))
    # Enter at an arbitrary point well after the only RF preamble/header.
    audio = audio[42000:]
    events = []
    receiver = StreamingDemodulator(mode.band, continuous=True, on_debug=events.append)
    decoded = []
    for start in range(0, len(audio), 733):
        decoded.extend(receiver.feed(audio[start : start + 733]))
    assert any(event["event"] == "blind_acquired" for event in events)
    assert decoded
    assert decoded[-1].callsign == "VE7TEST"


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
