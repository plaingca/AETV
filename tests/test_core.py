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
    for mode_name, band, geom in [
        ("V0", "N", BAND_N),
        ("V1", "W", BAND_W),
        ("V7", "U", BAND_U),
    ]:
        gop_lat = np.random.randn(geom.latents_per_gop).astype(np.float32)
        audio = modulate_gop_stream([gop_lat], mode_name=mode_name, callsign="N0CALL")
        assert len(audio) > 0
        demod_res = demodulate_gop_stream(audio, band=band, drift_track="off")
        assert demod_res.frames_received >= 8
        assert len(demod_res.gops_latents) >= 1
        recovered_lat = demod_res.gops_latents[0]
        corr = np.corrcoef(gop_lat, recovered_lat)[0, 1]
        assert corr > 0.95, f"{mode_name}: expected high correlation on clean loopback, got {corr:.3f}"
