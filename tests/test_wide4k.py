"""V9: the 4 kHz band M mode for 6 kHz transmit filters."""

import numpy as np
import torch

from aetv import framing, ofdm
from aetv.config import AETV_MODES, AETV_MODES_BY_INDEX, BAND_M, BANDS, LATENTS_PER_GOP_M
from aetv.modem import demodulate_gop_stream, modulate_continuous_chunks, modulate_gop_stream
from aetv.models import AETVAutoencoder, widen_latent_channels


def test_band_m_geometry_fits_a_6khz_filter():
    assert BANDS["M"] is BAND_M
    assert BAND_M.carriers == 76
    assert BAND_M.latent_carriers == 75
    assert BAND_M.beacon_carrier == 75
    assert BAND_M.fs == 12000
    assert LATENTS_PER_GOP_M == BAND_M.latents_per_gop == 75 * 4 * 2 * 8 == 4800
    top = BAND_M.carrier0_hz + 50 * (BAND_M.carriers - 1)
    assert 3.7e3 <= top - BAND_M.carrier0_hz + 50 <= 4.0e3
    low, high = BAND_M.tx_bandpass
    assert 300 <= low < BAND_M.carrier0_hz and top < high <= 6000 - 900
    assert high < BAND_M.fs / 2


def test_v9_mode_spec():
    mode = AETV_MODES["V9"]
    assert mode.band == "M" and mode.index == 9
    assert AETV_MODES_BY_INDEX[9] is mode
    assert (mode.width, mode.height, mode.fps, mode.gop_frames) == (192, 108, 6.0, 6)
    assert mode.latents_per_gop == 4800
    assert AETV_MODES["V8"].latents_per_gop < mode.latents_per_gop < AETV_MODES["V7"].latents_per_gop


def test_band_m_pilots_and_framing_round_trip():
    pilots = ofdm.pilot_sequence("M")
    assert len(pilots) == BAND_M.carriers and np.allclose(np.abs(pilots), 1.0)
    latents = np.random.default_rng(9).standard_normal(BAND_M.latents_per_gop).astype(np.float32)
    packed = framing.pack_gop_symbols(latents, np.ones(32, np.float32), band="M")
    assert packed.shape == (32, BAND_M.carriers)
    assert np.all(packed[:, BAND_M.beacon_carrier] == 1.0)
    unpacked, _ = framing.unpack_gop_symbols(packed, np.ones(packed.shape, np.float32), band="M")
    assert np.allclose(unpacked, latents, atol=1e-5)


def test_v9_clean_modem_loopback_reports_v9():
    original = np.random.default_rng(49).standard_normal(LATENTS_PER_GOP_M).astype(np.float32)
    audio = modulate_gop_stream([original], mode_name="V9", callsign="N0CALL")
    decoded = demodulate_gop_stream(audio, band="M", drift_track="off")
    assert decoded.mode.name == "V9"
    assert len(decoded.gops_latents) == 1
    assert np.corrcoef(original, decoded.gops_latents[0])[0, 1] > 0.95


def test_v9_transmit_waveform_stays_inside_6khz_filter():
    mode = AETV_MODES["V9"]
    rng = np.random.default_rng(4096)
    gops = [rng.standard_normal(mode.latents_per_gop).astype(np.float32) for _ in range(3)]
    audio = np.concatenate(list(modulate_continuous_chunks(gops, "V9")))
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / mode.geometry.fs)
    outside = spectrum[(frequencies < 700.0) | (frequencies > 5300.0)].sum()
    assert outside / spectrum.sum() < 1e-4


def test_widened_v8_decodes_exactly_like_v8_before_training():
    torch.manual_seed(0)
    v8 = AETVAutoencoder(mode=AETV_MODES["V8"], width=64, latent_channels=3).eval()
    v9 = AETVAutoencoder(mode=AETV_MODES["V9"], width=64, latent_channels=6).eval()
    v9.load_state_dict(widen_latent_channels(v8.state_dict(), 6), strict=True)
    video = torch.rand(1, 3, 6, 108, 192)
    used = 3 * 3 * 13 * 24  # V8 decoder grid: the transmitted values V8 reads
    emitted = 3 * 3 * 14 * 24  # V8 encoder grid (the encoder keeps a 14th row)
    with torch.no_grad():
        z8, z9 = v8.encoder(video), v9.encoder(video)
        # Repeated channels keep the transmitted scale.
        assert torch.allclose(z9[:, emitted:], z9[:, : LATENTS_PER_GOP_M - emitted], atol=1e-5)
        scale = float((z9[:, :used] * z8[:, :used]).sum() / z8[:, :used].square().sum())
        assert torch.allclose(z9[:, :used], scale * z8[:, :used], atol=1e-5)
        assert abs(scale - 1.0) < 0.1
        wire = torch.zeros(1, LATENTS_PER_GOP_M)
        wire[:, :used] = z8[:, :used]
        weights = torch.rand(1, LATENTS_PER_GOP_M)
        weights8 = torch.zeros(1, z8.shape[1])
        weights8[:, :used] = weights[:, :used]
        expected = v8.decoder(z8, weights8, output_shape=(6, 108, 192))
        # New channels start ignored, whatever arrives on them.
        wire[:, used:] = torch.randn(1, LATENTS_PER_GOP_M - used)
        actual = v9.decoder(wire, weights, output_shape=(6, 108, 192))
    assert torch.allclose(actual, expected, atol=1e-5)
