"""Coarse SDR tuning must not prevent the modem from seeing weak RF."""
import numpy as np
import pytest

from aetv.sdr_dsp import ModemToIQ, estimate_weak_signal_offset
from aetv.modem import modulate_continuous_chunks


@pytest.mark.parametrize("composite", [False, True])
def test_weak_band_edges_survive_noise_without_transmit_timing(composite):
    from aetv.settings import StationSettings
    from aetv.station import Station, TxEngine

    rng = np.random.default_rng(194)
    sent = rng.normal(size=(3, 19200)).astype(np.float32)
    chunks = modulate_continuous_chunks(sent, "AC16", total_gops=3)
    if composite:
        settings = StationSettings(mode="AC16", waveform_mode="analog_av", av_microphone_mix=0)
        t = np.arange(24000)/8000
        chunks = TxEngine(Station(settings))._composite_chunks(
            chunks, .2*np.sin(2*np.pi*730*t), 3, capture_microphone=False)
    audio = np.concatenate(list(chunks))
    tx = ModemToIQ(sample_rate=960000, offset_hz=-105050,
                   center_hz=10000 if composite else 8000, peak_limit=.9)
    # The receiver sees an arbitrary complete second of payload, not its start.
    iq = tx.feed(audio*.1)[960000:2*960000]
    # Composite default splits half its power into video. Set approximately
    # -4 dB video SNR inside its occupied band, including wideband ADC noise.
    video_fraction = .5 if composite else 1
    noise_power = np.mean(abs(iq)**2) * video_fraction * (960000/15000) / 10**(-4/10)
    iq = iq + np.sqrt(noise_power/2)*(rng.normal(size=len(iq))+1j*rng.normal(size=len(iq)))
    result = estimate_weak_signal_offset(iq, composite=composite)
    assert abs(result["offset_hz"] + 105050) <= 400


@pytest.mark.parametrize("interference", ["noise", "tone", "zero"])
def test_weak_frequency_estimator_rejects_idle_and_narrow_interference(interference):
    rng = np.random.default_rng(492)
    iq = rng.normal(size=960000) + 1j*rng.normal(size=960000)
    if interference == "tone":
        iq += 30*np.exp(-2j*np.pi*105000*np.arange(len(iq))/960000)
    elif interference == "zero":
        iq *= 0
    for composite in (False, True):
        with pytest.raises(ValueError):
            estimate_weak_signal_offset(iq, composite=composite)
