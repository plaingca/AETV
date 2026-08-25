import numpy as np

from aetv.analog_av import (
    AETV_HIGH_HZ,
    AETV_LOW_HZ,
    COMPOSITE_FS,
    compose_delayed_stream,
    extract_aetv,
    extract_voice,
    prepare_voice,
    translate_aetv_up,
)


def _tone_power(values: np.ndarray, frequency_hz: float, fs: int) -> float:
    spectrum = np.fft.rfft(values)
    frequency = np.fft.rfftfreq(len(values), 1.0 / fs)
    return float(np.abs(spectrum[np.argmin(np.abs(frequency - frequency_hz))]) ** 2)


def test_voice_filter_rejects_content_above_2200_hz():
    time = np.arange(8000) / 8000
    source = np.sin(2 * np.pi * 1000 * time) + np.sin(2 * np.pi * 3000 * time)
    voice = prepare_voice(source)
    assert _tone_power(voice, 1000, COMPOSITE_FS) > 1e6 * _tone_power(
        voice, 3000, COMPOSITE_FS
    )


def test_aetv_translation_occupies_upper_slice_and_round_trips():
    time = np.arange(8000) / 8000
    native = np.sin(2 * np.pi * 450 * time) + np.sin(2 * np.pi * 2650 * time)
    shifted = translate_aetv_up(native)
    assert _tone_power(shifted, AETV_LOW_HZ, COMPOSITE_FS) > 1e6
    assert _tone_power(shifted, AETV_HIGH_HZ, COMPOSITE_FS) > 1e6
    recovered = extract_aetv(np.concatenate((shifted, shifted)))[:8000]
    assert np.corrcoef(native, recovered)[0, 1] > 0.999


def test_composite_delays_voice_by_exactly_one_gop():
    upper = [np.sin(2 * np.pi * 800 * np.arange(8000) / 8000)] * 2
    voice = [np.full(8000, 0.25), np.full(8000, -0.25)]
    composite, _, _ = compose_delayed_stream(upper, voice)
    recovered = extract_voice(composite)
    # Ignore the resampler's startup transient; the interior pre-delay interval
    # must remain silent.
    assert np.max(np.abs(recovered[200:7800])) < 1e-3
    assert recovered[8500:15500].mean() > 0.05
    assert recovered[16500:23500].mean() < -0.05
