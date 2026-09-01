import numpy as np

from aetv.audio_io import StreamResampler, resample_ratio
from aetv.analog_av import (
    AETV_HIGH_HZ,
    AETV_LOW_HZ,
    COMPOSITE_FS,
    compose_delayed_stream,
    extract_aetv,
    extract_voice,
    mix_composite_chunk,
    StreamingCompositeSeparator,
    prepare_voice,
    translate_aetv_up,
)
from aetv.config import AETV_MODES
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.settings import StationSettings
from aetv.station import Station, TxEngine


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


def test_live_composite_power_control_sets_branch_power_ratio():
    time = np.arange(8000) / 8000
    video = np.sin(2 * np.pi * 900 * time)
    voice = np.sin(2 * np.pi * 1200 * time)
    composite = mix_composite_chunk(video, voice, video_power=0.8)
    spectrum = np.abs(np.fft.rfft(composite)) ** 2
    frequency = np.fft.rfftfreq(len(composite), 1.0 / COMPOSITE_FS)
    voice_power = spectrum[(frequency > 1100) & (frequency < 1300)].sum()
    video_power = spectrum[(frequency > 2950) & (frequency < 3150)].sum()
    assert 3.5 < video_power / voice_power < 4.5


def test_production_streaming_separator_recovers_clean_v8_composite():
    mode = AETV_MODES["V8"]
    rng = np.random.default_rng(17)
    latents = [
        rng.normal(0.0, 0.2, mode.latents_per_gop).astype(np.float32)
        for _ in range(4)
    ]
    engine = TxEngine(Station(StationSettings(
        mode="V8", waveform_mode="analog_av", av_microphone_mix=0.0,
        av_video_power=0.7,
    )))
    composite = engine._composite_chunks(
        modulate_continuous_chunks(latents, "V8"),
        np.zeros(4 * 8000, dtype=np.float32),
        4,
    )
    separator = StreamingCompositeSeparator()
    resampler = StreamResampler(*resample_ratio(12000, 8000))
    demodulator = StreamingDemodulator("W", continuous=True, mode_name="V8")
    decoded = []
    for chunk in composite:
        for start in range(0, len(chunk), 1200):
            _voice, native_12k = separator.process(chunk[start : start + 1200])
            native = resampler(native_12k)
            if native.size:
                decoded.extend(demodulator.feed(native))
    assert sum(len(result.gops_latents) for result in decoded) == 4
