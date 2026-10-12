"""Offline callback/ring/HF path regression; no audio device or Torch required."""
import numpy as np
import pytest
from scipy.signal import resample_poly

from aetv.audio_io import StreamResampler
from aetv.audio_validation import AudioPathScenario, virtual_audio_path


@pytest.mark.parametrize("up,down", [(6, 1), (1, 6), (441, 80), (80, 441), (1, 1)])
@pytest.mark.parametrize("length", [0, 1, 19, 127, 4097])
def test_finite_stream_resampler_flush_is_exact(up, down, length):
    source = np.random.default_rng(3).normal(size=length)
    worker = StreamResampler(up, down)
    pieces = [worker(source[start:start + 37]) for start in range(0, length, 37)]
    pieces.append(worker.flush())
    actual = np.concatenate(pieces)
    expected = resample_poly(source, up, down) if length else np.zeros(0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert worker.flush().size == 0
    with pytest.raises(RuntimeError, match="flushed"):
        worker(np.ones(1))


def test_actual_audio_callbacks_survive_native_rate_mismatch():
    rate = 8000
    source = (.3 * np.sin(2 * np.pi * 1100 * np.arange(rate) / rate)).astype(np.float32)
    actual, health = virtual_audio_path([source[:777], source[777:]], rate,
                                      AudioPathScenario("mixed", rx_rate=44100))
    assert np.corrcoef(source[40:-40], actual[40:rate - 40])[0, 1] > .99999
    assert health["output_samples"] == 48000
    assert health["callbacks"] > 100
    assert health["max_write_ms"] <= 20
    assert health["ring_overruns"] == 0
    assert health["input_stopped"] and health["input_closed"] and health["output_closed"]


def test_cancel_is_checked_inside_a_gop_and_aborts_endpoint():
    _, health = virtual_audio_path([np.ones(8000, np.float32)], 8000,
                                  AudioPathScenario("cancel", cancel_after_s=.35))
    assert not health["completed"]
    assert health["aborted"]
    assert .35 <= health["output_seconds"] <= .37
    assert health["input_closed"] and health["output_closed"]


def test_capture_overflow_and_slow_consumer_are_reported():
    _, health = virtual_audio_path([np.zeros(16000, np.float32)], 8000,
        AudioPathScenario("loss", capture_overflow_s=.3, consumer_stall_s=.8, ring_seconds=.1))
    assert health["capture_lost_samples"] == 960
    assert health["ring_overruns"] >= 1
    assert health["discontinuities"] >= 2
    assert "overflow" in health["errors"][0]


def test_hf_path_is_repeatable_and_exercises_clipping_dropout():
    source = np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000).astype(np.float32)
    scenario = AudioPathScenario("hf", snr_db=12, fading="mpp", gain=2,
                                dropout_start_s=.3, dropout_duration_s=.1)
    first, health = virtual_audio_path([source], 8000, scenario, seed=17)
    second, _ = virtual_audio_path([source], 8000, scenario, seed=17)
    np.testing.assert_array_equal(first, second)
    assert health["clipped_samples"] > 100
    assert health["dropped_rf_samples"] == 4800
    assert np.isfinite(first).all()


def test_output_underrun_stops_instead_of_silently_corrupting_waveform(monkeypatch):
    from types import SimpleNamespace
    from aetv import audio_io
    state = {}
    class Stream:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            state['closed'] = True
        def write(self, samples):
            return True
        def abort(self):
            state['aborted'] = True
    monkeypatch.setattr(audio_io, '_sd', lambda: SimpleNamespace(
        OutputStream=Stream, query_devices=lambda *_args: {'default_samplerate':8000}))
    with pytest.raises(audio_io.AudioUnavailable, match='underrun'):
        audio_io._play_chunk_stream_direct([np.zeros(8000)], 8000)
    assert state == {'aborted':True, 'closed':True}


@pytest.mark.parametrize('mode', ['V8', 'V9', 'V7', 'AC16'])
def test_complete_audio_path_recovers_clean_gops(mode):
    from scripts.validate_audio_path import run_trial
    row, _ = run_trial(mode, AudioPathScenario('mixed_native_rates', rx_rate=44100), gops=2)
    assert row['baseline_pass']


def test_receiver_recovers_after_fresh_transmission_following_gap():
    from aetv.analog_validation import unit_rms_latents
    from aetv.config import AETV_MODES
    from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
    mode = AETV_MODES['V8']
    originals = [unit_rms_latents('V8', 104 + n) for n in range(4)]
    # Two independent transmissions. A capture fault interrupts the first;
    # fresh preamble in the second must recover without restarting the app.
    chunks = []
    for group in (originals[:2], originals[2:]):
        for chunk in modulate_continuous_chunks(group, 'V8', 'N0CALL'):
            chunks.append(chunk * (.6 / max(float(np.max(np.abs(chunk))), 1e-12)))
        chunks.append(np.zeros(mode.geometry.fs // 2, np.float32))
    receiver = StreamingDemodulator(mode.band, mode_name='V8', continuous=True)
    recovered = []
    def receive(values):
        for result in receiver.feed(values):
            recovered.extend(result.gops_latents)
    def gap():
        nonlocal receiver
        receiver = StreamingDemodulator(mode.band, mode_name='V8', continuous=True)
    _, health = virtual_audio_path(chunks, mode.geometry.fs,
        AudioPathScenario('restart', capture_overflow_s=1.5), on_samples=receive, on_gap=gap)
    assert health['discontinuities'] == 1
    assert len(recovered) >= 2
    assert np.corrcoef(recovered[-2], originals[2])[0, 1] > .9
    assert np.corrcoef(recovered[-1], originals[3])[0, 1] > .9


@pytest.mark.parametrize('normal_return', [False, None])
def test_startup_underflow_is_not_mistaken_for_lost_waveform(monkeypatch, normal_return):
    from types import SimpleNamespace
    from aetv import audio_io
    writes = []
    class Stream:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def write(self, samples):
            writes.append(samples.copy())
            return True if len(writes) == 1 else normal_return
        def abort(self):
            pytest.fail('startup-only underflow must not abort')
    monkeypatch.setattr(audio_io, '_sd', lambda: SimpleNamespace(
        OutputStream=Stream, query_devices=lambda *_args: {'default_samplerate':8000}))
    assert audio_io._play_chunk_stream_direct([np.zeros(8000)], 8000)
    assert sum(len(block) for block in writes) == 8000
