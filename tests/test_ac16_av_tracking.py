"""Received-pilot audio correction and joint A/V playout regressions."""
from types import SimpleNamespace

import numpy as np

from aetv.analog_av import AC16ProgramAudio
from aetv.av_playout import PairedAVPlayout


def observation(start, frequency, coherence=1):
    return SimpleNamespace(stream_start_sample=round(start*48000),
                           freq_offset=frequency, pilot_coherence=coherence)


def test_drifting_modem_and_blind_acquisition_keep_audio_pitch_and_source_identity():
    from aetv.ac16_av_smoke import av_tracking_smoke
    report = av_tracking_smoke(gops=26)
    assert report['passed']
    assert report['cases'][0]['paired_gops'] == 26
    assert report['cases'][1]['blind_acquired']
    assert all(case['last_source_gop'] == 25 for case in report['cases'])


def test_audio_correction_carries_phase_across_gops_and_the_voice_only_tail():
    program = AC16ProgramAudio()
    fs = 8000
    t = np.arange(9*fs)/fs
    tone, initial, slope = 997.25, 13.2, .6
    program.voice.write(np.cos(2*np.pi*((tone+initial)*t + .5*slope*t*t)))
    # Deliver observations incrementally, as on air, with the final voice GOP
    # arriving after the last video pilot. Weak/invalid estimates cannot steer.
    output = []
    for i in range(8):
        program.add(observation(i, initial + slope*(i+.4525)), i)
        if i:
            # Only allow the previous audio second to become ready.
            output.append(next(program.ready())[1])
    output.extend(audio for _,audio in program.ready())
    values = np.concatenate(output)
    times = np.arange(len(values))/fs + 1
    design = np.column_stack((np.cos(2*np.pi*tone*times), np.sin(2*np.pi*tone*times)))
    fit = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    # Reject phase resets at the boundaries, not merely correct FFT peaks.
    assert np.sqrt(np.mean((values[400:-400]-fit[400:-400])**2)) < .015
    assert max(abs(values[k*fs]-fit[k*fs]) for k in range(1,8)) < .03


def test_audio_frequency_holds_after_lost_pilots_and_ignores_weak_or_nan_estimates():
    program = AC16ProgramAudio()
    program.add(observation(0, 12), 0)
    program.add(observation(1, 12.6), 1)
    expected = program._frequency_at(np.array([1,2,3,100]))
    program.add(observation(2, 1000, .1), 2)
    program.add(observation(3, float('nan')), 3)
    np.testing.assert_array_equal(program._frequency_at(np.array([1,2,3,100])), expected)
    assert expected[-1] < 14  # no unlimited extrapolation through noise/silence


def test_small_received_boundary_adjustments_do_not_reset_audio_correction_phase():
    program = AC16ProgramAudio()
    fs, tone, cfo = 8000, 811.5, 12.3
    t = np.arange(5*fs)/fs
    program.voice.write(np.cos(2*np.pi*(tone+cfo)*t))
    for i in range(3):
        program.add(observation(i+i/fs, cfo), i)
    blocks = list(program.ready())
    values = np.concatenate([audio[320:-320] for _,audio in blocks])
    times = np.concatenate([1+i+i/fs + np.arange(320, fs-320)/fs for i,_ in blocks])
    design = np.column_stack((np.cos(2*np.pi*tone*times), np.sin(2*np.pi*tone*times)))
    fit = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    assert np.sqrt(np.mean((values-fit)**2)) < .001


def test_paired_playout_bounds_latency_without_bursting_after_a_stall_or_reset():
    queue = PairedAVPlayout()
    for i in range(8):
        queue.push((i, i))
    assert queue.dropped == 4
    assert queue.pop(10) == (4,4)
    assert queue.pop(10.99) is None
    assert queue.pop(11) == (5,5)
    assert queue.pop(30) == (6,6)
    assert queue.pop(30) is None
    queue.clear()
    queue.push((9,9))
    assert queue.pop(30.5) is None  # already released audio still occupies a second
    assert queue.pop(31) == (9,9)
    assert queue.pop(100) is None
    queue.push((10,10))
    assert queue.pop(100) == (10,10)


def test_receive_burst_drops_whole_pairs_and_preserves_all_recorded_media():
    from aetv.config import AETV_MODES
    from aetv.settings import StationSettings
    from aetv.station import RxEngine, Station

    station = Station(StationSettings(mode='AC16', waveform_mode='analog_av'))
    station.codec = SimpleNamespace(mode=AETV_MODES['AC16'])
    audio, video = [], []
    engine = RxEngine(station, on_video=lambda frames,state: video.append(int(frames[0,0,0,0])))
    engine._audio_playback = SimpleNamespace(write=lambda values, on_start: (audio.append(int(values[0])), on_start(), True)[-1])
    now = [100.0]
    engine._av_playout = PairedAVPlayout(clock=lambda: now[0])
    for i in range(8):
        result = SimpleNamespace(freq_offset=0, callsign='TEST', sync_metric=1, snr_db=20,
                                 pilot_evm_pct=0, pilot_timing_ppm=0)
        engine._deliver_received(result, np.full((10,1,1,3), i, np.uint8), np.full(8000,i))
    assert len(engine.last_video) == 80 and len(engine.last_audio) == 64000
    assert not audio and not video
    for _ in range(4):
        engine._drain_av_playout()
        engine._drain_av_playout()  # polling twice cannot release a second pair
        now[0] += 1
    assert audio == video == [4,5,6,7]
    assert [int(v) for v in engine.last_audio[::8000]] == list(range(8))
