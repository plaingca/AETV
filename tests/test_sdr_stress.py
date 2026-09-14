"""Long-run scheduling, weak RF setup, and bounded endpoint failure checks."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aetv.config import AETV_MODES
from aetv.av_playout import PairedAVPlayout


def test_paired_clock_does_not_accumulate_polling_jitter_over_ten_minutes():
    playout = PairedAVPlayout()
    next_gop = 0
    shown = []
    # Ordinary 100 ms worker polling with scheduler jitter and occasional
    # missed polls. A relative now+1 deadline loses dozens of GOPs here.
    for step in range(6050):
        now = step / 10 + .002 * (1 + np.sin(step))
        while next_gop < min(600, int(now) + 1):
            playout.push(next_gop)
            next_gop += 1
        if step % 37 == 9:
            continue
        item = playout.pop(now)
        if item is not None:
            shown.append(item)
    assert shown == list(range(600))
    assert playout.dropped == 0


@pytest.mark.parametrize("name,av", [("V8", False), ("V7", False), ("V8", True)])
@pytest.mark.parametrize("snr", [0., 35.])
@pytest.mark.parametrize("seed", [8473, 836])
def test_non_ac16_coarse_tuning_finds_offset_video_without_source_timing(name, av, snr, seed):
    from aetv.modem import modulate_continuous_chunks
    from aetv.sdr_dsp import ModemToIQ, estimate_mode_signal_offset
    from aetv.audio_io import StreamResampler, resample_ratio
    from aetv.analog_av import waveform_center_hz, waveform_sample_rate
    from aetv.settings import StationSettings
    from aetv.station import Station, TxEngine

    rng = np.random.default_rng(seed)
    mode = AETV_MODES[name]
    settings = StationSettings(mode=name, waveform_mode="analog_av" if av else "video")
    latents = rng.normal(size=(3, mode.latents_per_gop)).astype(np.float32)
    chunks = modulate_continuous_chunks(latents, name, total_gops=3)
    if av:
        voice = .2*np.sin(2*np.pi*731*np.arange(24000)/8000)
        chunks = TxEngine(Station(settings))._composite_chunks(chunks, voice, 3, capture_microphone=False)
    audio = StreamResampler(*resample_ratio(waveform_sample_rate(settings), 48000))(np.concatenate(list(chunks)))
    iq = ModemToIQ(960000, -105150, waveform_center_hz(settings), peak_limit=.9).feed(audio*.1)[960000:1920000]
    power = np.mean(abs(iq)**2) * (.5 if av else 1) * (960000/(50*mode.geometry.carriers)) / 10**(snr/10)
    iq += np.sqrt(power/2)*(rng.normal(size=len(iq))+1j*rng.normal(size=len(iq)))
    result = estimate_mode_signal_offset(iq, mode, composite=av)
    # Merely being inside the modem's +/-600 Hz search is insufficient: a
    # narrow A/V receive filter can lose its edge beacon before its pilots.
    assert abs(result["offset_hz"] + 105150) < 150, result


@pytest.mark.parametrize("name", ["V8", "V7"])
def test_coarse_tuning_rejects_noise_or_one_tone(name):
    from aetv.sdr_dsp import estimate_mode_signal_offset
    rng = np.random.default_rng(836)
    iq = rng.normal(size=960000)+1j*rng.normal(size=960000)
    for tone in [False, True]:
        values = iq + (30*np.exp(-2j*np.pi*105000*np.arange(len(iq))/960000) if tone else 0)
        with pytest.raises(ValueError):
            estimate_mode_signal_offset(values, AETV_MODES[name])


def test_v8_program_audio_pairs_late_gops_at_native_rate_and_corrects_pitch():
    from aetv.analog_av import ProgramAudio
    fs, frequency, offset = 8000, 811.5, 14.25
    program = ProgramAudio("V8")
    t = np.arange(8*fs)/fs
    program.voice.write(np.cos(2*np.pi*(frequency+offset)*t))
    for index in (3, 4, 6):
        result = SimpleNamespace(stream_start_sample=index*fs, freq_offset=offset, pilot_coherence=1)
        program.add(result, index)
    blocks = list(program.ready())
    assert [index for index, _ in blocks] == [3, 4, 6]
    values = np.concatenate([audio[320:-320] for _, audio in blocks])
    times = np.concatenate([index+1+np.arange(320,fs-320)/fs for index,_ in blocks])
    design = np.column_stack((np.cos(2*np.pi*frequency*times),np.sin(2*np.pi*frequency*times)))
    fit = design@np.linalg.lstsq(design,values,rcond=None)[0]
    assert np.sqrt(np.mean((values-fit)**2)) < .002


def test_stalled_audio_endpoint_never_blocks_receive_or_grows_beyond_two_seconds(monkeypatch):
    from aetv import audio_io
    playing, release = threading.Event(), threading.Event()

    def player(chunks, *_args, **_kwargs):
        for block in chunks:
            if len(block) != 8000:
                continue
            playing.set()
            release.wait(2)

    monkeypatch.setattr(audio_io, "play_chunk_stream", player)
    output = audio_io.AudioPlaybackStream(8000)
    try:
        assert output.write(np.zeros(8000))
        assert playing.wait(1)
        assert output.write(np.zeros(8000))
        assert output.write(np.zeros(8000))
        assert output.write(np.zeros(8000)) is False
        assert output.health["queue_high_water_samples"] == 16000
        assert output.health["dropped_blocks"] == 1
    finally:
        release.set()
        output.close()
    assert not output._thread.is_alive()


def test_iq_overflow_retains_recent_samples_and_marks_the_clock_gap():
    from aetv.sdr import SDRCapture
    from aetv.settings import StationSettings
    messages = []
    cap = SDRCapture(StationSettings(mode="V8", rx_source="pluto"), AETV_MODES["V8"],
                     SimpleNamespace(write=lambda _: None), on_error=pytest.fail, on_status=messages.append)
    count = [0]

    def receive():
        count[0] += 1
        if count[0] == 83:
            cap._stop.set()
        return np.ones(100, np.complex64)

    cap._radio = SimpleNamespace(rx=receive)
    cap._capture()
    assert cap.health["iq_overruns"] == 1
    assert cap.health["iq_discarded_samples"] == 8000
    assert cap._queue.get_nowait() is cap._gap
    assert cap._queue.qsize() == 3
    assert messages and "reacquiring" in messages[0]


@pytest.mark.parametrize("name", ["V0", "V7", "V8", "AC16"])
@pytest.mark.parametrize("jump", [-1, 1])
@pytest.mark.parametrize("duration", [.010625, .135625])
def test_plausible_pilots_after_transport_timing_slip_do_not_leave_corrupt_tracking(name, jump, duration):
    from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
    mode = AETV_MODES[name]
    fs = mode.geometry.fs
    rng = np.random.default_rng(8084)
    sent = rng.normal(size=(30, mode.latents_per_gop)).astype(np.float32)
    waveform = np.concatenate(list(modulate_continuous_chunks(sent, name)))
    cut, samples = round(6.237 * fs), round(duration * fs)
    waveform = (np.r_[waveform[:cut], np.zeros(samples), waveform[cut:]]
                if jump > 0 else np.r_[waveform[:cut], waveform[cut + samples:]])
    demod = StreamingDemodulator(mode.band, continuous=True, mode_name=name,
                                 boundary_tracking=True)
    received = []
    for start in range(0, len(waveform), fs // 10):
        received.extend(r.gops_latents[0] for r in demod.feed(waveform[start:start + fs // 10]))
    z = np.asarray(received)
    scores = (z / np.linalg.norm(z, axis=1)[:, None]) @ (sent / np.linalg.norm(sent, axis=1)[:, None]).T
    # The interrupted GOP can be lost; every final GOP must recover with the
    # correct source identity, not merely a plausible pilot-presence score.
    assert scores[-6:].argmax(axis=1).tolist() == list(range(24, 30))
    assert np.min(scores[-6:].max(axis=1)) > .90


def test_slow_debug_disk_keeps_valid_prefix_without_blocking_receiver():
    from aetv.recording import BufferedWriter
    started, release = threading.Event(), threading.Event()
    saved = []

    def write(data):
        started.set()
        release.wait(2)
        saved.append(data)

    writer = BufferedWriter(write, lambda: None, max_bytes=4)
    try:
        assert writer.write(b"first") is False  # explicit byte budget
    finally:
        release.set()
        writer.close()
    assert writer.health["error"] and not saved
    release.clear()
    writer = BufferedWriter(write, lambda: None, max_bytes=6)
    try:
        assert writer.write(b"aa")
        assert started.wait(1)
        assert writer.write(b"bbb")
        assert writer.write(b"ccc")
        assert writer.write(b"d") is False
        assert writer.write(b"e") is False
    finally:
        release.set()
        writer.close()
    assert saved == [b"aa", b"bbb", b"ccc"]
    assert writer.health["rejected_bytes"] == 2
    assert not writer._thread.is_alive()


def test_video_callback_follows_its_audio_block_at_output_worker(monkeypatch):
    from aetv import audio_io
    active, release, second = threading.Event(), threading.Event(), threading.Event()
    shown = []

    def player(chunks, *_args, **_kwargs):
        for block in chunks:
            if len(block) != 8000:
                continue
            if not active.is_set():
                active.set()
                release.wait(2)
            else:
                second.set()

    monkeypatch.setattr(audio_io, "play_chunk_stream", player)
    output = audio_io.AudioPlaybackStream(8000)
    try:
        assert output.write(np.zeros(8000), on_start=lambda: shown.append(1))
        assert active.wait(1)
        assert output.write(np.zeros(8000), on_start=lambda: shown.append(2))
        assert shown == [1]
        release.set()
        assert second.wait(1)
        assert shown == [1, 2]
    finally:
        release.set()
        output.close()
