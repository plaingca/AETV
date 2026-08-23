"""Ham-station helpers: ring buffer, resample, Kiwi IQ mix, CAT, fail-safe PTT."""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace
import wave
from pathlib import Path

import numpy as np
import pytest

from aetv.audio_io import StreamResampler, resample_ratio
from aetv.cat import CatConfig, NullPtt, RigctldClient, open_ptt
from aetv.kiwi import IqToPassband, iq_to_passband, kiwi_center_khz
from aetv.ringbuffer import RingBuffer
from aetv.settings import StationSettings, load_settings, normalize_callsign, save_settings
from aetv.station import (
    RxState,
    Station,
    TxEngine,
    TxPhase,
    WATCHDOG_MARGIN_S,
    _PcmWaveRecorder,
    _PttWatchdog,
)


def test_ringbuffer_wrap_and_tail():
    ring = RingBuffer(seconds=1.0, fs=100)
    ring.write(np.arange(150, dtype=np.float64))
    snap, total = ring.snapshot()
    assert total == 150
    assert len(snap) == 100
    assert snap[0] == 50
    assert snap[-1] == 149
    tail = ring.tail(10)
    assert np.array_equal(tail, np.arange(140, 150))


def test_ringbuffer_incremental_reader_reports_overrun():
    ring = RingBuffer(seconds=1, fs=5)
    ring.write(np.array([1.0, 2.0, 3.0]))
    first, cursor, overrun = ring.read_since(0)
    assert np.array_equal(first, [1.0, 2.0, 3.0])
    assert not overrun
    ring.write(np.array([4.0, 5.0, 6.0, 7.0]))
    fresh, cursor, overrun = ring.read_since(cursor)
    assert np.array_equal(fresh, [4.0, 5.0, 6.0, 7.0])
    old, _, overrun = ring.read_since(0)
    assert overrun
    assert np.array_equal(old, [3.0, 4.0, 5.0, 6.0, 7.0])


def test_stream_resampler_matches_oneshot():
    from scipy.signal import resample_poly

    rng = np.random.default_rng(0)
    src = rng.standard_normal(8000).astype(np.float64)
    up, down = resample_ratio(8000, 24000)
    one = resample_poly(src, up, down)
    stream = StreamResampler(up, down)
    parts = [stream(src[i : i + 256]) for i in range(0, len(src), 256)]
    got = np.concatenate(parts) if any(len(p) for p in parts) else np.zeros(0)
    n = min(len(one), len(got))
    assert n > 1000
    err = np.max(np.abs(one[:n] - got[:n]))
    assert err < 1e-9


def test_kiwi_center_and_passband_tone():
    dial_mhz = 7.088
    fcenter = 5000.0
    center = kiwi_center_khz(dial_mhz, fcenter)
    assert center == pytest.approx(7093.0)
    src_rate, dst_rate = 12000, 24000
    tone_hz = 3000.0
    t = np.arange(src_rate) / src_rate
    iq_hz = tone_hz - fcenter
    iq = np.exp(2j * np.pi * iq_hz * t)
    audio, _phase = iq_to_passband(iq, src_rate, dst_rate, offset_hz=dial_mhz * 1e6 - center * 1e3)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / dst_rate)
    peak = freqs[int(np.argmax(spec))]
    assert peak == pytest.approx(tone_hz, abs=40.0)


def test_iq_stream_phase_continuity():
    src_rate, dst_rate = 12000, 24000
    offset = -5000.0
    t = np.arange(2400) / src_rate
    iq = np.exp(2j * np.pi * (-2000.0) * t)
    whole, _ = iq_to_passband(iq, src_rate, dst_rate, offset)
    conv = IqToPassband(src_rate, dst_rate, offset)
    parts = [conv(iq[i : i + 300]) for i in range(0, len(iq), 300)]
    got = np.concatenate(parts)
    n = min(len(whole), len(got))
    # Streaming FIR has latency; compare the overlapping body.
    body = slice(200, n - 200)
    corr = np.corrcoef(whole[body], got[body])[0, 1]
    assert corr > 0.99


def test_settings_roundtrip(tmp_path: Path):
    settings = StationSettings(
        callsign="va7eet", mode="V7", kiwi_host="1.2.3.4:8073",
        tx_channel_profile="mpp6",
    )
    path = tmp_path / "settings.json"
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.callsign == "va7eet"
    assert loaded.mode == "V7"
    assert loaded.kiwi_host == "1.2.3.4:8073"
    assert loaded.tx_channel_profile == "mpp6"


def test_settings_reject_unknown_channel_profile():
    settings = StationSettings(tx_channel_profile="invented")
    assert "unknown TX channel profile 'invented'" in settings.validate()


def test_native_flex_settings_do_not_silently_use_soundcard(tmp_path: Path):
    settings = StationSettings(
        cat_backend="flex", flex_host="192.0.2.1", flex_native_audio=True,
        rx_source="soundcard",
    )
    path = tmp_path / "settings.json"
    save_settings(settings, path)
    assert load_settings(path).rx_source == "flex"


def test_normalize_callsign():
    assert normalize_callsign("va7eet") == "VA7EET"
    assert normalize_callsign("w1aw/7 extra") == "W1AW/7"
    assert len(normalize_callsign("ABCDEFGHIJK")) == 8


def test_open_ptt_none():
    ptt = open_ptt(CatConfig(backend="none"))
    assert isinstance(ptt, NullPtt)
    ptt.set_ptt(True)
    ptt.set_ptt(False)


class _FakeRigctld(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.ptt = []
        self.ready = threading.Event()

    def run(self):
        self.ready.set()
        conn, _addr = self.sock.accept()
        with conn:
            buf = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode("ascii").strip()
                    if cmd.startswith("T "):
                        self.ptt.append(cmd.split()[1] == "1")
                        conn.sendall(b"RPRT 0\n")
                    elif cmd == "f":
                        conn.sendall(b"7088000\n")
                    elif cmd == "m":
                        conn.sendall(b"DIGU\n")
                    else:
                        conn.sendall(b"RPRT 0\n")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def test_rigctld_ptt_and_freq():
    server = _FakeRigctld()
    server.start()
    server.ready.wait(1.0)
    client = RigctldClient("127.0.0.1", server.port, timeout=1.0)
    try:
        assert client.get_frequency_hz() == 7088000.0
        assert client.get_mode() == "DIGU"
        client.set_ptt(True)
        client.set_ptt(False)
        time.sleep(0.05)
        assert server.ptt == [True, False]
    finally:
        client.close()
        server.close()


class _FakePtt:
    def __init__(self):
        self.events = []

    def set_ptt(self, on: bool) -> None:
        self.events.append(bool(on))

    def describe(self) -> str:
        return "fake"

    def close(self) -> None:
        return None


def test_tx_ptt_brackets_playback_and_unkeys_on_error():
    from aetv.settings import StationSettings
    from aetv.station import Station

    station = Station(StationSettings(audio_only=False, cat_backend="none"))
    ptt = _FakePtt()

    def player(wave, fs, device=None, should_stop=None, on_progress=None):
        if on_progress:
            on_progress(1.0)
        return True

    engine = TxEngine(station, ptt=ptt, player=player)
    audio = np.zeros(2400, dtype=np.float32)
    assert engine._keyed_send(audio, 24000) is True
    assert ptt.events == [True, False]
    assert engine.state.phase == TxPhase.DONE


def test_tx_ptt_unkeys_when_player_raises():
    from aetv.settings import StationSettings
    from aetv.station import Station

    station = Station(StationSettings())
    ptt = _FakePtt()

    def player(*_args, **_kwargs):
        raise RuntimeError("soundcard gone")

    engine = TxEngine(station, ptt=ptt, player=player)
    engine._keyed_send(np.zeros(100, dtype=np.float32), 24000)
    assert ptt.events[0] is True
    assert ptt.events[-1] is False
    assert engine.state.phase == TxPhase.FAILED


def test_streaming_tx_prepares_first_gop_before_ptt(monkeypatch):
    from aetv.station import Station

    events = []

    class Ptt:
        def set_ptt(self, on):
            events.append(f"ptt:{on}")

        def describe(self):
            return "test PTT"

        def close(self):
            events.append("close")

    def chunks():
        for index in range(2):
            events.append(f"produce:{index}")
            yield np.zeros(32, dtype=np.float32)

    def play(stream, rate, **kwargs):
        for index, _chunk in enumerate(stream):
            events.append(f"play:{index}")
            kwargs["on_chunk"](index + 1)
        return True

    monkeypatch.setattr("aetv.station.play_chunk_stream", play)
    station = Station(StationSettings(ptt_lead_s=0, ptt_tail_s=0))
    engine = TxEngine(station, ptt=Ptt())
    assert engine._keyed_send_stream(chunks(), 24000, 2)
    assert events.index("produce:0") < events.index("ptt:True") < events.index("play:0")


def test_webcam_gops_are_captured_and_encoded_lazily(monkeypatch):
    from aetv.station import Station

    events = []

    def camera(_mode, camera=0, duration_s=None):
        assert duration_s is None
        for index in range(6):
            events.append(f"frame:{index}")
            yield np.full((2, 3, 3), index, dtype=np.uint8)

    class Codec:
        mode = SimpleNamespace(gop_frames=2)

        def encode_gop(self, frames):
            events.append(f"encode:{int(frames[0, 0, 0, 0])}")
            return np.array([frames.mean()], dtype=np.float32)

    monkeypatch.setattr("aetv.station.iter_webcam", camera)
    station = Station(StationSettings(debug_capture=False))
    engine = TxEngine(station)
    stream = engine._live_webcam_gops(Codec(), 3)
    next(stream)
    assert events == ["frame:0", "frame:1", "encode:0"]
    next(stream)
    assert events[-3:] == ["frame:2", "frame:3", "encode:2"]
    stream.close()


def test_stream_producer_does_not_encode_past_available_buffer(monkeypatch):
    """TX startup must not eagerly encode several GOPs before playback."""
    encoded = []
    first_chunk_seen = threading.Event()
    allow_playback = threading.Event()

    class Ptt:
        def set_ptt(self, _on):
            pass

        def describe(self):
            return "test PTT"

        def close(self):
            pass

    def chunks():
        for index in range(5):
            encoded.append(index)
            yield np.zeros(32, dtype=np.float32)

    def play(stream, rate, **kwargs):
        iterator = iter(stream)
        next(iterator)
        first_chunk_seen.set()
        assert allow_playback.wait(timeout=2.0)
        for index, _chunk in enumerate(iterator, start=2):
            kwargs["on_chunk"](index)
        return True

    monkeypatch.setattr("aetv.station.play_chunk_stream", play)
    station = Station(StationSettings(ptt_lead_s=0, ptt_tail_s=0))
    engine = TxEngine(station, ptt=Ptt())
    result = []
    worker = threading.Thread(
        target=lambda: result.append(engine._keyed_send_stream(chunks(), 24000, 5))
    )
    worker.start()
    assert first_chunk_seen.wait(timeout=2.0)
    # One GOP is playing and one may be encoded ahead; the producer must not
    # run through the rest merely because transmission startup is still busy.
    time.sleep(0.05)
    assert encoded == [0, 1]
    allow_playback.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert result == [True]


def test_debug_wave_recorder_streams_pcm_to_disk(tmp_path):
    path = tmp_path / "tx.tx.wav"
    recorder = _PcmWaveRecorder(path, 24000)
    recorder.write(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    recorder.close()
    with wave.open(str(path), "rb") as saved:
        assert saved.getframerate() == 24000
        assert saved.getnchannels() == 1
        assert saved.getnframes() == 3


def test_channel_loopback_decodes_without_keying(monkeypatch):
    from aetv.station import Station

    result = SimpleNamespace(
        gops_latents=[np.array([1.0])],
        gops_weights=[np.array([1.0])],
        freq_offset=0.0,
        sync_metric=0.9,
        snr_db=6.2,
        callsign="N0CALL",
    )
    class FakeChannel:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, audio):
            return audio

    monkeypatch.setattr("aetv.station.StreamingChannelEmulator", FakeChannel)

    class FakeDemodulator:
        def __init__(self, *args, **kwargs):
            pass

        def feed(self, _audio):
            return [result]

    monkeypatch.setattr("aetv.station.StreamingDemodulator", FakeDemodulator)

    class Codec:
        mode = SimpleNamespace(name="V7", band="U", gop_frames=2)

        def decode_gop(self, latents, _weights):
            return np.full((2, 2, 2, 3), latents[0], dtype=np.uint8)

    shown = []
    def chunks():
        yield np.ones(8, dtype=np.float32)
        assert len(shown) == 1  # First GOP is decoded before TX requests the next.
        yield np.ones(8, dtype=np.float32)

    station = Station(StationSettings(tx_channel_profile="awgn6", tx_level=0.7))
    engine = TxEngine(station, on_loopback=lambda video, state: shown.append((video, state)))
    assert engine._emulated_send_stream(
        chunks(),
        24000,
        2,
        Codec(),
        "awgn6",
    )
    assert len(shown) == 2
    assert all(isinstance(state, RxState) and state.source == "emulator" for _, state in shown)
    assert engine.state.phase == TxPhase.DONE


def test_ptt_watchdog_fires(monkeypatch):
    ptt = _FakePtt()
    fired = []
    dog = _PttWatchdog(ptt, timeout_s=0.05, on_fire=lambda: fired.append(True))
    dog.start()
    time.sleep(0.2)
    assert ptt.events == [False]
    assert fired == [True]
    dog.cancel()
    assert WATCHDOG_MARGIN_S == 15.0
