"""Soundcard rate selection and Windows process-isolation behavior."""

from io import BytesIO
import struct
from types import SimpleNamespace

import numpy as np

import aetv.audio_io as audio_io


def test_wasapi_inventory_uses_stable_endpoint_ids(monkeypatch):
    default = SimpleNamespace(id="speaker-2")
    speakers = [
        SimpleNamespace(id="speaker-1", name="Radio DAX", channels=2),
        SimpleNamespace(id="speaker-2", name="Desk speakers", channels=2),
    ]
    sc = SimpleNamespace(
        all_speakers=lambda: speakers,
        default_speaker=lambda: default,
    )
    monkeypatch.setattr(audio_io, "_sc", lambda: sc)

    devices = audio_io._list_wasapi_devices_direct("output")

    assert [item.name for item in devices] == ["Radio DAX", "Desk speakers"]
    assert devices[1].is_default
    assert devices[0].selection_value() == "wasapi:speaker-1"


def test_wasapi_device_accepts_prefixed_endpoint_id(monkeypatch):
    selected = object()
    calls = []
    sc = SimpleNamespace(
        get_microphone=lambda value, include_loopback: calls.append(
            (value, include_loopback)
        )
        or selected,
    )
    monkeypatch.setattr(audio_io, "_sc", lambda: sc)

    assert audio_io._wasapi_device("wasapi:mic-guid", "input") is selected
    assert calls == [("mic-guid", False)]


def test_default_output_uses_native_device_rate(monkeypatch):
    opened = {}

    class Stream:
        def __init__(self, **kwargs):
            opened.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, samples):
            opened["samples"] = samples

    sd = SimpleNamespace(
        query_devices=lambda device, kind: {
            "default_samplerate": 48000,
        },
        OutputStream=Stream,
    )
    monkeypatch.setattr(audio_io, "_sd", lambda: sd)

    assert audio_io._play_chunk_stream_direct(
        [np.ones(8000, dtype=np.float32)], 8000, device=None
    )

    assert opened["samplerate"] == 48000
    assert opened["samples"].shape[1] == 1


def test_default_input_uses_native_device_rate(monkeypatch):
    opened = {}

    class Stream:
        def __init__(self, **kwargs):
            opened.update(kwargs)

        def start(self):
            opened["started"] = True

    sd = SimpleNamespace(
        query_devices=lambda device, kind: {"default_samplerate": 48000},
        InputStream=Stream,
    )
    monkeypatch.setattr(audio_io, "_sd", lambda: sd)
    ring = SimpleNamespace(write=lambda _samples: None)
    gaps = []

    _stream, native = audio_io._open_input_stream_direct(
        None, ring, 8000, on_discontinuity=lambda: gaps.append(True)
    )
    opened["callback"](
        np.zeros((32, 1), dtype=np.float32), 32, None, "input overflow"
    )

    assert native == 48000
    assert opened["samplerate"] == 48000
    assert opened["started"]
    assert gaps == [True]


def test_windows_heap_corruption_is_reported_as_audio_error():
    proc = SimpleNamespace(
        poll=lambda: -1073740940,
        wait=lambda timeout: -1073740940,
        stderr=BytesIO(),
    )

    error = audio_io._worker_error(proc, "input")

    assert "heap corruption" in str(error)
    assert "soundcard input helper stopped" in str(error)


def test_windows_chunk_playback_dispatches_to_isolated_worker(monkeypatch):
    called = []
    monkeypatch.setattr(audio_io.os, "name", "nt")
    monkeypatch.delenv("AETV_AUDIO_WORKER_CHILD", raising=False)
    monkeypatch.setattr(
        audio_io,
        "_play_chunk_stream_isolated",
        lambda *args, **kwargs: called.append((args, kwargs)) or True,
    )

    assert audio_io.play_chunk_stream([np.zeros(4)], 8000)
    assert len(called) == 1


def test_isolated_output_protocol_waits_for_each_played_chunk(monkeypatch):
    class WritablePipe(BytesIO):
        def close(self):
            pass

    class Proc:
        def __init__(self):
            self.stdin = WritablePipe()
            self.stdout = BytesIO(b"RA")
            self.stderr = BytesIO()
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -1

        def kill(self):
            self.returncode = -1

    proc = Proc()
    progress = []
    monkeypatch.setattr(audio_io, "_start_audio_worker", lambda *_args: proc)
    samples = np.array([0.25, -0.5], dtype=np.float32)

    assert audio_io._play_chunk_stream_isolated(
        [samples], 8000, on_chunk=progress.append
    )

    payload = proc.stdin.getvalue()
    assert struct.unpack("<I", payload[:4])[0] == samples.nbytes
    assert np.array_equal(np.frombuffer(payload[4:], dtype="<f4"), samples)
    assert progress == [1]
