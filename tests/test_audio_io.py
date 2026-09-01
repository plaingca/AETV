"""Soundcard rate selection and Windows process-isolation behavior."""

from io import BytesIO
import json
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


def test_wasapi_stereo_capture_is_downmixed_after_transport():
    captured = np.array(
        [[0.25, 0.75], [-0.5, 0.5], [1.0, 0.0]], dtype=np.float32
    )

    mono = audio_io._downmix_wasapi_capture(captured)

    assert mono.dtype == np.float32
    assert np.allclose(mono, [0.5, 0.0, 0.5])


def test_wasapi_live_blocks_are_twenty_milliseconds_or_longer():
    assert audio_io.wasapi_blocksize(8000) == 160
    assert audio_io.wasapi_blocksize(24000) == 480
    assert audio_io.wasapi_blocksize(4000) == 128


def test_wasapi_output_releases_only_initialized_frames():
    class Player:
        def __init__(self):
            self.available = iter([5, 10])
            self.buffers = []
            self.released = []

        def _render_available_frames(self):
            return next(self.available)

        def _render_buffer(self, count):
            buffer = bytearray()
            self.buffers.append((count, buffer))
            return [buffer]

        def _render_release(self, count):
            self.released.append(count)

    player = Player()

    def copy(buffer, payload, size):
        buffer.extend(payload[:size])

    audio_io._play_wasapi_exact(
        player, np.arange(7, dtype=np.float32), memmove=copy
    )

    assert player.released == [5, 2]
    assert [count for count, _buffer in player.buffers] == [5, 2]
    rendered = np.frombuffer(
        b"".join(bytes(buffer) for _count, buffer in player.buffers),
        dtype=np.float32,
    ).reshape(-1, 2)
    assert np.allclose(rendered[:, 0], np.arange(7))
    assert np.allclose(rendered[:, 1], np.arange(7))


def test_wasapi_stereo_output_preserves_iq_channels():
    class Player:
        def __init__(self):
            self.buffers = []

        def _render_available_frames(self):
            return 16

        def _render_buffer(self, count):
            buffer = bytearray()
            self.buffers.append(buffer)
            return [buffer]

        def _render_release(self, _count):
            pass

    player = Player()
    iq = np.array([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32)
    audio_io._play_wasapi_exact(
        player,
        iq,
        channels=2,
        memmove=lambda buffer, payload, size: buffer.extend(payload[:size]),
    )

    assert np.array_equal(
        np.frombuffer(b"".join(player.buffers), dtype=np.float32).reshape(-1, 2),
        iq,
    )


def test_streaming_hilbert_iq_is_continuous_and_suppresses_image():
    fs = 24000
    frequency = 3000
    count = 2 * fs
    source = np.cos(2 * np.pi * frequency * np.arange(count) / fs).astype(
        np.float32
    )
    whole = audio_io.StreamingHilbertIQ("iq_lr").process(source)
    split = audio_io.StreamingHilbertIQ("iq_lr")
    streamed = np.concatenate(
        [split.process(source[:731]), split.process(source[731:10003]), split.process(source[10003:])]
    )

    assert np.allclose(streamed, whole, rtol=1e-6, atol=1e-7)
    delay = split.delay
    assert np.array_equal(streamed[delay:, 0], source[:-delay])
    body = streamed[delay + 1024 : -1024]
    analytic = body[:, 0].astype(np.complex128) + 1j * body[:, 1]
    time_index = np.arange(delay + 1024, count - 1024)
    positive = abs(
        np.sum(analytic * np.exp(-2j * np.pi * frequency * time_index / fs))
    )
    negative = abs(
        np.sum(analytic * np.exp(2j * np.pi * frequency * time_index / fs))
    )
    assert 20 * np.log10(positive / negative) > 80.0
    assert np.isclose(np.sqrt(np.mean(body[:, 0] ** 2)), 1 / np.sqrt(2), rtol=1e-4)
    assert np.isclose(np.sqrt(np.mean(body[:, 1] ** 2)), 1 / np.sqrt(2), rtol=1e-3)


def test_streaming_hilbert_iq_mapping_swaps_left_and_right():
    samples = np.sin(np.linspace(0, 20, 2000, dtype=np.float32))
    lr = audio_io.StreamingHilbertIQ("iq_lr").process(samples)
    rl = audio_io.StreamingHilbertIQ("iq_rl").process(samples)

    assert np.array_equal(lr[:, 0], rl[:, 1])
    assert np.array_equal(lr[:, 1], rl[:, 0])


def test_stereo_iq_transmit_receive_loopback_reconstructs_mono():
    fs = 24000
    index = np.arange(2 * fs)
    source = (
        0.2 * np.cos(2 * np.pi * 1000 * index / fs)
        + 0.3 * np.cos(2 * np.pi * 5000 * index / fs)
        + 0.1 * np.sin(2 * np.pi * 8000 * index / fs)
    ).astype(np.float32)
    source_chunks = [source[:7777], source[7777:30001], source[30001:]]

    for mapping in ("iq_lr", "iq_rl"):
        iq_chunks = list(audio_io.iq_chunk_stream(source_chunks, mapping))
        receiver = audio_io.StreamingIQToMono(mapping)
        recovered = np.concatenate(
            [receiver.process(chunk) for chunk in iq_chunks]
        )
        # One group delay in each direction: 256 TX + 256 RX samples.
        aligned = recovered[512 : 512 + len(source)]
        assert np.corrcoef(aligned, source)[0, 1] > 0.9999999
        assert np.max(np.abs(aligned - source)) < 0.001


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


def test_stereo_iq_is_resampled_to_48k_without_channel_mixing(monkeypatch):
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
        query_devices=lambda device, kind: {"default_samplerate": 48000},
        OutputStream=Stream,
    )
    monkeypatch.setattr(audio_io, "_sd", lambda: sd)
    iq = np.column_stack(
        [np.ones(2400, dtype=np.float32), np.zeros(2400, dtype=np.float32)]
    )

    assert audio_io._play_chunk_stream_direct([iq], 24000, channels=2)
    rendered = opened["samples"]
    assert opened["samplerate"] == 48000
    assert opened["channels"] == 2
    assert rendered.shape[1] == 2
    assert np.max(np.abs(rendered[:, 0])) > 0.9
    assert np.array_equal(rendered[:, 1], np.zeros(len(rendered), dtype=np.float32))


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


def test_iq_input_opens_stereo_and_writes_reconstructed_mono(monkeypatch):
    opened = {}
    written = []

    class Stream:
        def __init__(self, **kwargs):
            opened.update(kwargs)

        def start(self):
            opened["started"] = True

    sd = SimpleNamespace(
        query_devices=lambda device, kind: {"default_samplerate": 24000},
        InputStream=Stream,
    )
    monkeypatch.setattr(audio_io, "_sd", lambda: sd)
    ring = SimpleNamespace(write=lambda samples: written.append(samples.copy()))
    source = np.cos(2 * np.pi * 3000 * np.arange(4096) / 24000).astype(np.float32)
    iq = audio_io.StreamingHilbertIQ("iq_lr").process(source)

    _stream, native = audio_io._open_input_stream_direct(
        None, ring, 24000, iq_mapping="iq_lr"
    )
    opened["callback"](iq, len(iq), None, None)

    assert native == 24000
    assert opened["channels"] == 2
    assert written[0].ndim == 1
    assert np.corrcoef(written[0][512:], source[:-512])[0, 1] > 0.999999


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


def test_packaged_audio_operations_use_console_helper(monkeypatch, tmp_path):
    app = tmp_path / "AETV.exe"
    helper = tmp_path / "audio-helper" / "AETV-Audio.exe"
    helper.parent.mkdir()
    helper.touch()
    monkeypatch.setattr(audio_io.sys, "frozen", True, raising=False)
    monkeypatch.setattr(audio_io.sys, "executable", str(app))

    assert audio_io._audio_worker_args("capture", 8000, "wasapi:mic") == [
        str(helper),
        "--audio-worker",
        "capture",
        "8000",
        '"wasapi:mic"',
        "1",
    ]


def test_audio_probe_protocol_returns_json(monkeypatch, capsys):
    devices = [audio_io.DeviceInfo(1, "Cable", 2, 0.0, True, "endpoint")]
    monkeypatch.setattr(audio_io, "_list_wasapi_devices_direct", lambda kind: devices)

    assert audio_io._audio_worker_main(["--audio-probe", "input"]) == 0

    assert json.loads(capsys.readouterr().out) == [
        {
            "index": 1,
            "name": "Cable",
            "channels": 2,
            "default_samplerate": 0.0,
            "is_default": True,
            "identifier": "endpoint",
        }
    ]


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
