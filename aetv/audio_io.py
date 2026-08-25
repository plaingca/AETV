"""Soundcard and WAV helpers for 8 kHz / 24 kHz AETV passband audio."""

from __future__ import annotations

import threading
import json
import os
import queue
import struct
import subprocess
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV as float32 mono in about ±1."""
    rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / abs(float(np.iinfo(audio.dtype).min))
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return int(rate), audio


def write_wav(path: str | Path, rate: int, audio: np.ndarray, peak: float = 0.7) -> Path:
    """Write mono float32 audio, peak-scaled so DAX/soundcards stay out of clip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    if max_abs > 0:
        samples = samples * (peak / max_abs)
    wavfile.write(path, rate, samples)
    return path


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    return resample_poly(audio, dst_rate, src_rate).astype(np.float32)


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool

    def label(self) -> str:
        star = " (default)" if self.is_default else ""
        return f"{self.index}: {self.name}{star}"


class AudioUnavailable(RuntimeError):
    """PortAudio/sounddevice could not be loaded or found no devices."""


def _sd():
    try:
        import sounddevice as sd
    except Exception as error:
        raise AudioUnavailable(f"sounddevice/PortAudio is not available: {error}") from error
    return sd


def list_devices() -> list[dict]:
    sd = _sd()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": index,
                "name": info["name"],
                "inputs": int(info["max_input_channels"]),
                "outputs": int(info["max_output_channels"]),
                "default_rate": int(info.get("default_samplerate") or 0),
            }
        )
    return devices


def _list_audio_devices_direct(kind: str) -> list[DeviceInfo]:
    if kind not in {"input", "output"}:
        raise ValueError(f"kind must be input or output, got {kind!r}")
    sd = _sd()
    key = f"max_{kind}_channels"
    try:
        default_idx = sd.default.device[0 if kind == "input" else 1]
    except Exception:
        default_idx = None
    out = []
    for index, info in enumerate(sd.query_devices()):
        if info[key] < 1:
            continue
        out.append(
            DeviceInfo(
                index=index,
                name=info["name"],
                channels=int(info[key]),
                default_samplerate=float(info.get("default_samplerate") or 0),
                is_default=(index == default_idx),
            )
        )
    return out


def list_audio_devices(kind: str) -> list[DeviceInfo]:
    """List devices without allowing a broken Windows driver to kill the GUI.

    PortAudio enumerates every installed host API. Some stale ASIO/webcam audio
    drivers crash inside native code rather than raising an exception, so the
    Windows probe is isolated in a short-lived helper process.
    """
    if os.name != "nt" or os.environ.get("AETV_AUDIO_PROBE_CHILD") == "1":
        return _list_audio_devices_direct(kind)
    code = (
        "import json; from dataclasses import asdict; "
        "from aetv.audio_io import _list_audio_devices_direct; "
        f"print(json.dumps([asdict(x) for x in _list_audio_devices_direct({kind!r})]))"
    )
    env = os.environ.copy()
    env["AETV_AUDIO_PROBE_CHILD"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=12, env=env
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        return [DeviceInfo(**item) for item in json.loads(proc.stdout)]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def resolve_device(name_or_index: str | int | None, kind: str) -> str | int | None:
    """Prefer a stored device *name*; indices change when USB cards move."""
    if name_or_index is None or name_or_index == "":
        return None
    if isinstance(name_or_index, int):
        return name_or_index
    text = str(name_or_index)
    if text.isdigit():
        return int(text)
    for item in list_audio_devices(kind):
        if item.name == text:
            return item.index
    return text


def _native_device_rate(sd, device, kind: str, requested_rate: int) -> int:
    """Return the selected device rate, including PortAudio's default device."""
    info = sd.query_devices(device, kind)
    return int(round(info.get("default_samplerate") or requested_rate))


def resample_ratio(src_rate: int, dst_rate: int) -> tuple[int, int]:
    g = gcd(int(src_rate), int(dst_rate))
    return int(dst_rate) // g, int(src_rate) // g


class StreamResampler:
    """`resample_poly` for a stream that arrives in chunks.

    Calling `resample_poly` on each callback independently pads both
    ends and puts a filter transient on every boundary. This keeps FIR
    context so the output matches a one-shot resample of the whole
    stream, sample for sample.
    """

    def __init__(self, up: int, down: int):
        self.up = int(up)
        self.down = int(down)
        span = -(-(20 * max(self.up, self.down) + 1) // self.up)
        self.pad = -(-span // self.down) * self.down
        self._buf = np.zeros(self.pad, dtype=np.float64)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float64).reshape(-1)])
        usable = len(self._buf) - 2 * self.pad
        n = (usable // self.down) * self.down if usable > 0 else 0
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        block = self._buf[: 2 * self.pad + n]
        y = resample_poly(block, self.up, self.down)
        skip = self.pad * self.up // self.down
        take = n * self.up // self.down
        self._buf = self._buf[n:]
        return y[skip : skip + take]



def play_audio(audio: np.ndarray, rate: int, device: str | int | None = None) -> None:
    """Play mono audio, resampling to the device rate when needed."""
    sd = _sd()
    device = resolve_device(device, "output")
    target_rate = _native_device_rate(sd, device, "output", rate)
    if target_rate != rate:
        audio = resample_audio(audio, rate, target_rate)
    sd.play(audio.astype(np.float32), samplerate=target_rate, device=device)
    sd.wait()


def play_cancellable(
    audio: np.ndarray,
    rate: int,
    device: str | int | None = None,
    should_stop=None,
    on_progress=None,
) -> bool:
    """Play a prepared waveform; return False if `should_stop` fired."""
    sd = _sd()
    device = resolve_device(device, "output")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    target_rate = _native_device_rate(sd, device, "output", rate)
    if target_rate != rate:
        samples = resample_audio(samples, rate, target_rate)
    cursor = 0
    done = threading.Event()
    cancelled = threading.Event()

    def callback(outdata, frames, _time, status):
        nonlocal cursor
        if should_stop is not None and should_stop():
            cancelled.set()
            outdata[:] = 0
            raise sd.CallbackStop
        need = frames
        remaining = len(samples) - cursor
        if remaining <= 0:
            outdata[:] = 0
            done.set()
            raise sd.CallbackStop
        take = min(need, remaining)
        outdata[:take, 0] = samples[cursor : cursor + take]
        if take < need:
            outdata[take:] = 0
            done.set()
            raise sd.CallbackStop
        cursor += take
        if on_progress is not None:
            on_progress(cursor / max(1, len(samples)))

    stream = sd.OutputStream(
        samplerate=target_rate,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    )
    with stream:
        while not done.is_set() and not cancelled.is_set():
            if should_stop is not None and should_stop():
                cancelled.set()
                break
            done.wait(0.05)
    return not cancelled.is_set()


def play_chunk_stream(
    chunks,
    rate: int,
    device: str | int | None = None,
    should_stop=None,
    on_chunk=None,
) -> bool:
    """Write a lazy sequence of waveform chunks to one PortAudio stream."""
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        return _play_chunk_stream_isolated(
            chunks,
            rate,
            device=device,
            should_stop=should_stop,
            on_chunk=on_chunk,
        )
    return _play_chunk_stream_direct(
        chunks,
        rate,
        device=device,
        should_stop=should_stop,
        on_chunk=on_chunk,
    )


def _play_chunk_stream_direct(
    chunks,
    rate: int,
    device: str | int | None = None,
    should_stop=None,
    on_chunk=None,
) -> bool:
    sd = _sd()
    device = resolve_device(device, "output")
    target_rate = _native_device_rate(sd, device, "output", rate)
    resampler = (
        None if target_rate == rate else StreamResampler(*resample_ratio(rate, target_rate))
    )
    with sd.OutputStream(
        samplerate=target_rate, channels=1, dtype="float32", device=device
    ) as stream:
        for index, chunk in enumerate(chunks):
            if should_stop is not None and should_stop():
                return False
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if resampler is not None:
                samples = resampler(samples).astype(np.float32)
            if samples.size:
                stream.write(samples.reshape(-1, 1))
            if on_chunk is not None:
                on_chunk(index + 1)
    return not (should_stop is not None and should_stop())


def _audio_worker_args(operation: str, rate: int, device) -> list[str]:
    return [
        sys.executable,
        "-m",
        "aetv.audio_io",
        "--audio-worker",
        operation,
        str(int(rate)),
        json.dumps(device),
    ]


def _start_audio_worker(operation: str, rate: int, device):
    env = os.environ.copy()
    env["AETV_AUDIO_WORKER_CHILD"] = "1"
    env["AETV_AUDIO_PROBE_CHILD"] = "1"
    return subprocess.Popen(
        _audio_worker_args(operation, rate, device),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _worker_error(proc, operation: str) -> AudioUnavailable:
    code = proc.poll()
    if code is None:
        try:
            code = proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    detail = ""
    if code is not None and proc.stderr is not None:
        try:
            detail = proc.stderr.read().decode("utf-8", "replace").strip()
        except OSError:
            pass
    if code is not None and (code & 0xFFFFFFFF) == 0xC0000374:
        detail = "an installed Windows audio driver reported heap corruption"
    suffix = f": {detail.splitlines()[-1]}" if detail else ""
    return AudioUnavailable(
        f"soundcard {operation} helper stopped"
        f" (exit {code if code is not None else 'unknown'}){suffix}"
    )


def _terminate_worker(proc) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def _signal_reader(pipe, signals: queue.Queue) -> None:
    try:
        while True:
            value = pipe.read(1)
            signals.put(value or None)
            if not value:
                return
    except OSError:
        signals.put(None)


def _wait_for_worker_signal(
    proc,
    signals: queue.Queue,
    expected: bytes,
    *,
    operation: str,
    should_stop=None,
    timeout_s: float = 15.0,
) -> bool:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            _terminate_worker(proc)
            return False
        try:
            value = signals.get(timeout=0.05)
        except queue.Empty:
            if proc.poll() is not None:
                raise _worker_error(proc, operation)
            continue
        if value == expected:
            return True
        raise _worker_error(proc, operation)
    _terminate_worker(proc)
    raise AudioUnavailable(f"soundcard {operation} helper did not respond")


def _play_chunk_stream_isolated(
    chunks,
    rate: int,
    device: str | int | None = None,
    should_stop=None,
    on_chunk=None,
) -> bool:
    """Stream via a child so a faulty native Windows driver cannot kill Qt."""
    proc = _start_audio_worker("play", rate, device)
    assert proc.stdin is not None and proc.stdout is not None
    signals: queue.Queue = queue.Queue()
    reader = threading.Thread(
        target=_signal_reader,
        args=(proc.stdout, signals),
        daemon=True,
        name="aetv-audio-output-status",
    )
    reader.start()
    cancelled = False
    try:
        if not _wait_for_worker_signal(
            proc, signals, b"R", operation="output", should_stop=should_stop
        ):
            return False
        for index, chunk in enumerate(chunks):
            if should_stop is not None and should_stop():
                cancelled = True
                return False
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
            payload = samples.astype("<f4", copy=False).tobytes()
            try:
                proc.stdin.write(struct.pack("<I", len(payload)))
                proc.stdin.write(payload)
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise _worker_error(proc, "output") from error
            duration = len(samples) / max(1, rate)
            if not _wait_for_worker_signal(
                proc,
                signals,
                b"A",
                operation="output",
                should_stop=should_stop,
                timeout_s=max(15.0, duration + 10.0),
            ):
                cancelled = True
                return False
            if on_chunk is not None:
                on_chunk(index + 1)
        proc.stdin.close()
        try:
            code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired as error:
            _terminate_worker(proc)
            raise AudioUnavailable(
                "soundcard output helper did not close cleanly"
            ) from error
        if code != 0:
            raise _worker_error(proc, "output")
        return not (should_stop is not None and should_stop())
    finally:
        if proc.poll() is None:
            if not cancelled:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            _terminate_worker(proc)


def record_audio(
    duration_s: float,
    rate: int,
    device: str | int | None = None,
) -> np.ndarray:
    """Record mono float32 audio for ``duration_s`` seconds."""
    sd = _sd()
    device = resolve_device(device, "input")
    capture_rate = _native_device_rate(sd, device, "input", rate)
    recorded = sd.rec(
        int(round(duration_s * capture_rate)),
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    mono = recorded[:, 0]
    if capture_rate != rate:
        mono = resample_audio(mono, capture_rate, rate)
    return mono.astype(np.float32)


def open_input_stream(
    device, ring, samplerate: int, on_error=None, on_discontinuity=None
):
    """Open a capture stream that writes resampled mono float audio into `ring`."""
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        return _open_input_stream_isolated(
            device, ring, samplerate, on_error, on_discontinuity
        )
    return _open_input_stream_direct(
        device, ring, samplerate, on_error, on_discontinuity
    )


def _open_input_stream_direct(
    device, ring, samplerate: int, on_error=None, on_discontinuity=None
):
    sd = _sd()
    device = resolve_device(device, "input")
    report = on_error or (lambda _msg: None)
    report_gap = on_discontinuity or (lambda: None)

    def make_callback(resample_fn=None):
        def callback(indata, _frames, _time, status):
            if status:
                report(f"audio in: {status}")
                report_gap()
            mono = indata[:, 0] if indata.ndim > 1 else indata
            ring.write(resample_fn(mono) if resample_fn else mono)

        return callback

    try:
        native = _native_device_rate(sd, device, "input", samplerate)
    except Exception as error:
        report(f"audio in: could not query device rate ({error})")
        native = samplerate
    resampler = None if native == samplerate else StreamResampler(*resample_ratio(native, samplerate))
    stream = sd.InputStream(
        samplerate=native,
        channels=1,
        dtype="float32",
        device=device,
        callback=make_callback(resampler),
    )
    stream.start()
    return stream, native


def _read_exact(pipe, size: int) -> bytes:
    parts = []
    remaining = size
    while remaining:
        part = pipe.read(remaining)
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


class _InputProcessStream:
    def __init__(self, proc, ring, report, report_gap):
        self._proc = proc
        self._ring = ring
        self._report = report
        self._report_gap = report_gap
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._read,
            daemon=True,
            name="aetv-audio-input-reader",
        )
        self._thread.start()

    def _read(self) -> None:
        pipe = self._proc.stdout
        assert pipe is not None
        while not self._stopping.is_set():
            header = _read_exact(pipe, 4)
            if len(header) != 4:
                break
            size = struct.unpack("<I", header)[0]
            if size == 0:
                self._report("audio in: helper buffer overrun; reacquiring")
                self._report_gap()
                continue
            if size % 4 or size > 16 * 1024 * 1024:
                self._report("audio in: invalid helper frame")
                break
            payload = _read_exact(pipe, size)
            if len(payload) != size:
                break
            self._ring.write(np.frombuffer(payload, dtype="<f4").copy())
        if not self._stopping.is_set():
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _terminate_worker(self._proc)
            self._report(str(_worker_error(self._proc, "input")))

    def stop(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        _terminate_worker(self._proc)
        self._thread.join(timeout=3)

    def close(self) -> None:
        self.stop()


def _open_input_stream_isolated(
    device, ring, samplerate: int, on_error=None, on_discontinuity=None
):
    report = on_error or (lambda _msg: None)
    report_gap = on_discontinuity or (lambda: None)
    proc = _start_audio_worker("capture", samplerate, device)
    assert proc.stdout is not None
    ready: queue.Queue = queue.Queue()

    def read_ready() -> None:
        ready.put(_read_exact(proc.stdout, 5))

    bootstrap = threading.Thread(target=read_ready, daemon=True)
    bootstrap.start()
    try:
        header = ready.get(timeout=15)
    except queue.Empty:
        _terminate_worker(proc)
        raise AudioUnavailable("soundcard input helper did not respond")
    if len(header) != 5 or header[:1] != b"R":
        _terminate_worker(proc)
        raise _worker_error(proc, "input")
    native = struct.unpack("<I", header[1:])[0]
    return _InputProcessStream(proc, ring, report, report_gap), native


def _audio_play_worker(rate: int, device) -> int:
    sd = _sd()
    device = resolve_device(device, "output")
    target_rate = _native_device_rate(sd, device, "output", rate)
    resampler = (
        None if target_rate == rate else StreamResampler(*resample_ratio(rate, target_rate))
    )
    source = sys.stdin.buffer
    target = sys.stdout.buffer
    with sd.OutputStream(
        samplerate=target_rate, channels=1, dtype="float32", device=device
    ) as stream:
        target.write(b"R")
        target.flush()
        while True:
            header = _read_exact(source, 4)
            if not header:
                break
            if len(header) != 4:
                raise AudioUnavailable("truncated audio output frame")
            size = struct.unpack("<I", header)[0]
            if size % 4 or size > 64 * 1024 * 1024:
                raise AudioUnavailable("invalid audio output frame")
            payload = _read_exact(source, size)
            if len(payload) != size:
                raise AudioUnavailable("truncated audio output samples")
            samples = np.frombuffer(payload, dtype="<f4")
            if resampler is not None:
                samples = resampler(samples).astype(np.float32)
            if samples.size:
                stream.write(samples.reshape(-1, 1))
            target.write(b"A")
            target.flush()
    return 0


def _audio_capture_worker(rate: int, device) -> int:
    sd = _sd()
    device = resolve_device(device, "input")
    native = _native_device_rate(sd, device, "input", rate)
    resampler = None if native == rate else StreamResampler(*resample_ratio(native, rate))
    blocks: queue.Queue = queue.Queue(maxsize=32)
    overflow = threading.Event()

    def callback(indata, _frames, _time, status):
        if status:
            overflow.set()
        mono = indata[:, 0] if indata.ndim > 1 else indata
        samples = resampler(mono) if resampler is not None else mono
        if len(samples):
            try:
                blocks.put_nowait(np.asarray(samples, dtype="<f4").tobytes())
            except queue.Full:
                overflow.set()

    stream = sd.InputStream(
        samplerate=native,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    )
    stream.start()
    target = sys.stdout.buffer
    try:
        target.write(b"R" + struct.pack("<I", native))
        target.flush()
        while True:
            payload = blocks.get()
            if overflow.is_set():
                overflow.clear()
                target.write(struct.pack("<I", 0))
            target.write(struct.pack("<I", len(payload)))
            target.write(payload)
            target.flush()
    except (BrokenPipeError, OSError):
        return 0
    finally:
        stream.stop()
        stream.close()


def _audio_worker_main(args: list[str]) -> int:
    if len(args) != 4 or args[0] != "--audio-worker":
        return 2
    operation, rate_text, device_text = args[1:]
    try:
        rate = int(rate_text)
        device = json.loads(device_text)
        if operation == "play":
            return _audio_play_worker(rate, device)
        if operation == "capture":
            return _audio_capture_worker(rate, device)
        raise ValueError(f"unknown audio operation {operation!r}")
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_audio_worker_main(sys.argv[1:]))
