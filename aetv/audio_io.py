"""Soundcard and WAV helpers for 8 kHz / 24 kHz AETV passband audio."""

from __future__ import annotations

import threading
import json
import os
import queue
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV as float32 mono in about ±1."""
    from scipy.io import wavfile

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
    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    if max_abs > 0:
        samples = samples * (peak / max_abs)
    wavfile.write(path, rate, samples)
    return path


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

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
    identifier: str = ""

    def label(self) -> str:
        star = " (default)" if self.is_default else ""
        return f"{self.index}: {self.name}{star}"

    def selection_value(self) -> str:
        return f"wasapi:{self.identifier}" if self.identifier else self.name


class AudioUnavailable(RuntimeError):
    """The platform audio backend could not be loaded or opened."""


def _sd():
    try:
        import sounddevice as sd
    except Exception as error:
        raise AudioUnavailable(f"sounddevice/PortAudio is not available: {error}") from error
    return sd


def _sc():
    try:
        import soundcard as sc
    except Exception as error:
        raise AudioUnavailable(f"SoundCard/WASAPI is not available: {error}") from error
    return sc


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


def _list_wasapi_devices_direct(kind: str) -> list[DeviceInfo]:
    if kind not in {"input", "output"}:
        raise ValueError(f"kind must be input or output, got {kind!r}")
    sc = _sc()
    if kind == "input":
        devices = sc.all_microphones(include_loopback=False)
        default = sc.default_microphone()
    else:
        devices = sc.all_speakers()
        default = sc.default_speaker()
    default_id = str(getattr(default, "id", "")).lower()
    return [
        DeviceInfo(
            index=index,
            name=str(device.name),
            channels=int(getattr(device, "channels", 1)),
            # WASAPI shared mode performs the rate conversion itself.
            default_samplerate=0.0,
            is_default=str(device.id).lower() == default_id,
            identifier=str(device.id),
        )
        for index, device in enumerate(devices)
    ]


def list_audio_devices(kind: str) -> list[DeviceInfo]:
    """List devices without allowing a broken native driver to kill the GUI.

    Windows uses the native WASAPI endpoint inventory because PortAudio loads
    every installed host API and one bad ASIO/webcam driver can abort the whole
    enumeration. The probe remains isolated as a final safety boundary.
    """
    if os.name != "nt":
        return _list_audio_devices_direct(kind)
    if os.environ.get("AETV_AUDIO_PROBE_CHILD") == "1":
        return _list_wasapi_devices_direct(kind)
    env = os.environ.copy()
    env["AETV_AUDIO_PROBE_CHILD"] = "1"
    try:
        proc = subprocess.run(
            [*_audio_helper_command(), "--audio-probe", kind],
            capture_output=True,
            text=True,
            timeout=12,
            env=env,
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


def wasapi_blocksize(rate: int) -> int:
    """Use a conservative 20 ms endpoint block for Windows live audio.

    The former 50 ms block was visible end to end and made a virtual-cable
    clock correction much larger than the OFDM cyclic prefix. Twenty
    milliseconds stays above common 10 ms shared-mode device periods while
    cutting both latency and the size of any endpoint-buffer correction.
    """
    return max(128, int(round(int(rate) * 0.020)))


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
        from scipy.signal import resample_poly

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


HILBERT_NUM_TAPS = 513


@lru_cache(maxsize=1)
def _hilbert_taps() -> np.ndarray:
    from scipy.signal import remez

    # DC and exact Nyquist have no unique quadrature value. AETV's useful
    # passband lies inside these conservative transition regions.
    return remez(
        HILBERT_NUM_TAPS, [0.01, 0.99], [1.0], type="hilbert", fs=2.0
    )


class StreamingHilbertIQ:
    """Convert real samples to phase-aligned analytic I/Q stereo frames.

    The FIR and exact I-path delay retain state across GOP chunks, avoiding
    phase or amplitude discontinuities at their boundaries. The convention is
    ``I + jQ`` with positive input frequencies.
    """

    NUM_TAPS = HILBERT_NUM_TAPS

    def __init__(self, mapping: str = "iq_lr"):
        if mapping not in {"iq_lr", "iq_rl"}:
            raise ValueError(f"unknown I/Q channel mapping {mapping!r}")
        self.mapping = mapping
        self.delay = (self.NUM_TAPS - 1) // 2
        self._taps = _hilbert_taps()
        self._q_state = np.zeros(self.NUM_TAPS - 1, dtype=np.float64)
        self._i_state = np.zeros(self.delay, dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Return float32 stereo frames for one mono chunk."""
        from scipy.signal import lfilter

        samples = np.asarray(chunk, dtype=np.float64).reshape(-1)
        if not samples.size:
            return np.empty((0, 2), dtype=np.float32)
        delayed = np.concatenate((self._i_state, samples))
        i = delayed[: len(samples)]
        self._i_state = delayed[len(samples) :]
        filtered, self._q_state = lfilter(
            self._taps, [1.0], samples, zi=self._q_state
        )
        # remez(type="hilbert") uses -j for positive frequencies. Negation
        # makes the emitted complex signal I+jQ analytic at +frequency.
        q = -filtered
        channels = (i, q) if self.mapping == "iq_lr" else (q, i)
        return np.column_stack(channels).astype(np.float32)

    def flush(self) -> np.ndarray:
        """Drain the complete FIR tail after the final source sample."""
        return self.process(np.zeros(self.NUM_TAPS - 1, dtype=np.float64))


class StreamingIQToMono:
    """Project analytic stereo I/Q back onto the real AETV waveform.

    Both inputs contribute: I is delayed by the Hilbert FIR group delay while
    Q passes through the same Hilbert transformer. For valid analytic input,
    ``0.5 * (I - H(Q))`` reconstructs the real component and averages the two
    independent channel observations.
    """

    NUM_TAPS = HILBERT_NUM_TAPS

    def __init__(self, mapping: str = "iq_lr"):
        if mapping not in {"iq_lr", "iq_rl"}:
            raise ValueError(f"unknown I/Q channel mapping {mapping!r}")
        self.mapping = mapping
        self.delay = (self.NUM_TAPS - 1) // 2
        self._taps = _hilbert_taps()
        self._q_state = np.zeros(self.NUM_TAPS - 1, dtype=np.float64)
        self._i_state = np.zeros(self.delay, dtype=np.float64)

    def process(self, frames: np.ndarray) -> np.ndarray:
        """Return reconstructed float32 mono samples for stereo IQ frames."""
        from scipy.signal import lfilter

        stereo = np.asarray(frames, dtype=np.float64).reshape(-1, 2)
        if not stereo.size:
            return np.empty(0, dtype=np.float32)
        if self.mapping == "iq_lr":
            i, q = stereo[:, 0], stereo[:, 1]
        else:
            q, i = stereo[:, 0], stereo[:, 1]
        delayed = np.concatenate((self._i_state, i))
        delayed_i = delayed[: len(i)]
        self._i_state = delayed[len(i) :]
        raw_q, self._q_state = lfilter(
            self._taps, [1.0], q, zi=self._q_state
        )
        h_q = -raw_q
        return (0.5 * (delayed_i - h_q)).astype(np.float32)


def iq_chunk_stream(chunks, mapping: str = "iq_lr", peak: float | None = None):
    """Yield a continuous analytic stereo stream, including its FIR tail."""
    converter = StreamingHilbertIQ(mapping)

    def limited(frames: np.ndarray) -> np.ndarray:
        if peak is not None and frames.size:
            maximum = float(np.max(np.abs(frames)))
            if maximum > peak:
                frames = frames * (float(peak) / maximum)
        return frames.astype(np.float32, copy=False)

    for chunk in chunks:
        yield limited(converter.process(chunk))
    yield limited(converter.flush())



def play_audio(audio: np.ndarray, rate: int, device: str | int | None = None) -> None:
    """Play mono audio, resampling to the device rate when needed."""
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        if not _play_chunk_stream_isolated([audio], rate, device=device):
            raise AudioUnavailable("soundcard output was cancelled")
        return
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
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        return _play_chunk_stream_isolated(
            [audio],
            rate,
            device=device,
            should_stop=should_stop,
            on_chunk=(lambda _count: on_progress(1.0)) if on_progress else None,
        )
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
    channels: int = 1,
) -> bool:
    """Write a lazy sequence of waveform chunks to one PortAudio stream."""
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        return _play_chunk_stream_isolated(
            chunks,
            rate,
            device=device,
            should_stop=should_stop,
            on_chunk=on_chunk,
            channels=channels,
        )
    return _play_chunk_stream_direct(
        chunks,
        rate,
        device=device,
        should_stop=should_stop,
        on_chunk=on_chunk,
        channels=channels,
    )


def _play_chunk_stream_direct(
    chunks,
    rate: int,
    device: str | int | None = None,
    should_stop=None,
    on_chunk=None,
    channels: int = 1,
) -> bool:
    if channels not in {1, 2}:
        raise ValueError("output channels must be 1 or 2")
    sd = _sd()
    device = resolve_device(device, "output")
    target_rate = _native_device_rate(sd, device, "output", rate)
    resamplers = None
    if target_rate != rate:
        ratio = resample_ratio(rate, target_rate)
        resamplers = [StreamResampler(*ratio) for _ in range(channels)]
    with sd.OutputStream(
        samplerate=target_rate, channels=channels, dtype="float32", device=device
    ) as stream:
        for index, chunk in enumerate(chunks):
            if should_stop is not None and should_stop():
                return False
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1, channels)
            if resamplers is not None:
                samples = np.column_stack(
                    [
                        resampler(samples[:, channel])
                        for channel, resampler in enumerate(resamplers)
                    ]
                ).astype(np.float32)
            if samples.size:
                stream.write(samples)
            if on_chunk is not None:
                on_chunk(index + 1)
    return not (should_stop is not None and should_stop())


def _audio_helper_command() -> list[str]:
    """Return the source or packaged executable for isolated Windows audio."""
    if getattr(sys, "frozen", False):
        suffix = ".exe" if str(sys.executable).lower().endswith(".exe") else ""
        helper = Path(sys.executable).resolve().parent / "audio-helper" / f"AETV-Audio{suffix}"
        if not helper.is_file():
            raise AudioUnavailable(f"packaged audio helper is missing: {helper}")
        return [str(helper)]
    return [sys.executable, "-m", "aetv.audio_io"]


def _audio_worker_args(
    operation: str, rate: int, device, channels: int = 1
) -> list[str]:
    return [
        *_audio_helper_command(),
        "--audio-worker",
        operation,
        str(int(rate)),
        json.dumps(device),
        str(int(channels)),
    ]


def _start_audio_worker(operation: str, rate: int, device, channels: int = 1):
    env = os.environ.copy()
    env["AETV_AUDIO_WORKER_CHILD"] = "1"
    env["AETV_AUDIO_PROBE_CHILD"] = "1"
    return subprocess.Popen(
        _audio_worker_args(operation, rate, device, channels),
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
    channels: int = 1,
) -> bool:
    """Stream via a child so a faulty native Windows driver cannot kill Qt."""
    if channels not in {1, 2}:
        raise ValueError("output channels must be 1 or 2")
    proc = _start_audio_worker("play", rate, device, channels)
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
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1, channels)
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
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        chunks = []
        lock = threading.Lock()

        class Collector:
            def write(self, samples) -> None:
                with lock:
                    chunks.append(np.asarray(samples, dtype=np.float32).copy())

        stream, _native = open_input_stream(device, Collector(), rate)
        try:
            threading.Event().wait(max(0.0, float(duration_s)))
        finally:
            stream.stop()
            stream.close()
        with lock:
            recorded = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return recorded[: int(round(duration_s * rate))]
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


class AudioPlaybackStream:
    """A persistent mono output used for received analog program audio."""

    def __init__(self, rate: int, device: str | int | None = None):
        self._rate = int(rate)
        self._device = device
        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self._stop = threading.Event()
        self._sentinel = object()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="aetv-program-audio",
        )
        self._thread.start()

    def _chunks(self):
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                return
            yield item

    def _run(self) -> None:
        try:
            play_chunk_stream(
                self._chunks(),
                self._rate,
                device=self._device,
                should_stop=self._stop.is_set,
            )
        except Exception as error:
            self._error = error

    def write(self, audio: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size:
            self._queue.put(samples.copy())

    def close(self) -> None:
        try:
            self._queue.put(self._sentinel, timeout=1.0)
            self._thread.join(timeout=5.0)
        finally:
            if self._thread.is_alive():
                self._stop.set()
                self._thread.join(timeout=3.0)


def open_input_stream(
    device,
    ring,
    samplerate: int,
    on_error=None,
    on_discontinuity=None,
    iq_mapping: str | None = None,
):
    """Open a capture stream that writes resampled mono float audio into `ring`."""
    if os.name == "nt" and os.environ.get("AETV_AUDIO_WORKER_CHILD") != "1":
        return _open_input_stream_isolated(
            device, ring, samplerate, on_error, on_discontinuity, iq_mapping
        )
    return _open_input_stream_direct(
        device, ring, samplerate, on_error, on_discontinuity, iq_mapping
    )


def _open_input_stream_direct(
    device,
    ring,
    samplerate: int,
    on_error=None,
    on_discontinuity=None,
    iq_mapping: str | None = None,
):
    sd = _sd()
    device = resolve_device(device, "input")
    report = on_error or (lambda _msg: None)
    report_gap = on_discontinuity or (lambda: None)

    channels = 2 if iq_mapping is not None else 1
    iq_decoder = StreamingIQToMono(iq_mapping) if iq_mapping is not None else None

    def make_callback(resamplers=None):
        def callback(indata, _frames, _time, status):
            if status:
                report(f"audio in: {status}")
                report_gap()
            samples = np.asarray(indata, dtype=np.float32).reshape(-1, channels)
            if resamplers is not None:
                samples = np.column_stack(
                    [
                        resampler(samples[:, channel])
                        for channel, resampler in enumerate(resamplers)
                    ]
                ).astype(np.float32)
            mono = iq_decoder.process(samples) if iq_decoder else samples[:, 0]
            ring.write(mono)

        return callback

    try:
        native = _native_device_rate(sd, device, "input", samplerate)
    except Exception as error:
        report(f"audio in: could not query device rate ({error})")
        native = samplerate
    resamplers = None
    if native != samplerate:
        ratio = resample_ratio(native, samplerate)
        resamplers = [StreamResampler(*ratio) for _ in range(channels)]
    stream = sd.InputStream(
        samplerate=native,
        channels=channels,
        dtype="float32",
        device=device,
        callback=make_callback(resamplers),
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
    def __init__(self, proc, ring, report, report_gap, iq_mapping=None):
        self._proc = proc
        self._ring = ring
        self._report = report
        self._report_gap = report_gap
        self._channels = 2 if iq_mapping is not None else 1
        self._iq_decoder = (
            StreamingIQToMono(iq_mapping) if iq_mapping is not None else None
        )
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
            if size % (4 * self._channels) or size > 16 * 1024 * 1024:
                self._report("audio in: invalid helper frame")
                break
            payload = _read_exact(pipe, size)
            if len(payload) != size:
                break
            samples = np.frombuffer(payload, dtype="<f4").reshape(
                -1, self._channels
            )
            mono = (
                self._iq_decoder.process(samples)
                if self._iq_decoder is not None
                else samples[:, 0].copy()
            )
            self._ring.write(mono)
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
    device,
    ring,
    samplerate: int,
    on_error=None,
    on_discontinuity=None,
    iq_mapping: str | None = None,
):
    report = on_error or (lambda _msg: None)
    report_gap = on_discontinuity or (lambda: None)
    channels = 2 if iq_mapping is not None else 1
    proc = _start_audio_worker("capture", samplerate, device, channels)
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
    return _InputProcessStream(proc, ring, report, report_gap, iq_mapping), native


def _audio_play_worker(rate: int, device, channels: int = 1) -> int:
    if os.name == "nt":
        return _wasapi_play_worker(rate, device, channels)
    sd = _sd()
    device = resolve_device(device, "output")
    target_rate = _native_device_rate(sd, device, "output", rate)
    resamplers = None
    if target_rate != rate:
        ratio = resample_ratio(rate, target_rate)
        resamplers = [StreamResampler(*ratio) for _ in range(channels)]
    source = sys.stdin.buffer
    target = sys.stdout.buffer
    with sd.OutputStream(
        samplerate=target_rate, channels=channels, dtype="float32", device=device
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
            samples = np.frombuffer(payload, dtype="<f4").reshape(-1, channels)
            if resamplers is not None:
                samples = np.column_stack(
                    [
                        resampler(samples[:, channel])
                        for channel, resampler in enumerate(resamplers)
                    ]
                ).astype(np.float32)
            if samples.size:
                stream.write(samples)
            target.write(b"A")
            target.flush()
    return 0


def _audio_capture_worker(rate: int, device, channels: int = 1) -> int:
    if os.name == "nt":
        return _wasapi_capture_worker(rate, device, channels)
    sd = _sd()
    device = resolve_device(device, "input")
    native = _native_device_rate(sd, device, "input", rate)
    resamplers = None
    if native != rate:
        ratio = resample_ratio(native, rate)
        resamplers = [StreamResampler(*ratio) for _ in range(channels)]
    blocks: queue.Queue = queue.Queue(maxsize=32)
    overflow = threading.Event()

    def callback(indata, _frames, _time, status):
        if status:
            overflow.set()
        samples = np.asarray(indata, dtype=np.float32).reshape(-1, channels)
        if resamplers is not None:
            samples = np.column_stack(
                [
                    resampler(samples[:, channel])
                    for channel, resampler in enumerate(resamplers)
                ]
            )
        if len(samples):
            try:
                blocks.put_nowait(np.asarray(samples, dtype="<f4").tobytes())
            except queue.Full:
                overflow.set()

    stream = sd.InputStream(
        samplerate=native,
        channels=channels,
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


def _wasapi_device(device, kind: str):
    sc = _sc()
    if kind == "output":
        if device in {None, ""}:
            selected = sc.default_speaker()
        else:
            selector = str(device)
            if selector.startswith("wasapi:"):
                selector = selector.removeprefix("wasapi:")
            selected = sc.get_speaker(selector)
    else:
        if device in {None, ""}:
            selected = sc.default_microphone()
        else:
            selector = str(device)
            if selector.startswith("wasapi:"):
                selector = selector.removeprefix("wasapi:")
            selected = sc.get_microphone(selector, include_loopback=False)
    if selected is None:
        raise AudioUnavailable(f"selected WASAPI {kind} device is unavailable")
    return selected


def _wasapi_play_worker(rate: int, device, channels: int = 1) -> int:
    speaker = _wasapi_device(device, "output")
    source = sys.stdin.buffer
    target = sys.stdout.buffer
    blocksize = wasapi_blocksize(rate)
    # SoundCard's WASAPI backend has a long-standing single-channel capture
    # bug. Keep both sides of the Windows path stereo so virtual cables and
    # hardware endpoints negotiate an ordinary interleaved stream.
    with speaker.player(samplerate=rate, channels=2, blocksize=blocksize) as player:
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
            samples = np.frombuffer(payload, dtype="<f4").reshape(-1, channels)
            if samples.size:
                _play_wasapi_exact(player, samples, channels=channels)
            target.write(b"A")
            target.flush()
    return 0


def _play_wasapi_exact(
    player, samples: np.ndarray, memmove=None, channels: int = 1
) -> None:
    """Submit exactly the initialized frames to SoundCard's WASAPI client.

    SoundCard 0.4.x requests every currently available render frame inside
    ``Player.play``. When the final piece is shorter than that availability it
    copies only the short piece but releases the entire region, exposing stale
    endpoint memory. Continuous GOP chunks deliberately have unequal first and
    steady-state lengths, so the resulting boundary-dependent samples can move
    the OFDM window while still looking coherent enough to remain locked.

    Keep SoundCard's endpoint setup but release only the frames copied. The
    optional memmove is a narrow test seam; production uses SoundCard's CFFI.
    """
    if memmove is None:
        from soundcard import mediafoundation

        memmove = mediafoundation._ffi.memmove
    frames = np.asarray(samples, dtype=np.float32).reshape(-1, channels)
    stereo = np.repeat(frames, 2, axis=1) if channels == 1 else frames
    cursor = 0
    while cursor < len(stereo):
        available = int(player._render_available_frames())
        if available <= 0:
            time.sleep(0.001)
            continue
        count = min(available, len(stereo) - cursor)
        payload = stereo[cursor : cursor + count].ravel().tobytes()
        buffer = player._render_buffer(count)
        memmove(buffer[0], payload, len(payload))
        player._render_release(count)
        cursor += count


def _wasapi_capture_worker(rate: int, device, channels: int = 1) -> int:
    microphone = _wasapi_device(device, "input")
    target = sys.stdout.buffer
    blocksize = wasapi_blocksize(rate)
    with microphone.recorder(
        samplerate=rate, channels=2, blocksize=blocksize
    ) as recorder:
        try:
            target.write(b"R" + struct.pack("<I", rate))
            target.flush()
            while True:
                samples = recorder.record(numframes=blocksize)
                if channels == 2:
                    output = np.asarray(samples, dtype=np.float32).reshape(-1, 2)
                else:
                    output = _downmix_wasapi_capture(samples)
                payload = output.astype("<f4", copy=False).tobytes()
                target.write(struct.pack("<I", len(payload)))
                target.write(payload)
                target.flush()
        except (BrokenPipeError, OSError):
            return 0


def _downmix_wasapi_capture(samples: np.ndarray) -> np.ndarray:
    """Convert the stereo WASAPI transport to the modem's mono stream."""
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.shape[1] == 1:
        return array[:, 0]
    return np.mean(array, axis=1, dtype=np.float32)


def _audio_worker_main(args: list[str]) -> int:
    if len(args) == 2 and args[0] == "--audio-probe":
        try:
            devices = _list_wasapi_devices_direct(args[1])
            print(json.dumps([asdict(device) for device in devices]), flush=True)
            return 0
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return 1
    if len(args) not in {4, 5} or args[0] != "--audio-worker":
        return 2
    operation, rate_text, device_text = args[1:4]
    try:
        rate = int(rate_text)
        device = json.loads(device_text)
        channels = int(args[4]) if len(args) == 5 else 1
        if operation == "play":
            return _audio_play_worker(rate, device, channels)
        if operation == "capture":
            return _audio_capture_worker(rate, device, channels)
        raise ValueError(f"unknown audio operation {operation!r}")
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_audio_worker_main(sys.argv[1:]))
