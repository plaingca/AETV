"""Soundcard and WAV helpers for 8 kHz / 24 kHz AETV passband audio."""

from __future__ import annotations

import threading
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


def list_audio_devices(kind: str) -> list[DeviceInfo]:
    """Input or output devices, skipping cards with no channels that way."""
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
    target_rate = rate
    if device is not None:
        info = sd.query_devices(device)
        native = int(info.get("default_samplerate") or rate)
        if native != rate:
            audio = resample_audio(audio, rate, native)
            target_rate = native
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
    target_rate = rate
    if device is not None:
        info = sd.query_devices(device)
        native = int(info.get("default_samplerate") or rate)
        if native != rate:
            samples = resample_audio(samples, rate, native)
            target_rate = native
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


def record_audio(
    duration_s: float,
    rate: int,
    device: str | int | None = None,
) -> np.ndarray:
    """Record mono float32 audio for ``duration_s`` seconds."""
    sd = _sd()
    device = resolve_device(device, "input")
    capture_rate = rate
    if device is not None:
        info = sd.query_devices(device)
        native = int(info.get("default_samplerate") or rate)
        if native != rate:
            capture_rate = native
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


def open_input_stream(device, ring, samplerate: int, on_error=None):
    """Open a capture stream that writes resampled mono float audio into `ring`."""
    sd = _sd()
    device = resolve_device(device, "input")
    report = on_error or (lambda _msg: None)

    def make_callback(resample_fn=None):
        def callback(indata, _frames, _time, status):
            if status:
                report(f"audio in: {status}")
            mono = indata[:, 0] if indata.ndim > 1 else indata
            ring.write(resample_fn(mono) if resample_fn else mono)

        return callback

    native = samplerate
    if device is not None:
        try:
            native = int(round(sd.query_devices(device, "input").get("default_samplerate") or samplerate))
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
