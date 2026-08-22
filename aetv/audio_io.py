"""Soundcard and WAV helpers for 8 kHz / 24 kHz AETV passband audio."""

from __future__ import annotations

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


def list_devices() -> list[dict]:
    import sounddevice as sd

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


def play_audio(audio: np.ndarray, rate: int, device: str | int | None = None) -> None:
    """Play mono audio, resampling to the device rate when needed."""
    import sounddevice as sd

    target_rate = rate
    if device is not None:
        info = sd.query_devices(device)
        native = int(info.get("default_samplerate") or rate)
        if native != rate:
            audio = resample_audio(audio, rate, native)
            target_rate = native
    sd.play(audio.astype(np.float32), samplerate=target_rate, device=device)
    sd.wait()


def record_audio(
    duration_s: float,
    rate: int,
    device: str | int | None = None,
) -> np.ndarray:
    """Record mono float32 audio for ``duration_s`` seconds."""
    import sounddevice as sd

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
