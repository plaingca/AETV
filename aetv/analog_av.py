"""Experimental analog-voice plus frequency-stacked AETV waveform.

The composite keeps ordinary 0--2.2 kHz speech at baseband and translates the
native V8/W waveform from 450--2650 Hz to 2600--4800 Hz.  Its explicitly
filtered lower skirt begins at 2500 Hz, leaving a real 300 Hz guard band at a
12 kHz soundcard rate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


COMPOSITE_FS = 12_000
NATIVE_AETV_FS = 8_000
VOICE_HIGH_HZ = 2_200.0
AETV_LOW_HZ = 2_600.0
AETV_HIGH_HZ = 4_800.0
AETV_FILTER_LOW_HZ = 2_500.0
AETV_FILTER_HIGH_HZ = 4_900.0
AETV_SHIFT_HZ = 2_150.0
GOP_SECONDS = 1.0


def _cosine_lowpass(values: np.ndarray, fs: int, pass_hz: float, stop_hz: float) -> np.ndarray:
    """Zero-phase FFT low-pass with a raised-cosine transition."""
    x = np.asarray(values, dtype=np.float64)
    spectrum = np.fft.rfft(x)
    frequency = np.fft.rfftfreq(len(x), 1.0 / fs)
    response = np.ones_like(frequency)
    response[frequency >= stop_hz] = 0.0
    transition = (frequency > pass_hz) & (frequency < stop_hz)
    response[transition] = 0.5 + 0.5 * np.cos(
        np.pi * (frequency[transition] - pass_hz) / (stop_hz - pass_hz)
    )
    return np.fft.irfft(spectrum * response, n=len(x))


def _cosine_bandpass(
    values: np.ndarray,
    fs: int,
    low_stop_hz: float,
    low_pass_hz: float,
    high_pass_hz: float,
    high_stop_hz: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    spectrum = np.fft.rfft(x)
    frequency = np.fft.rfftfreq(len(x), 1.0 / fs)
    response = np.zeros_like(frequency)
    response[(frequency >= low_pass_hz) & (frequency <= high_pass_hz)] = 1.0
    lower = (frequency > low_stop_hz) & (frequency < low_pass_hz)
    upper = (frequency > high_pass_hz) & (frequency < high_stop_hz)
    response[lower] = 0.5 - 0.5 * np.cos(
        np.pi * (frequency[lower] - low_stop_hz) / (low_pass_hz - low_stop_hz)
    )
    response[upper] = 0.5 + 0.5 * np.cos(
        np.pi * (frequency[upper] - high_pass_hz) / (high_stop_hz - high_pass_hz)
    )
    return np.fft.irfft(spectrum * response, n=len(x))


def prepare_voice(audio_8k: np.ndarray) -> np.ndarray:
    """Band-limit speech and convert it to the 12 kHz composite rate."""
    audio = signal.resample_poly(np.asarray(audio_8k, dtype=np.float64), 3, 2)
    return _cosine_lowpass(audio, COMPOSITE_FS, 2_100.0, VOICE_HIGH_HZ)


def translate_aetv_up(native_waveform: np.ndarray) -> np.ndarray:
    """Move a native 8 kHz W waveform into the 2.6--4.8 kHz upper slice."""
    native_12k = signal.resample_poly(
        np.asarray(native_waveform, dtype=np.float64), 3, 2,
        window=("kaiser", 12.0),
    )
    time = np.arange(len(native_12k), dtype=np.float64) / COMPOSITE_FS
    shifted = np.real(
        signal.hilbert(native_12k) * np.exp(2j * np.pi * AETV_SHIFT_HZ * time)
    )
    return _cosine_bandpass(
        shifted, COMPOSITE_FS,
        AETV_FILTER_LOW_HZ, 2_550.0, 4_850.0, AETV_FILTER_HIGH_HZ,
    )


def extract_voice(composite: np.ndarray) -> np.ndarray:
    """Recover the analog speech slice at the native 8 kHz media rate."""
    voice_12k = _cosine_lowpass(
        np.asarray(composite, dtype=np.float64), COMPOSITE_FS, 2_100.0, VOICE_HIGH_HZ
    )
    return signal.resample_poly(voice_12k, 2, 3, window=("kaiser", 12.0))


def extract_aetv(composite: np.ndarray) -> np.ndarray:
    """Isolate, translate, and resample the upper AETV slice back to native W."""
    upper = _cosine_bandpass(
        np.asarray(composite, dtype=np.float64), COMPOSITE_FS,
        AETV_FILTER_LOW_HZ, 2_550.0, 4_850.0, AETV_FILTER_HIGH_HZ,
    )
    time = np.arange(len(upper), dtype=np.float64) / COMPOSITE_FS
    native_12k = np.real(signal.hilbert(upper) * np.exp(-2j * np.pi * AETV_SHIFT_HZ * time))
    return signal.resample_poly(native_12k, 2, 3, window=("kaiser", 12.0))


def compose_delayed_stream(
    aetv_gops_8k: list[np.ndarray],
    voice_gops_8k: list[np.ndarray],
    *,
    voice_rms: float = 0.22,
    aetv_rms: float = 0.22,
    peak: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack AETV and one-GOP-delayed speech into a continuous waveform.

    During interval ``n`` the upper slice carries video GOP ``n`` while the
    lower slice carries voice GOP ``n-1``.  At the receiver, decoded video GOP
    ``n-1`` and analog voice GOP ``n-1`` therefore become available together.
    One final voice-only interval drains the delay line.
    """
    if len(aetv_gops_8k) != len(voice_gops_8k) or not aetv_gops_8k:
        raise ValueError("AETV and voice GOP lists must have the same non-zero length")
    samples = int(COMPOSITE_FS * GOP_SECONDS)
    upper_gops = [translate_aetv_up(gop) for gop in aetv_gops_8k]
    speech_gops = [prepare_voice(gop) for gop in voice_gops_8k]
    if any(len(gop) != samples for gop in upper_gops + speech_gops):
        raise ValueError("every input GOP must be exactly one second")

    upper = np.concatenate(upper_gops + [np.zeros(samples)])
    speech = np.concatenate([np.zeros(samples)] + speech_gops)
    upper *= float(aetv_rms) / max(float(np.sqrt(np.mean(upper[: -samples] ** 2))), 1e-12)
    speech *= float(voice_rms) / max(float(np.sqrt(np.mean(speech[samples:] ** 2))), 1e-12)
    composite = upper + speech
    scale = float(peak) / max(float(np.max(np.abs(composite))), float(peak))
    return composite * scale, speech * scale, upper * scale
