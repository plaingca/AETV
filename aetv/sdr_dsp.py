"""Causal real-modem to hardware-IQ conversion for bounded live streaming."""

import math

import numpy as np
from scipy.signal import firwin, lfilter, upfirdn, welch


class IQDecimator:
    """Reduce wideband hardware IQ before calibration, keeping FIR/phase state."""

    def __init__(self, factor=10):
        self.factor = factor
        self.taps = firwin(20 * factor + 1, 0.8 / factor)
        self.history = np.zeros(len(self.taps) - 1, np.complex64)
        self.input_count = 0

    def feed(self, iq):
        if not len(iq):
            return np.empty(0, np.complex64)
        joined = np.concatenate((self.history, iq))
        origin = self.input_count - len(self.history)
        phase = (-origin) % self.factor
        filtered = upfirdn(self.taps, joined[phase:], down=self.factor)
        indices = origin + phase + np.arange(len(filtered)) * self.factor
        result = filtered[(indices >= self.input_count) &
                          (indices < self.input_count + len(iq))]
        self.history = joined[-len(self.history):].copy()
        self.input_count += len(iq)
        return result.astype(np.complex64)


class ModemToIQ:
    """Select the analytic sideband, interpolate and translate with retained state.

    AC16's occupied positive-real spectrum has guard space around 0 and 16 kHz.
    Mixing by -8 kHz and filtering at 8 kHz rejects its negative-frequency image.
    sqrt(2) preserves the real-modem power in complex IQ. This causal adapter
    adds 522/48000 seconds of FIR delay without changing the wire geometry.
    """

    def __init__(self, sample_rate=2400000, offset_hz=100000, center_hz=8000):
        if (
            sample_rate % 48000
            or sample_rate <= 48000
            or abs(offset_hz) + 8000 >= sample_rate / 2
        ):
            raise ValueError("Invalid hardware sample rate or LO offset")
        self.sample_rate, self.offset_hz = sample_rate, offset_hz
        self.center_hz = center_hz
        self.factor = sample_rate // 48000
        self.sideband_taps = firwin(1025, center_hz, fs=48000)
        self.sideband_state = np.zeros(1024, np.complex128)
        self.interpolation_taps = (
            firwin(20 * self.factor + 1, 1 / self.factor, window=("kaiser", 5.0))
            * self.factor
        )
        self.history = np.zeros(20, np.complex128)
        self.input_count = 0

    def feed(self, audio):
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError("Expected finite one-dimensional modem audio")
        if not len(audio):
            return np.empty(0, np.complex64)
        positions = self.input_count + np.arange(len(audio))
        shifted = (
            np.sqrt(2)
            * audio
            * np.exp(
                -2j * np.pi * np.remainder(positions * (self.center_hz / 48000), 1)
            )
        )
        low, self.sideband_state = lfilter(
            self.sideband_taps, [1.0], shifted, zi=self.sideband_state
        )
        joined = np.concatenate((self.history, low))
        interpolated = upfirdn(self.interpolation_taps, joined, up=self.factor)
        begin = len(self.history) * self.factor
        result = interpolated[begin : begin + len(audio) * self.factor]
        self.history = joined[-20:].copy()
        radio_positions = self.input_count * self.factor + np.arange(len(result))
        result *= np.exp(
            2j
            * np.pi
            * np.remainder(radio_positions * (self.offset_hz / self.sample_rate), 1)
        )
        self.input_count += len(audio)
        if max(abs(result.real).max(), abs(result.imag).max()) >= 1:
            raise ValueError("IQ exceeds DAC component range; reduce explicit TX RMS")
        return result.astype(np.complex64)


def estimate_signal_offset(iq, sample_rate=960000, nominal_hz=-100000):
    """Estimate a strong AC16 signal's center from received spectrum only.

    This is coarse radio calibration, before ordinary modem acquisition. The
    expected bandwidth is public waveform geometry; no TX pixels/latents or
    known payload timing are inputs. Reject noise and ambiguous broad spectra.
    """
    iq = np.asarray(iq)
    if len(iq) < 65536:
        raise ValueError("At least 65536 radio samples required for calibration")
    f, power = welch(iq, fs=sample_rate, nperseg=65536, return_onesided=False)
    order = np.argsort(f)
    f, power = f[order], power[order]
    selected = abs(f - nominal_hz) < 25000
    f, power = f[selected], power[selected]
    noise = np.median(power[abs(f - nominal_hz) > 18000])
    excess = np.maximum(power - noise, 0)
    if excess.sum() <= 0:
        raise ValueError("No signal energy for frequency calibration")
    cumulative = np.cumsum(excess) / excess.sum()
    low, high = np.interp([0.005, 0.995], cumulative, f)
    width = high - low
    if not 13000 <= width <= 17000:
        raise ValueError(
            f"Spectrum is not an isolated AC16 signal: OBW99={width:.0f} Hz"
        )
    center = (low + high) / 2
    snr = 10 * np.log10(np.mean(power[abs(f - center) < 7500]) / max(noise, 1e-20))
    if snr < 6:
        raise ValueError("Signal too weak for spectral frequency calibration")
    return dict(
        offset_hz=float(round(center / 50) * 50),
        spectral_center_hz=float(center),
        obw99_hz=float(width),
        inband_over_noise_db=float(snr),
        method="Received-only noise-subtracted 99-percent spectral edges, rounded to 50 Hz",
    )


class IQToModem:
    """Integer polyphase decimation with persistent FIR and oscillator state."""

    def __init__(self, sample_rate=960000, signal_offset_hz=-100000, center_hz=8000):
        if sample_rate < 48000 or sample_rate % 48000:
            raise ValueError("Radio rate must be an integer multiple of 48000")
        if abs(signal_offset_hz) + 8000 >= sample_rate / 2:
            raise ValueError("Signal falls outside the sampled band")
        self.sample_rate = sample_rate
        self.offset_hz = signal_offset_hz
        self.center_hz = center_hz
        self.factor = sample_rate // 48000
        self.taps = firwin(20 * self.factor + 1, 12000, fs=sample_rate)
        self.history = np.zeros(len(self.taps) - 1, np.complex128)
        self.input_count = 0
        self.output_count = 0

    def feed(self, iq):
        iq = np.asarray(iq, dtype=np.complex128).reshape(-1)
        if not len(iq):
            return np.zeros(0, np.float32)
        positions = self.input_count + np.arange(len(iq))
        shifted = iq * np.exp(
            -2j * np.pi * self.offset_hz * positions / self.sample_rate
        )
        joined = np.concatenate((self.history, shifted))
        origin = self.input_count - len(self.history)
        phase = (-origin) % self.factor
        filtered = upfirdn(self.taps, joined[phase:], down=self.factor)
        indices = origin + phase + np.arange(len(filtered)) * self.factor
        low = filtered[
            (indices >= self.input_count) & (indices < self.input_count + len(iq))
        ]
        self.history = joined[-len(self.history) :].copy()
        self.input_count += len(iq)
        times = self.output_count + np.arange(len(low))
        audio = (
            math.sqrt(2)
            * (low * np.exp(2j * np.pi * self.center_hz * times / 48000)).real
        )
        self.output_count += len(low)
        return audio.astype(np.float32)
