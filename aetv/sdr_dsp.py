"""Causal real-modem to hardware-IQ conversion for bounded live streaming."""

import math

import numpy as np
from scipy.signal import firwin, lfilter, upfirdn, welch
from scipy.ndimage import median_filter


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


def _oscillator(start, count, frequency_hz, sample_rate):
    """exp(2j*pi*f*n/fs) for n in [start, start+count), tiled when periodic.

    A complex exp per sample dominated 9.6 MS/s HackRF conversion. Integer
    offsets repeat after fs/gcd(f, fs) samples, so a short exact table suffices.
    """
    if float(frequency_hz).is_integer():
        period = sample_rate // math.gcd(abs(int(frequency_hz)), sample_rate)
        if period <= 1 << 16:
            table = np.exp(2j * np.pi * np.remainder(
                np.arange(period) * (frequency_hz / sample_rate), 1))
            return np.resize(np.roll(table, -(start % period)), count)
    positions = start + np.arange(count)
    return np.exp(2j * np.pi * np.remainder(positions * (frequency_hz / sample_rate), 1))


class ModemToIQ:
    """Select the analytic sideband, interpolate and translate with retained state.

    AC16's occupied positive-real spectrum has guard space around 0 and 16 kHz.
    Mixing by -8 kHz and filtering at 8 kHz rejects its negative-frequency image.
    sqrt(2) preserves the real-modem power in complex IQ. This causal adapter
    adds 522/48000 seconds of FIR delay without changing the wire geometry.
    AC16 A/V instead uses a 10 kHz center for its complete 20 kHz composite.

    Rates above 960 kHz that are multiples of it (HackRF's 9.6 MS/s)
    interpolate to 960 kHz with the same filter, then use a short image-reject
    stage: one 200x FIR cost more than real time. That stage adds
    30 output samples of delay.
    """

    def __init__(self, sample_rate=2400000, offset_hz=100000, center_hz=8000,
                 *, peak_limit=None):
        if (
            sample_rate % 48000
            or sample_rate <= 48000
            or not 0 < center_hz < 24000
            or abs(offset_hz) + center_hz >= sample_rate / 2
        ):
            raise ValueError("Invalid hardware sample rate or LO offset")
        if peak_limit is not None and not 0 < peak_limit < 1:
            raise ValueError("IQ peak limit must be between zero and one")
        self.peak_limit = peak_limit
        self.headroom_gain = 1.0
        self.sample_rate, self.offset_hz = sample_rate, offset_hz
        self.center_hz = center_hz
        self.factor = sample_rate // 48000
        self.sideband_taps = firwin(1025, center_hz, fs=48000)
        self.sideband_state = np.zeros(1024, np.complex128)
        self.image_factor = 1
        if sample_rate > 960000 and sample_rate % 960000 == 0:
            self.image_factor = sample_rate // 960000
        first = self.factor // self.image_factor
        self.interpolation_taps = (
            firwin(20 * first + 1, 1 / first, window=("kaiser", 5.0)) * first
        )
        self.history = np.zeros(20, np.complex128)
        if self.image_factor > 1:
            self.image_taps = firwin(
                6 * self.image_factor + 1, 1 / self.image_factor, window=("kaiser", 8.0)
            ) * self.image_factor
            self.image_history = np.zeros(6, np.complex128)
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
        result = self._interpolate(low, "history", self.interpolation_taps,
                                   self.factor // self.image_factor)
        if self.image_factor > 1:
            result = self._interpolate(result, "image_history", self.image_taps,
                                       self.image_factor)
        result *= _oscillator(self.input_count * self.factor, len(result),
                              self.offset_hz, self.sample_rate)
        self.input_count += len(audio)
        if self.peak_limit is not None:
            # GUI transports submit complete GOP-sized blocks, then split the
            # bounded output into USB buffers. Back off the entire complex
            # waveform uniformly; component clipping would corrupt OFDM and
            # produce out-of-band energy. Hold acquired headroom through the
            # rest of the transmission instead of pumping gain during speech.
            peak = float(np.max(np.abs(result)))
            if peak > 0:
                self.headroom_gain = min(self.headroom_gain, self.peak_limit / peak)
            result *= self.headroom_gain
        if max(abs(result.real).max(), abs(result.imag).max()) >= 1:
            raise ValueError("IQ exceeds DAC component range; reduce explicit TX RMS")
        return result.astype(np.complex64)

    def _interpolate(self, values, history_name, taps, factor):
        history = getattr(self, history_name)
        joined = np.concatenate((history, values))
        interpolated = upfirdn(taps, joined, up=factor)
        begin = len(history) * factor
        setattr(self, history_name, joined[-len(history):].copy())
        return interpolated[begin : begin + len(values) * factor]


def estimate_signal_offset(iq, sample_rate=960000, nominal_hz=-100000, *, composite=False):
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
    if composite:
        return _composite_video_offset(f, power, nominal_hz)
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


def _composite_video_offset(f, power, nominal_hz):
    """Find the broad video slice independently of speech level or silence.

    AC16 A/V places the unchanged carrier bank 2 kHz above the RF center.
    Median filtering removes narrow audio tones. The 3.3–4.2 kHz guard
    separates even broadband speech from the video plateau. Whole-composite
    power percentiles would incorrectly move the dial as people speak.
    """
    selected = abs(f - nominal_hz) < 45000
    f, power = f[selected], power[selected]
    noise = float(np.median(power[abs(f - nominal_hz) > 37000]))
    step = float(f[1] - f[0])
    smooth = median_filter(np.maximum(power - noise, 0), size=2 * round(150 / step) + 1)
    # A broad 8 kHz window fits wholly inside the video bank but cannot fit
    # inside the 3.3 kHz audio band. This reference survives audio-dominant
    # power settings without treating a speech peak as the video threshold.
    windows = np.lib.stride_tricks.sliding_window_view(smooth, max(3, round(8000 / step)))
    level = float(np.percentile(windows, 25, axis=-1).max())
    if level < 6 * max(noise, 1e-20):
        raise ValueError("Signal too weak for A/V frequency calibration")
    threshold = .2 * level
    occupied = smooth > threshold
    edges = np.diff(np.r_[False, occupied, False].astype(int))
    candidates = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or stop == len(f):
            continue
        low = np.interp(threshold, smooth[start-1:start+1], f[start-1:start+1])
        high = np.interp(threshold, smooth[stop-1:stop+1][::-1], f[stop-1:stop+1][::-1])
        width = high - low
        center = (low + high) / 2 - 2000
        if 14400 <= width <= 15600 and abs(center - nominal_hz) <= 25000:
            candidates.append((center, width, float(np.mean(power[start:stop]))))
    if len(candidates) != 1:
        raise ValueError("No isolated AC16 A/V video plateau for frequency calibration")
    center, width, inband = candidates[0]
    return dict(offset_hz=float(round(center / 50) * 50),
                spectral_center_hz=float(center), video_width_hz=float(width),
                inband_over_noise_db=float(10 * np.log10(inband / max(noise, 1e-20))),
                method="Received-only AC16 A/V video edges; speech excluded; rounded to 50 Hz")


def estimate_weak_signal_offset(iq, sample_rate=960000, nominal_hz=-100000, *, composite=False):
    """Locate a weak video bank without integrating rectified noise tails.

    The strong-signal OBW estimator integrates positive noise fluctuations
    over the entire search band, making its estimated width approach 50 kHz
    as SNR falls. Here signed, noise-relative power in both known band edges
    is compared with adjacent guard space. Four independent interior slices
    must also contain broadband energy. This is only coarse frequency setup;
    the modem still authenticates timing, mode, beacon CRC and payload pilots.
    """
    if len(iq) < sample_rate:
        raise ValueError("Weak calibration needs one second of IQ")
    f, power = welch(iq, fs=sample_rate, nperseg=16384, return_onesided=False)
    order = np.argsort(f)
    f, power = f[order], power[order]
    selected = abs(f - nominal_hz) < 45000
    f, power = f[selected], power[selected]
    noise = float(np.median(power[abs(f - nominal_hz) > 37000]))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("No finite calibration noise reference")
    smooth = median_filter(power, size=7) / noise - 1
    sums = np.r_[0., np.cumsum(smooth)]
    centers = nominal_hz + np.arange(-25000, 25001, 50) + (2000 if composite else 0)

    def mean(low, high):
        left = np.searchsorted(f, centers + low)
        right = np.searchsorted(f, centers + high)
        return (sums[right] - sums[left]) / np.maximum(right - left, 1)

    inner = (mean(-7300, -5700) + mean(5700, 7300)) / 2
    outer = mean(7800, 9400)
    if not composite:
        outer = (outer + mean(-9400, -7800)) / 2
    scores = inner - outer
    index = int(np.argmax(scores))
    slices = np.array([mean(x, x + 3000)[index] for x in (-6500, -3000, 500, 3500)])
    if scores[index] < .15 or np.min(slices) < .12:
        raise ValueError("No independently occupied weak AC16 video bank")
    # Reject an apparent bank dominated by a narrow interferer or one edge.
    if np.max(slices) > 4 * np.min(slices):
        raise ValueError("Weak calibration spectrum is not broadband video")
    center = centers[index] - (2000 if composite else 0)
    return dict(offset_hz=float(center), video_width_hz=15000.,
                inband_over_noise_db=float(10*np.log10(1 + np.mean(slices))),
                method="Received-only weak video edge contrast and four interior slices")


def estimate_mode_signal_offset(iq, mode, sample_rate=960000, nominal_hz=-100000,
                                *, composite=False):
    """Coarse received-only tuning for the released narrow/wide carrier banks.

    Use signed noise-relative energy, four occupied interior slices, and
    adjacent guard contrast. A speech tone cannot stand in for a video bank.
    This only selects the radio passband; modem framing must still succeed.
    """
    from .analog_av import composite_profile
    from .config import RS

    if len(iq) < sample_rate // 2:
        raise ValueError("Coarse tuning needs half a second of IQ")
    f, power = welch(iq, fs=sample_rate, nperseg=32768, return_onesided=False)
    order = np.argsort(f)
    f, power = f[order], power[order]
    selected = abs(f - nominal_hz) < 45000
    f, power = f[selected], power[selected]
    noise = float(np.median(power[abs(f - nominal_hz) > 37000]))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("No finite calibration noise reference")
    smooth = median_filter(power, size=5) / noise - 1
    sums = np.r_[0., np.cumsum(smooth)]
    geometry = mode.geometry
    width = geometry.carriers * RS
    video_center = geometry.carrier0_hz + (geometry.carriers - 1) * RS / 2
    if composite:
        profile = composite_profile(mode.name)
        video_center += profile.shift_hz - profile.bandwidth_hz / 2
    else:
        video_center -= geometry.fcenter_hz
    centers = nominal_hz + np.arange(-25000, 25001, 25) + video_center

    def mean(low, high):
        left = np.searchsorted(f, centers + low)
        right = np.searchsorted(f, centers + high)
        return (sums[right] - sums[left]) / np.maximum(right - left, 1)

    slices = np.array([mean(x * width, (x + .2) * width) for x in (-.4, -.2, 0, .2)])
    outside = mean(.6 * width, .9 * width)
    if not composite:
        outside = (outside + mean(-.9 * width, -.6 * width)) / 2
    scores = slices.mean(axis=0) - outside
    occupied = (slices.min(axis=0) > .12) & (slices.max(axis=0) < 6 * slices.min(axis=0))
    scores[~occupied] = -np.inf
    index = int(np.argmax(scores))
    if scores[index] < .15:
        raise ValueError(f"No independently occupied {mode.name} video bank")
    if index < 2 or index >= len(centers) - 2:
        raise ValueError("Calibration maximum lies at the search boundary")
    if scores[index] < .35 * slices[:, index].mean():
        raise ValueError("Calibration energy lacks distinct video-band edges")
    center = centers[index] - video_center
    # Strong signals permit a tighter edge estimate. The broad contrast score
    # alone can favor one side of an uneven bank by a few hundred hertz, which
    # the video modem tolerates but would cut into low-frequency A/V speech.
    inside = abs(f - centers[index]) < .35 * width
    level = float(np.percentile(smooth[inside], 25))
    if level > 5:
        threshold = .15 * level
        occupied_bins = smooth > threshold
        middle = int(np.searchsorted(f, centers[index]))
        left = right = middle
        while left > 0 and occupied_bins[left]:
            left -= 1
        while right < len(f) - 1 and occupied_bins[right]:
            right += 1
        if left < middle < right:
            low = np.interp(threshold, smooth[left:left+2], f[left:left+2])
            high = np.interp(threshold, smooth[right-1:right+1][::-1], f[right-1:right+1][::-1])
            if .8 * width <= high - low <= 1.2 * width:
                center = round(((low + high) / 2 - video_center) / 25) * 25
    else:
        # At low SNR the broad interior score has a flat maximum. An uneven
        # payload can move it hundreds of hertz without changing that score
        # much. This matters for late entry: the V8 A/V beacon sits at the
        # upper filter edge and can disappear before payload pilots do.
        # Refine against both *local* edges, retaining independent occupancy
        # checks and requiring positive evidence at each edge. Speech below
        # the guard cannot satisfy the upper-edge check on its own.
        lower = mean(-.5 * width, -.4 * width) - mean(-.6 * width, -.5 * width)
        upper = mean(.4 * width, .5 * width) - mean(.5 * width, .6 * width)
        edge_scores = (lower + upper) / 2
        plausible = (
            occupied & (abs(centers - centers[index]) <= .25 * width)
            & (lower > .12) & (upper > .12)
        )
        edge_scores[~plausible] = -np.inf
        refined = int(np.argmax(edge_scores))
        if np.isfinite(edge_scores[refined]):
            center = centers[refined] - video_center
    return dict(offset_hz=float(center), video_width_hz=float(width),
                inband_over_noise_db=float(10*np.log10(1 + slices[:, index].mean())),
                method=f"Received-only {mode.name} carrier-bank contrast and four interior slices")


class IQToModem:
    """Integer polyphase decimation with persistent FIR and oscillator state."""

    def __init__(self, sample_rate=960000, signal_offset_hz=-100000, center_hz=8000):
        if sample_rate < 48000 or sample_rate % 48000:
            raise ValueError("Radio rate must be an integer multiple of 48000")
        if not 0 < center_hz < 24000 or abs(signal_offset_hz) + center_hz >= sample_rate / 2:
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
