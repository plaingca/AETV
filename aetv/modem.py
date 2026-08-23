"""Top-level AETV modem: latent vector / GOP streams <-> passband audio samples.

Handles:
- Transmit modulation: preamble | header | continuous GOPs (interleaved latents + beacon chips).
- Envelope clipping (0.5 dB headroom) and post-clip bandpass filtering.
- Demodulation: baseband conversion, preamble sync / blind sync, pilot equalization,
  drift tracking loop, confidence weighting, beacon decode, and GOP de-interleaving.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from math import gcd
import time

import numpy as np
from scipy import signal

from . import golay
from . import beacon, framing, ofdm
from .beacon import AETVBeaconResult, find_beacon_superframe, generate_beacon_chips
from .config import (
    AETV_MODES,
    AETV_MODES_BY_INDEX,
    AETVModeSpec,
    BANDS,
    CLIP_HEADROOM_DB,
    DATA_SYMS_PER_FRAME,
    DEMOD_BACKOFF,
    DRIFT_FAST_ALPHA,
    DRIFT_FAST_BETA,
    DRIFT_SLOW_ALPHA,
    DRIFT_SLOW_BETA,
    FRAME_SAMPLES,
    FRAMES_PER_GOP,
    FS,
    GOP_SAMPLES,
    HEADER_SAMPLES,
    HEADER_SYMS,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    M,
    NCP,
    NSYM,
    PREAMBLE_CP,
    PREAMBLE_REPEATS,
    PREAMBLE_SAMPLES,
    PROTOCOL_VERSION,

    RS,
    SNR_REF_BW_HZ,
    SYMS_PER_FRAME,
)
from .sync import Acquisition, SyncError, acquire, freq_correct


@dataclass
class AETVDemodResult:
    gops_latents: list[np.ndarray]  # list of GOP latent arrays, each of shape (latents_per_gop,)
    gops_weights: list[np.ndarray]  # list of confidence weights
    mode: AETVModeSpec
    freq_offset: float
    sync_metric: float
    frames_received: int
    beacon: AETVBeaconResult | None = None
    callsign: str = ""
    preamble_start: int = 0
    snr_db: float = float("nan")
    beacon_chips: np.ndarray | None = None
    beacon_repeated_chips: np.ndarray | None = None
    header_score: float = 0.0
    pilot_coherence: float = 0.0


@dataclass(frozen=True)
class BlindPayloadAcquisition:
    payload_start: int
    freq_offset: float
    metric: float
    beacon: AETVBeaconResult


def to_baseband(x: np.ndarray, fcenter_hz: int, fs: int = FS) -> np.ndarray:
    """Heterodyne real passband signal to complex baseband centered at fcenter_hz."""
    g = gcd(fcenter_hz, fs)
    period = fs // g
    step = fcenter_hz // g
    table = np.exp(-2j * np.pi * np.arange(period) / period)
    n = np.arange(len(x))
    return x.astype(np.float64) * table[(step * n) % period]


def freq_correct(z: np.ndarray, f_hz: float, fs: int = FS) -> np.ndarray:
    n = np.arange(len(z))
    cycles = (f_hz * n / fs) % 1.0
    return z * np.exp(-2j * np.pi * cycles)


def clip_envelope_and_filter(
    x: np.ndarray,
    clip_headroom_db: float,
    bandpass_hz: tuple[float, float],
    fs: int = FS,
    iterations: int = 2,
) -> np.ndarray:
    """Envelope clip-and-filter for PAPR control."""
    power = np.mean(x**2)
    if power == 0:
        return x
    thresh = np.sqrt(2.0 * power) * 10.0 ** (clip_headroom_db / 20.0)
    taps = signal.firwin(201, bandpass_hz, fs=fs, pass_zero=False)
    for _ in range(iterations):
        z = signal.hilbert(x)
        mag = np.abs(z)
        scale = np.minimum(1.0, thresh / np.maximum(mag, 1e-12))
        x = np.real(z * scale)
        x = signal.convolve(x, taps, mode="same")
    rms = np.sqrt(np.mean(x**2))
    return x / max(1e-9, rms)


def encode_header(mode_index: int, version: int = PROTOCOL_VERSION) -> np.ndarray:
    """Pack mode index (4 bits) and version (4 bits) into 24-bit Golay codeword, BPSK."""
    payload_val = ((version & 0x0F) << 4) | (mode_index & 0x0F)
    coded = golay.codeword_bits(payload_val)
    return 1.0 - 2.0 * coded.astype(np.float64)


def decode_header(soft_symbols: np.ndarray) -> tuple[int, int] | None:
    """Decode Golay header; returns (mode_index, version) or None."""
    if len(soft_symbols) < 24:
        return None
    decoded_val = golay.decode_soft(soft_symbols[:24])
    version = (decoded_val >> 4) & 0x0F
    mode_index = decoded_val & 0x0F
    return mode_index, version


def _header_carriers(header_chips: np.ndarray, carriers: int) -> np.ndarray:
    """Repeat the Golay word across the header bandwidth for diversity.

    The first 24 carriers remain byte-for-byte compatible with protocol-v3
    receivers. Previously unused header carriers now carry repetitions.
    """
    return np.resize(np.asarray(header_chips), carriers)


def _header_candidates(header_soft: np.ndarray) -> list[np.ndarray]:
    """Return legacy and frequency-diversity soft header observations."""
    legacy = np.asarray(header_soft[:24], dtype=np.float64)
    groups = [
        np.asarray(header_soft[start : start + 24], dtype=np.float64)
        for start in range(0, len(header_soft) - 23, 24)
    ]
    if len(groups) < 2:
        return [legacy]
    # Ignore the short final repetition so every bit has equal weight.
    combined = np.mean(np.stack(groups), axis=0)
    return [legacy, combined]


def _band_params(band: str):
    geom = BANDS[band]
    fs = geom.fs
    m = fs // RS
    ncp = m // 4
    nsym = m + ncp
    frame_samples = SYMS_PER_FRAME * nsym
    preamble_cp = 2 * ncp
    preamble_samples = preamble_cp + PREAMBLE_REPEATS * m
    header_samples = HEADER_SYMS * nsym
    leadin_samples = int(0.1 * fs)
    leadout_samples = int(0.1 * fs)
    return geom, fs, m, ncp, nsym, frame_samples, preamble_samples, header_samples, leadin_samples, leadout_samples


def _payload_wave(
    latents: np.ndarray,
    beacon_chips: np.ndarray,
    mode: AETVModeSpec,
    interleave: bool,
) -> np.ndarray:
    """Modulate exactly one GOP payload, with no acquisition or silence."""
    geom = mode.geometry
    data_syms = framing.pack_gop_symbols(
        latents=latents,
        beacon_chips=beacon_chips,
        band=mode.band,
        interleave=interleave,
    )
    frame_syms = np.zeros(
        (FRAMES_PER_GOP * SYMS_PER_FRAME, geom.carriers), dtype=np.complex64
    )
    pilot = ofdm.pilot_sequence(mode.band)
    for frame in range(FRAMES_PER_GOP):
        frame_syms[frame * SYMS_PER_FRAME] = pilot
        frame_syms[frame * SYMS_PER_FRAME + 1 : (frame + 1) * SYMS_PER_FRAME] = data_syms[
            frame * DATA_SYMS_PER_FRAME : (frame + 1) * DATA_SYMS_PER_FRAME
        ]
    wave = ofdm.modulate_symbols(frame_syms, mode.band)
    assert len(wave) == mode.geometry.fs
    return wave


def _acquisition_wave(mode: AETVModeSpec) -> np.ndarray:
    """Return preamble + repeated Golay header, without lead silence."""
    geom = mode.geometry
    header_chips = encode_header(mode.index, PROTOCOL_VERSION)
    header_syms = np.zeros((HEADER_SYMS, geom.carriers), dtype=np.complex64)
    header_syms[:] = _header_carriers(header_chips, geom.carriers)
    return np.concatenate(
        [ofdm.preamble_waveform(mode.band), ofdm.modulate_symbols(header_syms, mode.band)]
    )


class _ContinuousTxConditioner:
    """Stateful clip/filter stage that preserves waveform continuity."""

    def __init__(self, mode: AETVModeSpec, iterations: int = 2):
        self.fs = mode.geometry.fs
        self.taps = signal.firwin(201, mode.geometry.tx_bandpass, fs=self.fs, pass_zero=False)
        self.states = [np.zeros(len(self.taps) - 1) for _ in range(iterations)]

    def process(self, values: np.ndarray) -> np.ndarray:
        y = np.asarray(values, dtype=np.float64)
        input_power = float(np.mean(y**2)) if y.size else 0.0
        for index, state in enumerate(self.states):
            power = float(np.mean(y**2)) if y.size else 0.0
            if power > 0.0:
                threshold = np.sqrt(2.0 * power) * 10.0 ** (CLIP_HEADROOM_DB / 20.0)
                analytic = signal.hilbert(y)
                scale = np.minimum(1.0, threshold / np.maximum(np.abs(analytic), 1e-12))
                y = np.real(analytic * scale)
            y, self.states[index] = signal.lfilter(
                self.taps, [1.0], y, zi=state
            )
        if input_power > 0.0:
            rms = float(np.sqrt(np.mean(y**2)))
            y = y / max(rms, 1e-9)
        return y.astype(np.float32)


def _header_aided_acquisitions(
    audio: np.ndarray,
    band: str,
    *,
    max_candidates: int = 8,
) -> list[Acquisition]:
    """Find startup timing by jointly testing CP phase and the mode header."""
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    geom, fs, m, ncp, nsym, _frame_samples, preamble_samples, header_samples, _leadin, _leadout = _band_params(band)
    minimum = preamble_samples + header_samples + FRAMES_PER_GOP * (
        SYMS_PER_FRAME * nsym
    )
    if len(values) < minimum:
        return []
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-12:
        return []
    z = to_baseband(values / peak, geom.fcenter_hz, fs)
    product = z[:-m] * np.conj(z[m:])
    kernel = np.ones(ncp)
    correlation = signal.fftconvolve(product, kernel, mode="valid")
    e1 = signal.fftconvolve(np.abs(z[:-m]) ** 2, kernel, mode="valid")
    e2 = signal.fftconvolve(np.abs(z[m:]) ** 2, kernel, mode="valid")
    cp_metric = np.abs(correlation) / np.maximum(
        np.sqrt(np.maximum(e1, 0.0) * np.maximum(e2, 0.0)), 1e-12
    )
    phase_scores = np.array(
        [np.mean(cp_metric[offset::nsym]) for offset in range(nsym)]
    )
    symbol_offset = int(np.argmax(phase_scores))
    timing_metric = float(phase_scores[symbol_offset])
    # The header carries only a compact mode/version code, so a chance Golay
    # match is not sufficient evidence by itself. Require an OFDM-wide CP
    # structure as well; the OTA case that motivated this fallback measured
    # about 0.29 here, while pre-transmission noise produced false headers near
    # 0.11.
    if timing_metric < 0.20:
        return []
    cp_samples = correlation[symbol_offset::nsym]
    fractional_cfo = (
        float(np.angle(np.mean(cp_samples)) / (2.0 * np.pi * (m / fs)))
        if cp_samples.size
        else 0.0
    )

    pilot = ofdm.pilot_sequence(band)
    preamble_useful = 2 * ncp + (PREAMBLE_REPEATS - 1) * m
    latest_start = len(values) - minimum
    first_start = (symbol_offset - preamble_samples) % nsym
    scored: list[tuple[float, Acquisition]] = []
    timing_step = max(1, ncp // 4)
    for coarse_start in range(first_start, latest_start + 1, nsym):
        for timing_delta in range(-m // 2, m // 2 + 1, timing_step):
            start = coarse_start + timing_delta
            if start < 0 or start > latest_start:
                continue
            for integer_bin in range(-5, 6):
                cfo = fractional_cfo + integer_bin * RS
                segment = freq_correct(
                    z[start : start + preamble_samples + header_samples],
                    cfo,
                    fs=fs,
                )
                received_preamble = ofdm.demod_window(
                    segment, preamble_useful, band=band
                )
                channel = received_preamble / pilot
                denominator = np.abs(channel) ** 2 + 1e-4
                header_symbols = []
                for symbol in range(HEADER_SYMS):
                    useful_start = preamble_samples + symbol * nsym + ncp
                    received = ofdm.demod_window(segment, useful_start, band=band)
                    header_symbols.append(
                        received * np.conj(channel) / denominator
                    )
                first_real = np.real(header_symbols[0])
                second_real = np.real(header_symbols[1])
                repetition_coherence = float(
                    np.dot(first_real, second_real)
                    / max(
                        np.linalg.norm(first_real)
                        * np.linalg.norm(second_real),
                        1e-12,
                    )
                )
                if repetition_coherence < 0.10:
                    continue
                header_carriers = np.real(np.mean(header_symbols, axis=0))
                for observed in _header_candidates(header_carriers):
                    mode_index, version = decode_header(observed)
                    mode = AETV_MODES_BY_INDEX.get(mode_index)
                    if (
                        version != PROTOCOL_VERSION
                        or mode is None
                        or mode.band != band
                    ):
                        continue
                    expected = encode_header(mode_index, version)
                    score = float(
                        np.dot(observed, expected)
                        / max(
                            np.linalg.norm(observed) * np.linalg.norm(expected),
                            1e-12,
                        )
                    )
                    repetition_support = max(
                        float(
                            np.dot(group, expected)
                            / max(
                                np.linalg.norm(group)
                                * np.linalg.norm(expected),
                                1e-12,
                            )
                        )
                        for group in (
                            header_carriers[start : start + 24]
                            for start in range(
                                0, len(header_carriers) - 23, 24
                            )
                        )
                    )
                    if score >= 0.45 and repetition_support >= 0.45:
                        scored.append(
                            (score, Acquisition(start, float(cfo), timing_metric))
                        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in scored[:max_candidates]]


def _ofdm_timing_metric(audio: np.ndarray, band: str) -> float:
    """Return cyclic-prefix phase coherence without requiring 12 seconds."""
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    geom, fs, m, ncp, nsym, *_rest = _band_params(band)
    if len(values) < m + ncp:
        return 0.0
    z = to_baseband(values, geom.fcenter_hz, fs)
    product = z[:-m] * np.conj(z[m:])
    kernel = np.ones(ncp)
    correlation = signal.fftconvolve(product, kernel, mode="valid")
    e1 = signal.fftconvolve(np.abs(z[:-m]) ** 2, kernel, mode="valid")
    e2 = signal.fftconvolve(np.abs(z[m:]) ** 2, kernel, mode="valid")
    metric = np.abs(correlation) / np.maximum(
        np.sqrt(np.maximum(e1, 0.0) * np.maximum(e2, 0.0)), 1e-12
    )
    return float(max(np.mean(metric[offset::nsym]) for offset in range(nsym)))


def modulate_gop_stream(
    gops: list[np.ndarray],
    mode_name: str = "V1",
    callsign: str = "N0CALL",
    start_frame: int = 0,
    interleave: bool = True,
) -> np.ndarray:
    """Modulate a sequence of GOP latents into an 8 kHz or 24 kHz real passband audio waveform."""
    mode = AETV_MODES[mode_name]
    band = mode.band
    geom, fs, m, ncp, nsym, frame_samples, preamble_samples, header_samples, leadin_samples, leadout_samples = _band_params(band)
    n_gops = len(gops)
    n_frames = n_gops * FRAMES_PER_GOP

    # Generate beacon chips for the entire stream
    beacon_chips = generate_beacon_chips(
        n_frames=n_frames,
        start_frame=start_frame,
        callsign=callsign,
        mode_index=mode.index,
    )

    # Modulate preamble
    preamble = ofdm.preamble_waveform(band)

    # Modulate header (2 identical BPSK symbols across carriers)
    header_chips = encode_header(mode.index, PROTOCOL_VERSION)
    header_syms = np.zeros((HEADER_SYMS, geom.carriers), dtype=np.complex64)
    for s in range(HEADER_SYMS):
        header_syms[s] = _header_carriers(header_chips, geom.carriers)
    header_wave = ofdm.modulate_symbols(header_syms, band)

    # Modulate GOP frames
    pilot = ofdm.pilot_sequence(band)
    gop_audio_chunks = []
    chip_offset = 0

    for gop_idx, gop_latents in enumerate(gops):
        chips_per_gop = FRAMES_PER_GOP * DATA_SYMS_PER_FRAME
        gop_chips = beacon_chips[chip_offset : chip_offset + chips_per_gop]
        chip_offset += len(gop_chips)

        # Pack (32 data syms, NC)
        data_syms = framing.pack_gop_symbols(
            latents=gop_latents,
            beacon_chips=gop_chips,
            band=band,
            interleave=interleave,
        )

        # Structure as 8 OFDM frames: each frame has 1 pilot + 4 data symbols
        frame_syms = np.zeros((FRAMES_PER_GOP * SYMS_PER_FRAME, geom.carriers), dtype=np.complex64)
        for f in range(FRAMES_PER_GOP):
            frame_syms[f * SYMS_PER_FRAME] = pilot
            frame_syms[f * SYMS_PER_FRAME + 1 : (f + 1) * SYMS_PER_FRAME] = data_syms[
                f * DATA_SYMS_PER_FRAME : (f + 1) * DATA_SYMS_PER_FRAME
            ]

        wave = ofdm.modulate_symbols(frame_syms, band)
        gop_audio_chunks.append(wave)

    raw_tx = np.concatenate(
        [
            np.zeros(leadin_samples, dtype=np.float64),
            preamble,
            header_wave,
            *gop_audio_chunks,
            np.zeros(leadout_samples, dtype=np.float64),
        ]
    )

    # Condition transmit audio (envelope clipping + bandpass)
    conditioned = clip_envelope_and_filter(
        raw_tx,
        clip_headroom_db=CLIP_HEADROOM_DB,
        bandpass_hz=geom.tx_bandpass,
        fs=fs,
    )
    return conditioned.astype(np.float32)


def modulate_gop_chunks(
    gops: Iterable[np.ndarray],
    mode_name: str = "V1",
    callsign: str = "N0CALL",
    start_frame: int = 0,
    interleave: bool = True,
):
    """Yield independently acquired GOP waveforms for live transmission.

    Each one-second video GOP gets its own preamble and mode header.  A station
    that tunes in late can therefore lock at the next GOP instead of needing
    the beginning of the transmission.  Beacon chips remain continuous across
    chunk boundaries so the normal callsign/counter beacon is recovered after
    enough received GOPs.
    """
    mode = AETV_MODES[mode_name]
    geom, fs, _m, _ncp, _nsym, _frame_samples, _preamble_samples, _header_samples, leadin_samples, leadout_samples = _band_params(mode.band)
    preamble = ofdm.preamble_waveform(mode.band)
    header_chips = encode_header(mode.index, PROTOCOL_VERSION)
    header_syms = np.zeros((HEADER_SYMS, geom.carriers), dtype=np.complex64)
    header_syms[:] = _header_carriers(header_chips, geom.carriers)
    header_wave = ofdm.modulate_symbols(header_syms, mode.band)
    pilot = ofdm.pilot_sequence(mode.band)

    for index, latents in enumerate(gops):
        chips_per_gop = FRAMES_PER_GOP * DATA_SYMS_PER_FRAME
        chip_start = index * chips_per_gop
        # Generate a window from one conceptual continuous beacon.  This keeps
        # the function lazy without resetting the beacon at every GOP.
        chips = generate_beacon_chips(
            n_frames=(index + 1) * FRAMES_PER_GOP,
            start_frame=start_frame,
            callsign=callsign,
            mode_index=mode.index,
        )[chip_start : chip_start + chips_per_gop]
        data_syms = framing.pack_gop_symbols(
            latents=latents, beacon_chips=chips, band=mode.band, interleave=interleave
        )
        frame_syms = np.zeros(
            (FRAMES_PER_GOP * SYMS_PER_FRAME, geom.carriers), dtype=np.complex64
        )
        for frame in range(FRAMES_PER_GOP):
            frame_syms[frame * SYMS_PER_FRAME] = pilot
            frame_syms[frame * SYMS_PER_FRAME + 1 : (frame + 1) * SYMS_PER_FRAME] = data_syms[
                frame * DATA_SYMS_PER_FRAME : (frame + 1) * DATA_SYMS_PER_FRAME
            ]
        raw = np.concatenate(
            [
                np.zeros(leadin_samples),
                preamble,
                header_wave,
                ofdm.modulate_symbols(frame_syms, mode.band),
                np.zeros(leadout_samples),
            ]
        )
        yield clip_envelope_and_filter(
            raw, CLIP_HEADROOM_DB, geom.tx_bandpass, fs=fs
        ).astype(np.float32)


def modulate_continuous_chunks(
    gops: Iterable[np.ndarray],
    mode_name: str = "V7",
    callsign: str = "N0CALL",
    start_frame: int = 0,
    interleave: bool = True,
):
    """Yield one acquisition followed by back-to-back one-second GOPs.

    Lead-in, preamble, and mode header occur once at stream start; lead-out
    occurs once at stream end. For N GOPs the total duration is N + 0.34 s in
    V7, rather than 1.34*N. Filtering retains state across yielded blocks.
    """
    mode = AETV_MODES[mode_name]
    fs = mode.geometry.fs
    lead = np.zeros(int(0.1 * fs), dtype=np.float64)
    acquisition = _acquisition_wave(mode)
    conditioner = _ContinuousTxConditioner(mode)
    iterator = iter(gops)
    try:
        current = next(iterator)
    except StopIteration:
        return

    index = 0
    while True:
        try:
            following = next(iterator)
            final = False
        except StopIteration:
            following = None
            final = True

        chip_start = index * FRAMES_PER_GOP * DATA_SYMS_PER_FRAME
        chips = generate_beacon_chips(
            n_frames=(index + 1) * FRAMES_PER_GOP,
            start_frame=start_frame,
            callsign=callsign,
            mode_index=mode.index,
        )[chip_start : chip_start + FRAMES_PER_GOP * DATA_SYMS_PER_FRAME]
        payload = _payload_wave(current, chips, mode, interleave)
        pieces = []
        if index == 0:
            pieces.extend((lead, acquisition))
        pieces.append(payload)
        if final:
            pieces.append(lead)
        yield conditioner.process(np.concatenate(pieces))

        if final:
            return
        current = following
        index += 1


class StreamingDemodulator:
    """Incremental GOP receiver with late-entry acquisition."""

    def __init__(
        self,
        band: str = "W",
        interleave: bool = True,
        on_debug=None,
        continuous: bool = False,
        mode_name: str | None = None,
    ):
        self.band = band
        self.interleave = interleave
        self.buffer = np.zeros(0, dtype=np.float32)
        self.beacon_chips = np.zeros(0, dtype=np.float64)
        self.beacon_repeated_chips = np.zeros(0, dtype=np.float64)
        self.last_beacon: AETVBeaconResult | None = None
        self._on_debug = on_debug or (lambda _event: None)
        self.samples_consumed = 0
        self._last_accepted_preamble: int | None = None
        self.continuous = bool(continuous)
        self._tracking_mode: AETVModeSpec | None = None
        self._tracking_freq_offset = 0.0
        self._tracking_bad_gops = 0
        self._tracking_pending: list[tuple[AETVDemodResult, int]] = []
        default_mode = {"N": "V0", "W": "V1", "U": "V7"}[band]
        self.expected_mode = AETV_MODES[mode_name or default_mode]
        if self.expected_mode.band != band:
            raise ValueError(
                f"mode {self.expected_mode.name} does not use band {band}"
            )
        self._last_blind_attempt_end = -10**18
        self._awaiting_blind = False
        self._pending_acquisition: Acquisition | None = None
        self._header_aided_allowed = False
        self._awaiting_search_offset = 0

    def _debug(self, event: str, **fields) -> None:
        try:
            self._on_debug({"event": event, "time": time.time(), **fields})
        except Exception:
            pass

    def _accumulate_beacon(
        self,
        result: AETVDemodResult,
        stream_sample: int,
        nominal_step: int,
    ) -> int:
        chips = result.beacon_chips
        repeated_chips = result.beacon_repeated_chips
        missing_gops = 0
        if self._last_accepted_preamble is not None:
            elapsed_gops = max(
                1,
                round(
                    (stream_sample - self._last_accepted_preamble)
                    / max(1, nominal_step)
                ),
            )
            missing_gops = max(0, elapsed_gops - 1)
        self._last_accepted_preamble = stream_sample
        if chips is not None and chips.size:
            if missing_gops:
                self.beacon_chips = np.concatenate(
                    [self.beacon_chips, np.zeros(missing_gops * len(chips))]
                )
            self.beacon_chips = np.concatenate([self.beacon_chips, chips])[
                -(3 * beacon.SUPERFRAME_LEN) :
            ]
            found = find_beacon_superframe(self.beacon_chips)
            if found is not None:
                self.last_beacon = found
        if repeated_chips is not None and repeated_chips.size:
            if missing_gops:
                self.beacon_repeated_chips = np.concatenate(
                    [
                        self.beacon_repeated_chips,
                        np.zeros(missing_gops * len(repeated_chips)),
                    ]
                )
            self.beacon_repeated_chips = np.concatenate(
                [self.beacon_repeated_chips, repeated_chips]
            )[-(3 * beacon.SUPERFRAME_LEN) :]
            found = find_beacon_superframe(self.beacon_repeated_chips)
            if found is not None:
                self.last_beacon = found
        if self.last_beacon is not None:
            result.beacon = self.last_beacon
            result.callsign = self.last_beacon.callsign
        return missing_gops

    def _attempt_blind_acquisition(self, fs: int) -> bool:
        blind_minimum = 12 * fs
        if not self.continuous or len(self.buffer) < blind_minimum:
            return False
        stream_end = self.samples_consumed + len(self.buffer)
        if stream_end - self._last_blind_attempt_end < fs:
            return False
        self._last_blind_attempt_end = stream_end
        try:
            blind = blind_acquire_continuous_payload(
                self.buffer[-blind_minimum:], self.expected_mode
            )
        except SyncError as error:
            self._debug("blind_candidate_rejected", reason=str(error))
            return False
        window_start = len(self.buffer) - blind_minimum
        discard = window_start + blind.payload_start
        self.buffer = self.buffer[discard:]
        self.samples_consumed += discard
        self._tracking_mode = self.expected_mode
        self._tracking_freq_offset = blind.freq_offset
        self._tracking_bad_gops = 0
        self._tracking_pending.clear()
        self.last_beacon = blind.beacon
        self._last_accepted_preamble = None
        self._awaiting_blind = False
        self._debug(
            "blind_acquired",
            stream_sample=int(self.samples_consumed),
            metric=float(blind.metric),
            freq_offset_hz=float(blind.freq_offset),
            callsign=blind.beacon.callsign,
        )
        return True

    def feed(self, audio: np.ndarray) -> list[AETVDemodResult]:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            self.buffer = np.concatenate([self.buffer, chunk])
        geom, fs, _m, _ncp, _nsym, frame_samples, preamble_samples, header_samples, _leadin, _leadout = _band_params(self.band)
        minimum = preamble_samples + header_samples + FRAMES_PER_GOP * frame_samples
        results: list[AETVDemodResult] = []
        while True:
            if self.continuous and self._tracking_mode is not None:
                payload_samples = self._tracking_mode.geometry.fs
                if len(self.buffer) < payload_samples:
                    break
                stream_sample = self.samples_consumed
                payload = self.buffer[:payload_samples]
                try:
                    result = demodulate_tracked_gop(
                        payload,
                        self._tracking_mode,
                        self._tracking_freq_offset,
                        interleave=self.interleave,
                    )
                except SyncError as error:
                    self.buffer = self.buffer[payload_samples:]
                    self.samples_consumed += payload_samples
                    self._tracking_bad_gops += 1
                    self._debug(
                        "tracking_weak",
                        stream_sample=int(stream_sample),
                        consecutive=int(self._tracking_bad_gops),
                        reason=str(error),
                    )
                    if self._tracking_bad_gops >= 3:
                        self._debug(
                            "tracking_lost",
                            stream_sample=int(stream_sample),
                            reason=f"three weak GOP intervals; last: {error}",
                        )
                        self._tracking_mode = None
                        self._header_aided_allowed = False
                        self._tracking_pending.clear()
                    continue
                self.buffer = self.buffer[payload_samples:]
                self.samples_consumed += payload_samples
                # A payload can be too marginal for the normal quality gate
                # while still being unmistakably on-air.  Do not use the
                # stricter 0.20 display-quality threshold to decide that the
                # transmitter disappeared: weak OTA fades commonly sit around
                # 0.15--0.19, while post-transmission noise clusters near the
                # eight-pilot incoherent baseline (0.125).  Hold only the truly
                # ambiguous intervals and release them if carrier evidence
                # returns before the three-GOP end detector expires.
                if result.pilot_coherence < 0.145:
                    self._tracking_bad_gops += 1
                    self._tracking_pending.append((result, stream_sample))
                    self._debug(
                        "tracking_weak",
                        stream_sample=int(stream_sample),
                        consecutive=int(self._tracking_bad_gops),
                        pilot_coherence=float(result.pilot_coherence),
                        reason="payload pilot coherence below 0.145 presence floor",
                    )
                    if self._tracking_bad_gops >= 3:
                        self._debug(
                            "tracking_lost",
                            stream_sample=int(stream_sample),
                            reason="three weak GOP intervals",
                        )
                        self._tracking_mode = None
                        self._header_aided_allowed = False
                        self._tracking_pending.clear()
                    continue
                self._tracking_bad_gops = 0
                recovered = [*self._tracking_pending, (result, stream_sample)]
                self._tracking_pending.clear()
                for recovered_result, recovered_sample in recovered:
                    missing_gops = self._accumulate_beacon(
                        recovered_result, recovered_sample, payload_samples
                    )
                    self._debug(
                        "gop_accepted",
                        tracked=True,
                        recovered_weak=(recovered_result is not result),
                        stream_sample=int(recovered_sample),
                        metric=float(recovered_result.sync_metric),
                        freq_offset_hz=float(recovered_result.freq_offset),
                        header_score=float(recovered_result.header_score),
                        pilot_coherence=float(recovered_result.pilot_coherence),
                        snr_db=float(recovered_result.snr_db),
                        beacon_chips=int(len(self.beacon_chips)),
                        beacon_repeated_chips=int(len(self.beacon_repeated_chips)),
                        missing_gops=int(missing_gops),
                        callsign=recovered_result.callsign,
                    )
                    results.append(recovered_result)
                continue
            if self.continuous and self._awaiting_blind:
                # The receiver commonly starts before the transmitter.  A
                # failed scan of that initial noise does not mean we joined a
                # preamble-less stream. Scan every newly completed overlapping
                # GOP window while retaining the full history needed by blind
                # acquisition. This must not depend on callback cadence: Kiwi
                # audio may arrive at the decoder in two-second batches.
                recent_length = minimum + fs // 4
                if len(self.buffer) >= minimum:
                    scan_step = fs // 4
                    # Do not retire an overlap until its full search margin is
                    # present; a candidate near the right edge may need that
                    # margin to contain the complete GOP.
                    latest_scan = len(self.buffer) - recent_length
                    while self._awaiting_search_offset <= latest_scan:
                        recent_start = self._awaiting_search_offset
                        recent = self.buffer[
                            recent_start : recent_start + recent_length
                        ]
                        recent_acq = None
                        expected_header_matched = False
                        try:
                            recent_acq = acquire(
                                to_baseband(recent, geom.fcenter_hz, fs),
                                band=self.band,
                            )
                            recent_needed = recent_acq.preamble_start + minimum
                            if len(recent) < recent_needed:
                                raise SyncError(
                                    "candidate falls beyond this overlapping window"
                                )
                            recent_result = demodulate_gop_stream(
                                recent[:recent_needed],
                                band=self.band,
                                drift_track="off",
                                interleave=self.interleave,
                                acquisition=recent_acq,
                            )
                            if recent_result.mode != self.expected_mode:
                                raise SyncError(
                                    f"expected {self.expected_mode.name}, got "
                                    f"{recent_result.mode.name}"
                                )
                            expected_header_matched = True
                            if recent_result.pilot_coherence < 0.20:
                                raise SyncError(
                                    "startup payload pilot coherence too low "
                                    f"({recent_result.pilot_coherence:.2f})"
                                )
                        except SyncError:
                            recent_acq = None
                            fallback_candidates = (
                                _header_aided_acquisitions(recent, self.band)
                                if self._header_aided_allowed
                                or expected_header_matched
                                else []
                            )
                            for candidate in fallback_candidates:
                                candidate_needed = (
                                    candidate.preamble_start + minimum
                                )
                                try:
                                    candidate_result = demodulate_gop_stream(
                                        recent[:candidate_needed],
                                        band=self.band,
                                        drift_track="off",
                                        interleave=self.interleave,
                                        acquisition=candidate,
                                    )
                                    if candidate_result.mode != self.expected_mode:
                                        continue
                                    if candidate_result.pilot_coherence < 0.20:
                                        continue
                                except SyncError:
                                    continue
                                recent_acq = candidate
                                self._debug(
                                    "header_aided_acquisition",
                                    offset=int(candidate.preamble_start),
                                    metric=float(candidate.metric),
                                    freq_offset_hz=float(candidate.freq_offset),
                                )
                                break
                        if recent_acq is not None:
                            if recent_start:
                                self.buffer = self.buffer[recent_start:]
                                self.samples_consumed += recent_start
                            self._pending_acquisition = recent_acq
                            self._awaiting_blind = False
                            self._awaiting_search_offset = 0
                            break
                        self._awaiting_search_offset += scan_step
                    if not self._awaiting_blind:
                        continue
                if len(self.buffer) < 12 * fs:
                    break
                if self._attempt_blind_acquisition(fs):
                    continue
                # Wait for one more second before another expensive scan.
                break
            if len(self.buffer) < minimum:
                break
            # A live decoder can accumulate multiple GOPs while the neural
            # decoder is busy. Searching the entire backlog chooses the
            # strongest (often latest) preamble and discards valid earlier
            # GOPs. Bound acquisition to the earliest complete-GOP window.
            # One chunk has 0.2 s of lead/tail around its acquisition+GOP.
            # A 0.25 s margin includes that first preamble but cannot include
            # the following chunk's preamble and tempt a stronger later lock.
            search_limit = min(len(self.buffer), minimum + fs // 4)
            try:
                if self._pending_acquisition is not None:
                    acq = self._pending_acquisition
                    self._pending_acquisition = None
                else:
                    acq = acquire(
                        to_baseband(
                            self.buffer[:search_limit], geom.fcenter_hz, fs
                        ),
                        band=self.band,
                    )
            except SyncError:
                if self.continuous:
                    blind_minimum = 12 * fs
                    if len(self.buffer) < blind_minimum:
                        # Unlike independently framed GOPs, a late-joined
                        # continuous stream has no next preamble. Retain audio
                        # until pilots plus one full beacon can identify phase.
                        if not self._awaiting_blind:
                            self._awaiting_search_offset = 0
                            self._header_aided_allowed = (
                                _ofdm_timing_metric(
                                    self.buffer[:search_limit], self.band
                                )
                                < 0.20
                            )
                        self._awaiting_blind = True
                        break
                    if self._attempt_blind_acquisition(fs):
                        continue
                    discarded = min(
                        _nsym, len(self.buffer) - blind_minimum + _nsym
                    )
                    self.buffer = self.buffer[discarded:]
                    self.samples_consumed += discarded
                    continue
                if len(self.buffer) > search_limit:
                    # Work through a backlog in bounded chronological steps.
                    # Jumping straight to its newest tail loses every queued
                    # GOP whenever neural decoding temporarily falls behind.
                    discarded = max(1, search_limit - minimum)
                    self.buffer = self.buffer[discarded:]
                    self.samples_consumed += discarded
                    continue
                # Preserve enough overlap for a preamble split across feeds.
                keep = min(len(self.buffer), max(minimum, 2 * fs))
                self.samples_consumed += len(self.buffer) - keep
                self.buffer = self.buffer[-keep:]
                break
            needed = acq.preamble_start + minimum
            if len(self.buffer) < needed:
                break
            candidate_sample = int(self.samples_consumed + acq.preamble_start)
            self._debug(
                "preamble_candidate",
                offset=int(acq.preamble_start),
                stream_sample=candidate_sample,
                metric=float(acq.metric),
                freq_offset_hz=float(acq.freq_offset),
            )
            segment = self.buffer[:needed]
            try:
                result = demodulate_gop_stream(
                    segment,
                    band=self.band,
                    drift_track="off",
                    interleave=self.interleave,
                    acquisition=acq,
                )
                if self.continuous and result.pilot_coherence < 0.20:
                    raise SyncError(
                        "startup payload pilot coherence too low "
                        f"({result.pilot_coherence:.2f})"
                    )
            except SyncError as error:
                self._debug(
                    "candidate_rejected",
                    offset=int(acq.preamble_start),
                    stream_sample=int(self.samples_consumed + acq.preamble_start),
                    metric=float(acq.metric),
                    reason=str(error),
                )
                if self.continuous:
                    if self._attempt_blind_acquisition(fs):
                        continue
                    if len(self.buffer) < 12 * fs:
                        # Retain a possible late-join payload until the
                        # pilot/beacon observation window is complete.
                        if not self._awaiting_blind:
                            self._awaiting_search_offset = 0
                            self._header_aided_allowed = (
                                _ofdm_timing_metric(
                                    self.buffer[:search_limit], self.band
                                )
                                < 0.20
                            )
                        self._awaiting_blind = True
                        break
                # A false peak must not pin the parser forever.
                discarded = max(1, acq.preamble_start + 1)
                self.buffer = self.buffer[discarded:]
                self.samples_consumed += discarded
                continue
            self.buffer = self.buffer[needed:]
            self.samples_consumed += needed
            missing_gops = self._accumulate_beacon(
                result, candidate_sample, minimum + 2 * int(0.1 * fs)
            )
            if self.continuous:
                self._tracking_mode = result.mode
                self._tracking_freq_offset = result.freq_offset
                self._tracking_bad_gops = 0
                self._tracking_pending.clear()
                self._awaiting_blind = False
            self._debug(
                "gop_accepted",
                metric=float(result.sync_metric),
                freq_offset_hz=float(result.freq_offset),
                header_score=float(result.header_score),
                pilot_coherence=float(result.pilot_coherence),
                snr_db=float(result.snr_db),
                beacon_chips=int(len(self.beacon_chips)),
                beacon_repeated_chips=int(len(self.beacon_repeated_chips)),
                missing_gops=int(missing_gops),
                callsign=result.callsign,
            )
            results.append(result)
        return results


def _estimate_snr_db(h_pilot: np.ndarray, band: str = "W") -> float:
    """Pilot-derived SNR in 2500 Hz reference bandwidth."""
    if len(h_pilot) < 2:
        return float("nan")
    diffs = np.diff(h_pilot, axis=0)
    noise_var = 0.5 * float(np.mean(np.abs(diffs) ** 2))
    sig_var = float(np.mean(np.abs(h_pilot) ** 2))
    if noise_var <= 0 or sig_var <= 0:
        return float("nan")
    geom = BANDS[band]
    snr_50hz = sig_var / noise_var
    snr_ref = snr_50hz * (geom.carriers * RS / SNR_REF_BW_HZ)
    return float(10.0 * np.log10(snr_ref))


def _pilot_coherence(h_pilot: np.ndarray) -> float:
    """Fraction of pilot-channel energy coherent across one GOP."""
    values = np.asarray(h_pilot)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    coherent = float(np.sum(np.abs(np.mean(values, axis=0)) ** 2))
    total = float(np.sum(np.mean(np.abs(values) ** 2, axis=0)))
    return coherent / max(total, 1e-12)


def blind_acquire_continuous_payload(
    audio: np.ndarray,
    mode: AETVModeSpec,
) -> BlindPayloadAcquisition:
    """Acquire a continuous payload using CP timing, pilots, and beacon phase.

    This consumes no RF overhead. CP repetition finds OFDM symbol timing, the
    temporally stable pilot identifies the one-pilot/four-data frame phase, and
    the beacon frame counter identifies the eight-frame GOP boundary.
    """
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    fs = mode.geometry.fs
    m = fs // RS
    ncp = m // 4
    nsym = m + ncp
    frame_samples = SYMS_PER_FRAME * nsym
    if len(values) < 12 * fs:
        raise SyncError("blind acquisition needs 12 seconds of continuous payload")

    z = to_baseband(values, mode.geometry.fcenter_hz, fs)
    product = z[:-m] * np.conj(z[m:])
    kernel = np.ones(ncp)
    correlation = signal.fftconvolve(product, kernel, mode="valid")
    e1 = signal.fftconvolve(np.abs(z[:-m]) ** 2, kernel, mode="valid")
    e2 = signal.fftconvolve(np.abs(z[m:]) ** 2, kernel, mode="valid")
    cp_metric = np.abs(correlation) / np.maximum(np.sqrt(e1 * e2), 1e-12)
    phase_scores = np.array(
        [np.mean(cp_metric[offset::nsym]) for offset in range(nsym)]
    )
    symbol_offset = int(np.argmax(phase_scores))
    timing_metric = float(phase_scores[symbol_offset])
    if timing_metric < 0.25:
        raise SyncError(f"blind CP timing confidence too low ({timing_metric:.2f})")

    starts = list(range(symbol_offset, len(values) - nsym + 1, nsym))
    symbols = np.array(
        [ofdm.demod_window(z, start + ncp, band=mode.band) for start in starts]
    )
    pilot = ofdm.pilot_sequence(mode.band)
    pilot_scores = []
    for phase in range(SYMS_PER_FRAME):
        estimates = symbols[phase::SYMS_PER_FRAME] / pilot
        coherent = np.sum(np.abs(np.mean(estimates, axis=0)) ** 2)
        total = np.sum(np.mean(np.abs(estimates) ** 2, axis=0)) + 1e-12
        pilot_scores.append(float(coherent / total))
    pilot_phase = int(np.argmax(pilot_scores))
    if pilot_scores[pilot_phase] < 0.20:
        raise SyncError(
            f"blind pilot confidence too low ({pilot_scores[pilot_phase]:.2f})"
        )

    logical_chips = []
    frame_starts = []
    for index in range(pilot_phase, len(symbols) - DATA_SYMS_PER_FRAME, SYMS_PER_FRAME):
        h_pilot = symbols[index] / pilot
        denominator = np.abs(h_pilot) ** 2 + 1e-4
        frame_starts.append(starts[index])
        for data_index in range(1, 1 + DATA_SYMS_PER_FRAME):
            equalized = symbols[index + data_index] * np.conj(h_pilot) / denominator
            logical_chips.append(
                0.25
                * (
                    np.real(equalized[mode.geometry.latent_carriers])
                    + np.imag(equalized[mode.geometry.latent_carriers])
                    + np.real(equalized[mode.geometry.beacon_carrier])
                    + np.imag(equalized[mode.geometry.beacon_carrier])
                )
            )
    found = find_beacon_superframe(np.asarray(logical_chips), threshold=0.4)
    if found is None or found.mode_index != mode.index:
        raise SyncError("blind acquisition has not decoded a matching beacon yet")

    relative_sync_frame = found.chip_offset // DATA_SYMS_PER_FRAME
    first_frame_counter = (found.frame_index - relative_sync_frame) % (
        beacon.MAX_FRAME_COUNTER + 1
    )
    frames_to_boundary = (-first_frame_counter) % FRAMES_PER_GOP
    payload_start = frame_starts[0] + frames_to_boundary * frame_samples
    if payload_start + fs > len(values):
        raise SyncError("blind acquisition found a boundary but needs more payload")
    # Start from the newest complete GOP so late join catches up immediately.
    payload_start += ((len(values) - fs - payload_start) // fs) * fs

    cp_samples = correlation[symbol_offset::nsym]
    phase = float(np.angle(np.mean(cp_samples))) if cp_samples.size else 0.0
    freq_offset = phase / (2.0 * np.pi * (m / fs))
    return BlindPayloadAcquisition(
        payload_start=int(payload_start),
        freq_offset=float(freq_offset),
        metric=timing_metric,
        beacon=found,
    )


def demodulate_gop_stream(
    audio: np.ndarray,
    band: str = "W",
    drift_track: str = "off",
    interleave: bool = True,
    acquisition: Acquisition | None = None,
) -> AETVDemodResult:
    """Demodulate an AETV passband transmission into reconstructed GOP latents and confidence weights.

    `drift_track` defaults to "off". The published receive path keeps it off.
    The loop below double-corrects: `h_f` comes from the *uncorrected* pilot, so
    the equalizer has already removed the per-frame common phase, and the
    `carrier_phase` rotation applies it a second time. `phase_err` is therefore
    never quite zero even on a noiseless signal, `carrier_freq` integrates that
    bias, and the phase ramps without bound. Measured on a clean 30-GOP
    loopback: 21.07 dB with the loop off against 12.67 dB on "slow" and
    10.83 dB on "fast", with the beacon decoding only in the first case. The
    damage grows with stream length, which is why single-GOP evaluation never
    saw it. Redesigning the loop is open work; until then this must stay off.
    """
    # The verified Kiwi OTA decoder normalized its recovered passband before
    # demodulation. Keep that behavior here: Wiener/equalizer regularizers are
    # otherwise absolute, so a low-level but clean capture produces shrunken
    # latents and weak beacon soft decisions while still looking half-valid.
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-12:
        raise SyncError("no signal energy")
    audio = audio / peak
    geom, fs, m, ncp, nsym, frame_samples, preamble_samples, header_samples, leadin_samples, leadout_samples = _band_params(band)
    z = to_baseband(audio, geom.fcenter_hz, fs)

    # Acquire preamble
    acq = acquisition if acquisition is not None else acquire(z, band=band)
    z_cfo = freq_correct(z, -acq.freq_offset, fs)

    # Position past preamble
    data_start = acq.preamble_start + preamble_samples

    # Estimate the channel from the final known preamble repetition. Header
    # BPSK cannot safely be decoded from its raw real component over an RF
    # channel with arbitrary per-carrier phase.
    preamble_pilot = ofdm.pilot_sequence(band)
    preamble_useful = acq.preamble_start + 2 * ncp + (PREAMBLE_REPEATS - 1) * m
    r_preamble = ofdm.demod_window(z_cfo, preamble_useful, band=band)
    h_preamble = r_preamble / preamble_pilot
    header_denom = np.abs(h_preamble) ** 2 + 1e-4

    # Demodulate and equalize the repeated header symbols.
    header_syms = []
    for s in range(HEADER_SYMS):
        sym_start = data_start + s * nsym + ncp
        received = ofdm.demod_window(z_cfo, sym_start, band=band)
        header_syms.append(received * np.conj(h_preamble) / header_denom)

    header_carriers = np.real(np.mean(header_syms, axis=0))
    valid_headers = []
    for observed_header in _header_candidates(header_carriers):
        mode_idx, version = decode_header(observed_header)
        mode = AETV_MODES_BY_INDEX.get(mode_idx)
        if version != PROTOCOL_VERSION or mode is None or mode.band != band:
            continue
        expected_header = encode_header(mode_idx, version)
        score = float(
            np.dot(observed_header, expected_header)
            / max(np.linalg.norm(observed_header) * np.linalg.norm(expected_header), 1e-12)
        )
        valid_headers.append((score, mode))
    if not valid_headers:
        raise SyncError("mode header rejected")
    header_score, mode = max(valid_headers, key=lambda item: item[0])
    if header_score < 0.45:
        raise SyncError(f"mode header confidence too low ({header_score:.2f})")

    frames_start = data_start + header_samples
    n_available_frames = max(0, (len(z_cfo) - frames_start) // frame_samples)
    n_gops = n_available_frames // FRAMES_PER_GOP


    if n_gops == 0:
        raise SyncError("insufficient audio length for a full GOP")

    pilot_seq = ofdm.pilot_sequence(band)
    all_data_syms = []
    all_data_weights = []
    all_beacon_chips = []
    all_beacon_repeated_chips = []
    h_pilots = []

    # Drift tracker loop setup
    alpha, beta = 0.0, 0.0
    if drift_track == "slow":
        alpha, beta = DRIFT_SLOW_ALPHA, DRIFT_SLOW_BETA
    elif drift_track == "fast":
        alpha, beta = DRIFT_FAST_ALPHA, DRIFT_FAST_BETA

    carrier_phase = 0.0
    carrier_freq = 0.0

    # Demodulate all frames
    total_frames = n_gops * FRAMES_PER_GOP
    for f in range(total_frames):
        frame_sample = frames_start + f * frame_samples

        # Demodulate pilot symbol
        pilot_sample = frame_sample + ncp
        r_pilot = ofdm.demod_window(z_cfo, pilot_sample, band=band)

        # Drift tracking correction
        if alpha > 0:
            phase_err = float(np.angle(np.mean(r_pilot * np.conj(pilot_seq))))
            carrier_freq += beta * phase_err
            carrier_phase += alpha * phase_err + carrier_freq

        # Channel estimate H on pilot
        h_f = r_pilot / pilot_seq
        h_pilots.append(h_f)

        # Equalize 4 data symbols in this frame
        for s in range(DATA_SYMS_PER_FRAME):
            sym_sample = frame_sample + (1 + s) * nsym + ncp
            r_sym = ofdm.demod_window(z_cfo, sym_sample, band=band)

            if alpha > 0:
                r_sym *= np.exp(-1j * carrier_phase)

            # Zero-forcing / Wiener equalization: S = R / H
            denom = np.abs(h_f) ** 2 + 1e-4
            eq_sym = r_sym * np.conj(h_f) / denom
            weight = np.clip(np.abs(h_f) / (np.abs(h_f) + 0.01), 0.0, 1.0)


            all_data_syms.append(eq_sym)
            all_data_weights.append(weight)
            if band == "U":
                all_beacon_chips.append(np.real(eq_sym[geom.latent_carriers]))
                all_beacon_chips.append(np.imag(eq_sym[geom.latent_carriers]))
                all_beacon_chips.append(np.real(eq_sym[geom.beacon_carrier]))
                all_beacon_chips.append(np.imag(eq_sym[geom.beacon_carrier]))
                all_beacon_repeated_chips.append(
                    0.25
                    * (
                        np.real(eq_sym[geom.latent_carriers])
                        + np.imag(eq_sym[geom.latent_carriers])
                        + np.real(eq_sym[geom.beacon_carrier])
                        + np.imag(eq_sym[geom.beacon_carrier])
                    )
                )
            else:
                all_beacon_chips.append(np.real(eq_sym[geom.beacon_carrier]))

    h_pilot_arr = np.array(h_pilots)
    snr_est = _estimate_snr_db(h_pilot_arr, band=band)
    pilot_coherence = _pilot_coherence(h_pilot_arr)

    # Beacon decode
    soft_beacon = np.array(all_beacon_chips)
    soft_repeated_beacon = np.array(all_beacon_repeated_chips)
    beacon_res = find_beacon_superframe(soft_beacon)

    # Unpack GOPs
    gops_latents = []
    gops_weights = []

    syms_arr = np.array(all_data_syms)
    weights_arr = np.array(all_data_weights)

    for g in range(n_gops):
        g_syms = syms_arr[g * 32 : (g + 1) * 32]
        g_w = weights_arr[g * 32 : (g + 1) * 32]
        latents, weights = framing.unpack_gop_symbols(
            data_symbols=g_syms,
            data_weights=g_w,
            band=band,
            interleave=interleave,
        )
        gops_latents.append(latents)
        gops_weights.append(weights)

    return AETVDemodResult(
        gops_latents=gops_latents,
        gops_weights=gops_weights,
        mode=mode,
        freq_offset=acq.freq_offset,
        sync_metric=acq.metric,
        frames_received=total_frames,
        beacon=beacon_res,
        callsign=beacon_res.callsign if beacon_res else "",
        preamble_start=acq.preamble_start,
        snr_db=snr_est,
        beacon_chips=soft_beacon,
        beacon_repeated_chips=soft_repeated_beacon,
        header_score=header_score,
        pilot_coherence=pilot_coherence,
    )


def demodulate_tracked_gop(
    audio: np.ndarray,
    mode: AETVModeSpec,
    freq_offset: float = 0.0,
    interleave: bool = True,
) -> AETVDemodResult:
    """Decode one boundary-aligned payload while already tracking a stream.

    The payload itself contains a known pilot in every OFDM frame, so channel
    equalization needs no repeated RF header. A locally generated, level-matched
    acquisition prefix reuses the verified payload decoder without consuming
    any on-air samples. The per-frame pilots remove the residual common phase;
    ``freq_offset`` is retained for receiver telemetry.
    """
    payload = np.asarray(audio, dtype=np.float32).reshape(-1)
    expected = mode.geometry.fs
    if len(payload) != expected:
        raise SyncError(
            f"tracked GOP needs exactly {expected} samples, got {len(payload)}"
        )
    prefix = _acquisition_wave(mode).astype(np.float64)
    payload_rms = float(np.sqrt(np.mean(payload.astype(np.float64) ** 2)))
    prefix_rms = float(np.sqrt(np.mean(prefix**2)))
    if payload_rms <= 1e-12:
        raise SyncError("tracked GOP has no signal energy")
    prefix *= payload_rms / max(prefix_rms, 1e-12)
    result = demodulate_gop_stream(
        np.concatenate([prefix, payload]),
        band=mode.band,
        drift_track="off",
        interleave=interleave,
    )
    result.freq_offset = float(freq_offset)
    result.preamble_start = 0
    return result
