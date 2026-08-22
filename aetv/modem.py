"""Top-level AETV modem: latent vector / GOP streams <-> passband audio samples.

Handles:
- Transmit modulation: preamble | header | continuous GOPs (interleaved latents + beacon chips).
- Envelope clipping (0.5 dB headroom) and post-clip bandpass filtering.
- Demodulation: baseband conversion, preamble sync / blind sync, pilot equalization,
  drift tracking loop, confidence weighting, beacon decode, and GOP de-interleaving.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

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
from .sync import SyncError, acquire


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
        header_syms[s, : min(len(header_chips), geom.carriers)] = header_chips[
            : min(len(header_chips), geom.carriers)
        ]
    header_wave = ofdm.modulate_symbols(header_syms, band)

    # Modulate GOP frames
    pilot = ofdm.pilot_sequence(band)
    gop_audio_chunks = []
    chip_offset = 0

    for gop_idx, gop_latents in enumerate(gops):
        gop_chips = beacon_chips[chip_offset : chip_offset + FRAMES_PER_GOP * DATA_SYMS_PER_FRAME]
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


def demodulate_gop_stream(
    audio: np.ndarray,
    band: str = "W",
    drift_track: str = "off",
    interleave: bool = True,
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
    geom, fs, m, ncp, nsym, frame_samples, preamble_samples, header_samples, leadin_samples, leadout_samples = _band_params(band)
    z = to_baseband(audio, geom.fcenter_hz, fs)

    # Acquire preamble
    acq = acquire(z, band=band)
    z_cfo = freq_correct(z, -acq.freq_offset, fs)

    # Position past preamble
    data_start = acq.preamble_start + preamble_samples

    # Demodulate header
    header_syms = []
    for s in range(HEADER_SYMS):
        sym_start = data_start + s * nsym + ncp
        h_sym = ofdm.demod_window(z_cfo, sym_start, band=band)
        header_syms.append(h_sym)

    header_soft = np.real(np.mean(header_syms, axis=0))
    header_res = decode_header(header_soft)
    if header_res is not None:
        mode_idx, _ = header_res
        mode = AETV_MODES_BY_INDEX.get(mode_idx, AETV_MODES["V1"])
    else:
        mode = AETV_MODES["V1"]

    frames_start = data_start + header_samples
    n_available_frames = max(0, (len(z_cfo) - frames_start) // frame_samples)
    n_gops = n_available_frames // FRAMES_PER_GOP


    if n_gops == 0:
        raise SyncError("insufficient audio length for a full GOP")

    pilot_seq = ofdm.pilot_sequence(band)
    all_data_syms = []
    all_data_weights = []
    all_beacon_chips = []
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
            all_beacon_chips.append(np.real(eq_sym[geom.beacon_carrier]))

    h_pilot_arr = np.array(h_pilots)
    snr_est = _estimate_snr_db(h_pilot_arr, band=band)

    # Beacon decode
    soft_beacon = np.array(all_beacon_chips)
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
    )
