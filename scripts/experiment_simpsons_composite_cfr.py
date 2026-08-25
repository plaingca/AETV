#!/usr/bin/env python3
"""Evaluate low-PAPR analog voice + AETV on a 60-second Simpsons excerpt.

The candidate transmitter combines three deliberately separable stages:

* the normal stateful AETV clip/filter conditioner;
* complex-envelope CESSB processing of the delayed speech branch;
* per-GOP common-phase selection followed by a gentle whole-composite CFR.

It keeps a paired conditioned-AETV-only control, sweeps CFR thresholds using
clean transport and audio gates, then renders both transmitters through the
same deterministic HF channel profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pystoi import stoi
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.analog_av import (
    COMPOSITE_FS,
    extract_aetv,
    extract_voice,
    prepare_voice,
    translate_aetv_up,
)
from aetv.audio_metrics import AudioPerceptualLoss
from aetv.beacon import generate_beacon_chips
from aetv.codec import AETVCodec
from aetv.config import FRAMES_PER_GOP
from aetv.hfchannel import CHANNEL_PROFILES, StreamingChannelEmulator
from aetv.modem import _ContinuousTxConditioner, _payload_wave, demodulate_tracked_gop
from experiment_analog_voice_aetv import write_wav
from experiment_simpsons_analog_channel import (
    estimate_native_delay,
    extract_source,
    label_frames,
    mux,
    psnr,
    shift_to_reference,
)


PROFILES = ("clean", "awgn12", "awgn6", "awgn0", "mpp12", "mpp6", "mpp0")
RENDER_PROFILES = ("clean", "awgn6", "mpp6")


def analytic_papr_db(values: np.ndarray, region: slice | None = None) -> float:
    z = signal.hilbert(np.asarray(values, dtype=np.float64))
    selected = z if region is None else z[region]
    return float(
        10.0
        * np.log10(np.max(np.abs(selected) ** 2) / np.mean(np.abs(selected) ** 2))
    )


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    scale = np.dot(estimate, reference) / max(np.dot(reference, reference), 1e-18)
    target = scale * reference
    error = estimate - target
    return float(10.0 * np.log10(np.dot(target, target) / max(np.dot(error, error), 1e-18)))


def complex_linear_phase_lowpass(
    values: np.ndarray,
    *,
    fs: int,
    pass_hz: float,
    stop_hz: float,
) -> np.ndarray:
    """Offline positive-frequency low-pass with zero circular boundary leakage."""
    pad = fs // 4
    padded = np.pad(np.asarray(values, dtype=np.complex128), (pad, pad))
    frequency = np.fft.fftfreq(len(padded), 1.0 / fs)
    response = np.zeros(len(padded), dtype=np.float64)
    response[(frequency >= 0.0) & (frequency <= pass_hz)] = 1.0
    transition = (frequency > pass_hz) & (frequency < stop_hz)
    response[transition] = 0.5 + 0.5 * np.cos(
        np.pi * (frequency[transition] - pass_hz) / (stop_hz - pass_hz)
    )
    filtered = np.fft.ifft(np.fft.fft(padded) * response)
    return filtered[pad:-pad]


def cessb_speech(
    speech: np.ndarray,
    *,
    active_start: int,
    drive_db: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Hershberger-style complex modulus clip and overshoot correction."""
    z = signal.hilbert(np.asarray(speech, dtype=np.float64))
    reference = float(np.max(np.abs(speech[active_start:]))) * 10.0 ** (-drive_db / 20.0)
    magnitude = np.abs(z)
    clipped = z / np.maximum(1.0, magnitude / max(reference, 1e-12))
    filtered = complex_linear_phase_lowpass(
        clipped, fs=COMPOSITE_FS, pass_hz=2_100.0, stop_hz=2_200.0
    )
    ratio = np.abs(filtered) / max(reference, 1e-12)
    # 0.3 / bandwidth at 12 kHz rounds to the minimum odd window of 3 samples.
    stretched = ndimage.maximum_filter1d(ratio, size=3, mode="nearest")
    corrected = filtered / (1.0 + 1.9 * np.maximum(stretched - 1.0, 0.0))
    final = complex_linear_phase_lowpass(
        corrected, fs=COMPOSITE_FS, pass_hz=2_100.0, stop_hz=2_200.0
    )
    result = np.real(final)
    diagnostics = {
        "drive_db": drive_db,
        "initial_samples_limited_pct": float(
            100.0 * np.mean(magnitude[active_start:] > reference)
        ),
        "postfilter_overshoot_pct": float(
            100.0 * max(np.max(np.abs(final[active_start:])) / reference - 1.0, 0.0)
        ),
        "si_sdr_db": si_sdr_db(speech[active_start:], result[active_start:]),
    }
    return result, diagnostics


def _fft_lowpass(values: np.ndarray, pass_hz: float, stop_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(np.asarray(values, dtype=np.float64))
    frequency = np.fft.rfftfreq(len(values), 1.0 / COMPOSITE_FS)
    response = np.ones_like(frequency)
    response[frequency >= stop_hz] = 0.0
    transition = (frequency > pass_hz) & (frequency < stop_hz)
    response[transition] = 0.5 + 0.5 * np.cos(
        np.pi * (frequency[transition] - pass_hz) / (stop_hz - pass_hz)
    )
    return np.fft.irfft(spectrum * response, n=len(values))


def _compress_band(
    values: np.ndarray,
    *,
    peak_reference: float,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    max_reduction_db: float,
) -> tuple[np.ndarray, dict[str, float]]:
    envelope = np.abs(signal.hilbert(values)) / max(peak_reference, 1e-12)
    level_db = 20.0 * np.log10(np.maximum(envelope, 1e-8))
    reduction = np.maximum(level_db - threshold_db, 0.0) * (1.0 - 1.0 / ratio)
    target_db = -np.minimum(reduction, max_reduction_db)
    attack = math.exp(-1.0 / (COMPOSITE_FS * attack_ms * 1e-3))
    release = math.exp(-1.0 / (COMPOSITE_FS * release_ms * 1e-3))
    smoothed = np.empty_like(target_db)
    state = 0.0
    for index, target in enumerate(target_db):
        coefficient = attack if target < state else release
        state = coefficient * state + (1.0 - coefficient) * float(target)
        smoothed[index] = state
    gain = 10.0 ** (smoothed / 20.0)
    active = envelope > 1e-4
    diagnostics = {
        "mean_gain_reduction_db": float(-np.mean(smoothed[active])) if active.any() else 0.0,
        "p95_gain_reduction_db": float(-np.percentile(smoothed[active], 5)) if active.any() else 0.0,
        "maximum_gain_reduction_db": float(-np.min(smoothed)),
    }
    return values * gain, diagnostics


def aggressive_multiband_speech(
    speech: np.ndarray,
    *,
    active_start: int,
    highpass: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Aggressive three-band HF speech compression and real peak limiting."""
    values = np.asarray(speech, dtype=np.float64)
    compressor_input = values
    highpass_diagnostics: dict[str, float] | None = None
    if highpass:
        compressor_input = values - _fft_lowpass(values, 150.0, 220.0)
        before_power = float(np.mean(values[active_start:] ** 2))
        after_power = float(np.mean(compressor_input[active_start:] ** 2))
        highpass_diagnostics = {
            "stop_hz": 150.0,
            "pass_hz": 220.0,
            "removed_input_power_pct": float(
                100.0 * max(0.0, 1.0 - after_power / max(before_power, 1e-12))
            ),
        }
    low_sum = _fft_lowpass(compressor_input, 450.0, 550.0)
    mid_sum = _fft_lowpass(compressor_input, 1_000.0, 1_100.0)
    full_sum = _fft_lowpass(compressor_input, 2_100.0, 2_200.0)
    bands = (low_sum, mid_sum - low_sum, full_sum - mid_sum)
    peak_reference = float(np.max(np.abs(compressor_input[active_start:])))
    processed = []
    band_diagnostics = []
    for band in bands:
        compressed, diagnostics = _compress_band(
            band,
            peak_reference=peak_reference,
            threshold_db=-20.0,
            ratio=6.0,
            attack_ms=3.0,
            release_ms=100.0,
            max_reduction_db=12.0,
        )
        processed.append(compressed)
        band_diagnostics.append(diagnostics)
    result = _fft_lowpass(sum(processed), 2_100.0, 2_200.0)
    input_rms = float(np.sqrt(np.mean(values[active_start:] ** 2)))
    result *= input_rms / max(float(np.sqrt(np.mean(result[active_start:] ** 2))), 1e-12)

    # A real-audio peak limiter supplies the dense, accurately bounded input
    # that the following complex-envelope CESSB stage expects.
    limiter = input_rms * 10.0 ** (7.0 / 20.0)
    for _ in range(2):
        result = np.clip(result, -limiter, limiter)
        result = _fft_lowpass(result, 2_100.0, 2_200.0)
    result *= input_rms / max(float(np.sqrt(np.mean(result[active_start:] ** 2))), 1e-12)

    def real_crest(values_: np.ndarray) -> float:
        active_values = values_[active_start:]
        return float(
            20.0
            * np.log10(
                np.max(np.abs(active_values))
                / max(float(np.sqrt(np.mean(active_values ** 2))), 1e-12)
            )
        )

    return result, {
        "bands_hz": (
            [[200, 500], [500, 1_050], [1_050, 2_200]]
            if highpass
            else [[0, 500], [500, 1_050], [1_050, 2_200]]
        ),
        "highpass": highpass_diagnostics,
        "threshold_db_peak_relative": -20.0,
        "ratio": 6.0,
        "attack_ms": 3.0,
        "release_ms": 100.0,
        "max_reduction_db": 12.0,
        "real_peak_limiter_crest_db": 7.0,
        "input_real_crest_db": real_crest(values),
        "output_real_crest_db": real_crest(result),
        "bands": band_diagnostics,
        "si_sdr_db": si_sdr_db(values[active_start:], result[active_start:]),
    }


def normalize_active(values: np.ndarray, active: slice, rms: float = 0.22) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result *= rms / max(float(np.sqrt(np.mean(result[active] ** 2))), 1e-12)
    return result


def choose_gop_phases(
    upper: np.ndarray,
    speech: np.ndarray,
    *,
    gops: int,
    phase_count: int = 32,
) -> tuple[np.ndarray, list[float], dict[str, float]]:
    """Choose a pilot-transparent common AETV phase independently per GOP."""
    samples = COMPOSITE_FS
    upper_analytic = signal.hilbert(upper)
    speech_analytic = signal.hilbert(speech)
    output = upper_analytic.copy()
    phases: list[float] = []
    candidate_phases = np.linspace(0.0, 2.0 * np.pi, phase_count, endpoint=False)
    before_peaks = []
    after_peaks = []
    for index in range(gops):
        segment = slice(index * samples, (index + 1) * samples)
        uz = upper_analytic[segment]
        vz = speech_analytic[segment]
        peaks = np.asarray([
            np.max(np.abs(vz + uz * np.exp(1j * phase)))
            for phase in candidate_phases
        ])
        selected = int(np.argmin(peaks))
        phase = float(candidate_phases[selected])
        output[segment] = uz * np.exp(1j * phase)
        phases.append(phase)
        before_peaks.append(float(peaks[0]))
        after_peaks.append(float(peaks[selected]))
    diagnostics = {
        "phase_candidates": phase_count,
        "mean_local_peak_reduction_db": float(
            np.mean(20.0 * np.log10(np.asarray(before_peaks) / np.asarray(after_peaks)))
        ),
        "maximum_local_peak_reduction_db": float(
            np.max(20.0 * np.log10(np.asarray(before_peaks) / np.asarray(after_peaks)))
        ),
    }
    return np.real(output), phases, diagnostics


def combine(upper: np.ndarray, speech: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    composite = np.asarray(upper, dtype=np.float64) + np.asarray(speech, dtype=np.float64)
    scale = 0.95 / max(float(np.max(np.abs(composite))), 0.95)
    return composite * scale, speech * scale, upper * scale


def composite_cfr(
    composite: np.ndarray,
    *,
    steady: slice,
    target_db: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Complex-envelope CFR with in-band overshoot correction and filtering."""
    values = np.asarray(composite, dtype=np.float64)
    analytic = signal.hilbert(values)
    rms_envelope = float(np.sqrt(np.mean(np.abs(analytic[steady]) ** 2)))
    threshold = rms_envelope * 10.0 ** (target_db / 20.0)
    magnitude = np.abs(analytic)
    clipped = analytic / np.maximum(1.0, magnitude / max(threshold, 1e-12))
    filtered = complex_linear_phase_lowpass(
        clipped, fs=COMPOSITE_FS, pass_hz=4_850.0, stop_hz=4_950.0
    )
    ratio = np.abs(filtered) / max(threshold, 1e-12)
    stretched = ndimage.maximum_filter1d(ratio, size=3, mode="nearest")
    corrected = filtered / (1.0 + 1.9 * np.maximum(stretched - 1.0, 0.0))
    final = complex_linear_phase_lowpass(
        corrected, fs=COMPOSITE_FS, pass_hz=4_850.0, stop_hz=4_950.0
    )
    result = np.real(final)
    # Keep average RF power fixed while comparing distortion; final peak scaling
    # only changes file headroom and cannot change PAPR.
    result *= float(np.sqrt(np.mean(values[steady] ** 2))) / max(
        float(np.sqrt(np.mean(result[steady] ** 2))), 1e-12
    )
    result *= 0.95 / max(float(np.max(np.abs(result))), 0.95)
    diagnostics = {
        "target_db": target_db,
        "samples_initially_limited_pct": float(100.0 * np.mean(magnitude > threshold)),
        "actual_papr_steady_db": analytic_papr_db(result, steady),
        "actual_papr_complete_db": analytic_papr_db(result),
    }
    return result, diagnostics


def clean_transport_metrics(
    composite: np.ndarray,
    *,
    codec: AETVCodec,
    latents: list[np.ndarray],
    reference_audio: np.ndarray,
) -> dict[str, float | int]:
    native = extract_aetv(composite)
    recovered_voice = extract_voice(composite)[8_000 : (len(latents) + 1) * 8_000]
    decoded = 0
    errors = []
    correlations = []
    for index, latent in enumerate(latents):
        segment = native[index * 8_000 : (index + 1) * 8_000]
        try:
            result = demodulate_tracked_gop(segment, codec.mode, interleave=True)
            recovered = result.gops_latents[0]
            decoded += 1
            error = recovered.astype(np.float64) - latent.astype(np.float64)
            errors.append(float(np.mean(error ** 2) / np.mean(latent.astype(np.float64) ** 2)))
            correlations.append(float(np.corrcoef(latent, recovered)[0, 1]))
        except Exception:
            errors.append(float("inf"))
            correlations.append(0.0)
    return {
        "decoded_gops": decoded,
        "mean_latent_nmse": float(np.mean(errors)),
        "mean_latent_correlation": float(np.mean(correlations)),
        "audio_stoi": float(stoi(reference_audio, recovered_voice, 8_000, extended=False)),
        "audio_si_sdr_db": si_sdr_db(reference_audio, recovered_voice),
    }


def evaluate_profile(
    composite: np.ndarray,
    *,
    profile: str,
    codec: AETVCodec,
    latents: list[np.ndarray],
    source_frames: np.ndarray,
    reference_audio: np.ndarray,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    channel = StreamingChannelEmulator(profile, seed=20260825, fs=COMPOSITE_FS)
    impaired = channel.process(composite)
    reference_native = extract_aetv(composite)
    native_unaligned = extract_aetv(impaired)
    lag = estimate_native_delay(reference_native, native_unaligned)
    native = shift_to_reference(native_unaligned, lag, len(reference_native))
    voice = shift_to_reference(extract_voice(impaired), lag, len(reference_native))
    recovered_audio = voice[8_000 : (len(latents) + 1) * 8_000]

    reconstructions = []
    psnrs = []
    errors = []
    decoded = 0
    for index, latent in enumerate(latents):
        segment = native[index * 8_000 : (index + 1) * 8_000]
        try:
            result = demodulate_tracked_gop(segment, codec.mode, interleave=True)
            recovered = result.gops_latents[0]
            weights = result.gops_weights[0]
            decoded += 1
        except Exception:
            recovered = np.zeros_like(latent)
            weights = np.zeros_like(latent, dtype=np.float32)
        reconstruction = codec.decode_gop(recovered, weights)
        reconstructions.append(reconstruction)
        source = source_frames[index * 6 : (index + 1) * 6]
        psnrs.append(psnr(source, reconstruction))
        error = recovered.astype(np.float64) - latent.astype(np.float64)
        errors.append(float(np.mean(error ** 2) / np.mean(latent.astype(np.float64) ** 2)))
    reconstructed_frames = np.concatenate(reconstructions)

    audio_metric = AudioPerceptualLoss(si_sdr_weight=0.0)
    components = audio_metric.components(
        torch.from_numpy(recovered_audio[None]).float(),
        torch.from_numpy(reference_audio[None]).float(),
    )
    row = {
        "profile": profile,
        "label": CHANNEL_PROFILES[profile].label,
        "alignment_lag_native_samples": lag,
        "decoded_gops": decoded,
        "gops": len(latents),
        "mean_video_psnr_db": float(np.mean(psnrs)),
        "p10_video_psnr_db": float(np.percentile(psnrs, 10)),
        "mean_latent_nmse": float(np.mean(errors)),
        "audio_si_sdr_db": float(-components["si_sdr"]),
        "audio_mel": float(components["mel"]),
        "audio_stoi": float(stoi(reference_audio, recovered_audio, 8_000, extended=False)),
    }
    return row, reconstructed_frames, recovered_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/home/plaing/SSTVAE/The Simpsons Season 31 Episode 20 - The Simpsons Full NoCuts-iex52uxH460.mp4",
    )
    parser.add_argument("--checkpoint", default="models/v8-hf3k-face-gan.pt")
    parser.add_argument("--out", default="runs/simpsons-composite-cfr-60s")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--name", default="simpsons", help="Filename prefix for rendered clips")
    parser.add_argument("--callsign", default="SIMPS", help="Beacon callsign (up to 8 characters)")
    parser.add_argument(
        "--voice-processor",
        choices=("cessb", "aggressive-multiband", "aggressive-multiband-highpass"),
        default="cessb",
    )
    parser.add_argument("--cfr-targets", nargs="+", type=float, default=(10.0, 9.0, 8.0, 7.0))
    parser.add_argument("--min-audio-si-sdr", type=float, default=20.0)
    parser.add_argument("--min-audio-stoi", type=float, default=0.0)
    parser.add_argument("--max-stoi-drop", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.duration < 2:
        raise ValueError("duration must be at least two seconds")
    if not args.name or "/" in args.name:
        raise ValueError("name must be a non-empty filename prefix without slashes")
    if not 1 <= len(args.callsign) <= 8:
        raise ValueError("callsign must contain 1 to 8 characters")

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_frames, raw_audio = extract_source(Path(args.input), args.duration)
    codec = AETVCodec(args.checkpoint, device=args.device, mode="V8")
    conditioner = _ContinuousTxConditioner(codec.mode)
    conditioned_native_chunks = []
    latents = []
    for index in range(args.duration):
        frames = source_frames[index * 6 : (index + 1) * 6]
        latent = codec.encode_gop(frames)
        chips = generate_beacon_chips(
            FRAMES_PER_GOP,
            start_frame=index * FRAMES_PER_GOP,
            callsign=args.callsign,
            mode_index=codec.mode.index,
        )
        native = _payload_wave(latent, chips, codec.mode, interleave=True)
        conditioned = conditioner.process(
            native, reference_power=float(np.mean(native.astype(np.float64) ** 2))
        )
        latents.append(latent)
        conditioned_native_chunks.append(conditioned)
        if (index + 1) % 10 == 0:
            print(f"encoded {index + 1}/{args.duration}", flush=True)

    # Two causal 201-tap FIR passes contribute 200 native samples of group
    # delay. Flush the state, remove that delay on the continuous stream, and
    # only then split GOPs for translation. Splitting first silently moves the
    # OFDM framing boundaries and produces plausible-looking demod calls with
    # completely wrong latents.
    native_group_delay = 2 * ((201 - 1) // 2)
    flush = conditioner.process(
        np.zeros(native_group_delay, dtype=np.float64), reference_power=0.0
    )
    conditioned_stream = np.concatenate(conditioned_native_chunks + [flush])
    conditioned_stream = conditioned_stream[
        native_group_delay : native_group_delay + args.duration * 8_000
    ]
    # Translate and mask the continuous stream in one operation. Per-GOP FFT
    # masking creates artificial wrap/boundary transients after the causal
    # conditioner, which badly inflates PAPR even though the payload decodes.
    upper_active = translate_aetv_up(
        np.concatenate([conditioned_stream, np.zeros(8_000, dtype=np.float64)])
    )

    second = COMPOSITE_FS
    steady = slice(second, -second)
    sustained = slice(second, -2 * second)
    upper = upper_active
    speech_gops = [
        prepare_voice(raw_audio[index * 8_000 : (index + 1) * 8_000])
        for index in range(args.duration)
    ]
    speech = np.concatenate([np.zeros(second)] + speech_gops)
    upper = normalize_active(upper, slice(0, -second))
    speech = normalize_active(speech, slice(second, None))
    reference_audio = np.concatenate([
        signal.resample_poly(gop, 2, 3)[:8_000] for gop in speech_gops
    ])

    current, _, _ = combine(upper, speech)
    multiband_diagnostics: dict[str, object] | None = None
    speech_for_cessb = speech
    if args.voice_processor in ("aggressive-multiband", "aggressive-multiband-highpass"):
        speech_for_cessb, multiband_diagnostics = aggressive_multiband_speech(
            speech,
            active_start=second,
            highpass=args.voice_processor == "aggressive-multiband-highpass",
        )
    cessb, cessb_diagnostics = cessb_speech(
        speech_for_cessb, active_start=second, drive_db=3.0
    )
    cessb = normalize_active(cessb, slice(second, None))
    phased_upper, phases, phase_diagnostics = choose_gop_phases(
        upper, cessb, gops=args.duration, phase_count=32
    )
    phased_upper = normalize_active(phased_upper, slice(0, -second))
    candidate_unlimited, _, _ = combine(phased_upper, cessb)

    baseline_clean = clean_transport_metrics(
        current, codec=codec, latents=latents, reference_audio=reference_audio
    )
    unlimited_clean = clean_transport_metrics(
        candidate_unlimited, codec=codec, latents=latents, reference_audio=reference_audio
    )
    sweep = []
    candidates: list[tuple[np.ndarray, dict[str, object]]] = []
    for target in args.cfr_targets:
        waveform, cfr_diagnostics = composite_cfr(
            candidate_unlimited, steady=steady, target_db=target
        )
        clean = clean_transport_metrics(
            waveform, codec=codec, latents=latents, reference_audio=reference_audio
        )
        row: dict[str, object] = {**cfr_diagnostics, **clean}
        sweep.append(row)
        candidates.append((waveform, row))
        print("sweep " + json.dumps(row, sort_keys=True), flush=True)

    nmse_gate = max(
        float(unlimited_clean["mean_latent_nmse"]) * 1.25,
        float(unlimited_clean["mean_latent_nmse"]) + 0.005,
    )
    acceptable = [
        item for item in candidates
        if int(item[1]["decoded_gops"]) == args.duration
        and float(item[1]["mean_latent_nmse"]) <= nmse_gate
        and float(item[1]["audio_stoi"]) >= float(unlimited_clean["audio_stoi"]) - args.max_stoi_drop
        and float(item[1]["audio_stoi"]) >= args.min_audio_stoi
        and float(item[1]["audio_si_sdr_db"]) >= args.min_audio_si_sdr
    ]
    if acceptable:
        selected_waveform, selected = min(
            acceptable, key=lambda item: float(item[1]["actual_papr_steady_db"])
        )
    else:
        selected_waveform = candidate_unlimited
        selected = {
            "target_db": None,
            "actual_papr_steady_db": analytic_papr_db(candidate_unlimited, steady),
            "actual_papr_complete_db": analytic_papr_db(candidate_unlimited),
            **unlimited_clean,
        }

    write_wav(out / "transmit_current.wav", current, COMPOSITE_FS)
    write_wav(out / "transmit_candidate.wav", selected_waveform, COMPOSITE_FS)
    (out / "phase_choices.json").write_text(json.dumps({
        "radians": phases,
        "degrees": [math.degrees(value) for value in phases],
        **phase_diagnostics,
    }, indent=2) + "\n")
    (out / "sweep.json").write_text(json.dumps({
        "baseline_papr_overlap_db": analytic_papr_db(current, steady),
        "baseline_papr_sustained_db": analytic_papr_db(current, sustained),
        "baseline_clean": baseline_clean,
        "cessb": cessb_diagnostics,
        "voice_processor": args.voice_processor,
        "multiband": multiband_diagnostics,
        "phase_selection": phase_diagnostics,
        "unlimited_candidate_papr_steady_db": analytic_papr_db(candidate_unlimited, steady),
        "unlimited_candidate_clean": unlimited_clean,
        "nmse_gate": nmse_gate,
        "sweep": sweep,
        "selected": selected,
    }, indent=2) + "\n")

    frames_by_tx: dict[str, dict[str, np.ndarray]] = {"current": {}, "candidate": {}}
    audio_by_tx: dict[str, dict[str, np.ndarray]] = {"current": {}, "candidate": {}}
    metrics = []
    for tx_name, waveform in (("current", current), ("candidate", selected_waveform)):
        for profile in PROFILES:
            row, reconstructed, recovered_audio = evaluate_profile(
                waveform,
                profile=profile,
                codec=codec,
                latents=latents,
                source_frames=source_frames,
                reference_audio=reference_audio,
            )
            row["transmitter"] = tx_name
            metrics.append(row)
            frames_by_tx[tx_name][profile] = reconstructed
            audio_by_tx[tx_name][profile] = recovered_audio
            print("eval " + json.dumps(row, sort_keys=True), flush=True)

    for profile in RENDER_PROFILES:
        candidate_comparison = label_frames([
            ("source | audio L", source_frames),
            (f"candidate {profile} | audio R", frames_by_tx["candidate"][profile]),
        ], columns=2)
        mux(
            out / f"{args.name}_{args.duration}s_candidate_{profile}.mp4",
            candidate_comparison,
            reference_audio,
            audio_by_tx["candidate"][profile],
        )
        paired = label_frames([
            ("source", source_frames),
            ("current | audio L", frames_by_tx["current"][profile]),
            ("low-PAPR candidate | audio R", frames_by_tx["candidate"][profile]),
        ], columns=3)
        mux(
            out / f"{args.name}_{args.duration}s_paired_{profile}.mp4",
            paired,
            audio_by_tx["current"][profile],
            audio_by_tx["candidate"][profile],
        )

    candidate_grid = label_frames(
        [("source", source_frames)] + [
            (profile, frames_by_tx["candidate"][profile]) for profile in PROFILES
        ],
        columns=4,
    )
    mux(
        out / f"{args.name}_{args.duration}s_candidate_channel_grid.mp4",
        candidate_grid,
        reference_audio,
        audio_by_tx["candidate"]["mpp6"],
    )
    Image.fromarray(candidate_grid[len(candidate_grid) // 2]).save(
        out / "candidate_channel_grid_midpoint.png"
    )

    report = {
        "input": str(Path(args.input).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "duration_seconds": args.duration,
        "voice_processor": args.voice_processor,
        "current_papr_steady_db": analytic_papr_db(current, steady),
        "current_papr_sustained_db": analytic_papr_db(current, sustained),
        "candidate_papr_steady_db": analytic_papr_db(selected_waveform, steady),
        "candidate_papr_sustained_db": analytic_papr_db(selected_waveform, sustained),
        "candidate_papr_complete_db": analytic_papr_db(selected_waveform),
        "candidate_average_w_at_100w_pep": float(
            100.0 / 10.0 ** (analytic_papr_db(selected_waveform) / 10.0)
        ),
        "selected": selected,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
