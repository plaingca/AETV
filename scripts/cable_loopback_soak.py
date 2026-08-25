"""Measure live modem latency and latent error through the configured audio cable.

This intentionally bypasses the neural codec so a run isolates the soundcard,
virtual cable, Voicemeeter, OFDM modem, and receive scheduler. It never keys a
radio; it opens the input/output endpoints saved in AETV settings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from aetv.audio_io import open_input_stream, play_chunk_stream
from aetv.config import AETV_MODES, RELEASE_MODES
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.ringbuffer import RingBuffer
from aetv.settings import load_settings


def _match_latents(
    originals: list[np.ndarray], recovered: list[np.ndarray]
) -> list[dict]:
    if not originals or not recovered:
        return []
    reference = np.asarray(originals, dtype=np.float64)
    observed = np.asarray(recovered, dtype=np.float64)
    reference_norm = np.linalg.norm(reference, axis=1)
    observed_norm = np.linalg.norm(observed, axis=1)
    correlations = observed @ reference.T
    correlations /= np.maximum(
        observed_norm[:, None] * reference_norm[None, :], 1e-12
    )
    matches = []
    for recovered_index, row in enumerate(correlations):
        original_index = int(np.argmax(row))
        error = observed[recovered_index] - reference[original_index]
        nmse = float(
            np.mean(error**2)
            / max(float(np.mean(reference[original_index] ** 2)), 1e-12)
        )
        matches.append(
            {
                "recovered_index": recovered_index,
                "original_index": original_index,
                "correlation": float(row[original_index]),
                "nmse": nmse,
            }
        )
    return matches


def run(args: argparse.Namespace) -> dict:
    settings = load_settings(Path(args.settings) if args.settings else None)
    mode_name = args.mode or settings.mode
    if mode_name not in RELEASE_MODES:
        raise ValueError(f"mode must be one of {', '.join(RELEASE_MODES)}")
    mode = AETV_MODES[mode_name]
    fs = mode.geometry.fs
    tx_level = float(settings.tx_level if args.tx_level is None else args.tx_level)
    if not 0.05 <= tx_level <= 1.0:
        raise ValueError("TX level must be between 0.05 and 1.0")

    rng = np.random.default_rng(args.seed)
    originals = [
        rng.standard_normal(mode.latents_per_gop).astype(np.float32)
        for _ in range(args.gops)
    ]

    waveform_chunks = []
    for chunk in modulate_continuous_chunks(
        originals,
        mode_name=mode_name,
        callsign=settings.callsign,
        total_gops=args.gops,
    ):
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        waveform_chunks.append(
            chunk * (tx_level / peak)
            if peak > 0
            else np.asarray(chunk, dtype=np.float32)
        )

    # Decode the exact pre-cable waveform once. Raw latent NMSE includes the
    # modem's intentional clipping/equalization contract; comparing the live
    # recovery with this reference isolates error added by the audio transport.
    clean_receiver = StreamingDemodulator(
        mode.band,
        continuous=True,
        mode_name=mode_name,
        timing_tracking=True,
    )
    clean_recovered = []
    clean_audio = np.concatenate(waveform_chunks)
    for start in range(0, len(clean_audio), max(1, fs // 10)):
        for result in clean_receiver.feed(
            clean_audio[start : start + max(1, fs // 10)]
        ):
            clean_recovered.extend(result.gops_latents)

    def tx_chunks():
        yield from waveform_chunks

    ring = RingBuffer(seconds=max(30.0, args.gops + args.settle_s + args.tail_s + 5.0), fs=fs)
    source_gaps = []
    audio_errors = []
    stream, native_rate = open_input_stream(
        settings.audio_input or None,
        ring,
        fs,
        on_error=audio_errors.append,
        on_discontinuity=lambda: source_gaps.append(time.time()),
    )
    events = []
    demodulator = StreamingDemodulator(
        mode.band,
        continuous=True,
        mode_name=mode_name,
        timing_tracking=True,
        on_debug=events.append,
    )
    recovered = []
    weights = []
    result_times = []
    stop = threading.Event()
    receive_stats = {
        "max_feed_results": 0,
        "max_ring_backlog_s": 0.0,
        "max_demod_ms": 0.0,
    }

    def receive() -> None:
        cursor = 0
        while not stop.wait(0.02):
            audio, cursor, overrun = ring.read_since(cursor)
            if overrun:
                source_gaps.append(time.time())
            if not audio.size:
                continue
            started = time.perf_counter()
            results = demodulator.feed(audio)
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            receive_stats["max_demod_ms"] = max(
                receive_stats["max_demod_ms"], elapsed_ms
            )
            receive_stats["max_feed_results"] = max(
                receive_stats["max_feed_results"], len(results)
            )
            with ring.lock:
                backlog = max(0.0, (ring.total_written - cursor) / fs)
            receive_stats["max_ring_backlog_s"] = max(
                receive_stats["max_ring_backlog_s"], backlog
            )
            for result in results:
                recovered.extend(result.gops_latents)
                weights.extend(result.gops_weights)
                result_times.extend([time.time()] * len(result.gops_latents))

    worker = threading.Thread(target=receive, name="aetv-cable-soak-rx", daemon=True)
    worker.start()
    try:
        time.sleep(max(0.0, args.settle_s))
        tx_started = time.time()
        completed = play_chunk_stream(
            tx_chunks(), fs, device=settings.audio_output or None
        )
        tx_finished = time.time()
        time.sleep(max(0.0, args.tail_s))
    finally:
        stop.set()
        worker.join(timeout=5.0)
        stream.stop()
        stream.close()

    matches = _match_latents(originals, recovered)
    clean_matches = _match_latents(originals, clean_recovered)
    clean_by_original = {
        item["original_index"]: clean_recovered[item["recovered_index"]]
        for item in clean_matches
    }
    transport_matches = []
    for item in matches:
        clean = clean_by_original.get(item["original_index"])
        if clean is None:
            continue
        observed = np.asarray(recovered[item["recovered_index"]], dtype=np.float64)
        reference = np.asarray(clean, dtype=np.float64)
        denominator = max(
            float(np.linalg.norm(observed) * np.linalg.norm(reference)), 1e-12
        )
        transport_matches.append(
            {
                "recovered_index": item["recovered_index"],
                "original_index": item["original_index"],
                "correlation": float(np.dot(observed, reference) / denominator),
                "nmse": float(
                    np.mean((observed - reference) ** 2)
                    / max(float(np.mean(reference**2)), 1e-12)
                ),
            }
        )
    unique_matches = len({item["original_index"] for item in matches})
    duplicate_matches = len(matches) - unique_matches
    ordered_matches = sum(
        item["original_index"] > previous["original_index"]
        for previous, item in zip(matches, matches[1:])
    )
    accepted = [event for event in events if event.get("event") == "gop_accepted"]
    realignments = [
        event for event in events if event.get("event") == "tracking_realign"
    ]
    losses = [event for event in events if event.get("event") == "tracking_lost"]
    timing = [
        float(event["pilot_timing_ppm"])
        for event in accepted
        if isinstance(event.get("pilot_timing_ppm"), (int, float))
        and math.isfinite(event["pilot_timing_ppm"])
    ]
    confidence = np.concatenate(weights) if weights else np.zeros(0)
    gop_confidence = [float(np.mean(item)) for item in weights]
    first_latency = (
        result_times[0] - tx_started if result_times else None
    )
    transport_correlations = [
        item["correlation"] for item in transport_matches
    ]
    transport_nmses = [item["nmse"] for item in transport_matches]
    passed = (
        bool(completed)
        and len(recovered) == args.gops
        and unique_matches == args.gops
        and duplicate_matches == 0
        and ordered_matches == max(0, args.gops - 1)
        and len(transport_matches) == args.gops
        and min(transport_correlations, default=-1.0)
        >= args.min_transport_correlation
        and max(transport_nmses, default=math.inf) <= args.max_transport_nmse
        and not losses
        and not source_gaps
        and not audio_errors
    )
    return {
        "passed": passed,
        "mode": mode_name,
        "sample_rate": fs,
        "native_input_rate": native_rate,
        "gops_sent": args.gops,
        "gops_recovered": len(recovered),
        "unique_gops_recovered": unique_matches,
        "duplicate_or_false_gops": duplicate_matches,
        "ordered_transitions": ordered_matches,
        "completed": bool(completed),
        "tx_level": tx_level,
        "tx_wall_seconds": tx_finished - tx_started,
        "first_result_seconds_after_tx_call": first_latency,
        "median_latent_correlation": (
            float(np.median([item["correlation"] for item in matches]))
            if matches
            else None
        ),
        "worst_latent_correlation": (
            float(min(item["correlation"] for item in matches))
            if matches
            else None
        ),
        "median_latent_nmse": (
            float(np.median([item["nmse"] for item in matches]))
            if matches
            else None
        ),
        "worst_latent_nmse": (
            float(max(item["nmse"] for item in matches))
            if matches
            else None
        ),
        "median_transport_correlation": (
            float(np.median(transport_correlations))
            if transport_correlations
            else None
        ),
        "worst_transport_correlation": (
            float(min(transport_correlations))
            if transport_correlations
            else None
        ),
        "median_transport_nmse": (
            float(np.median(transport_nmses))
            if transport_nmses
            else None
        ),
        "worst_transport_nmse": (
            float(max(transport_nmses))
            if transport_nmses
            else None
        ),
        "min_transport_correlation_required": args.min_transport_correlation,
        "max_transport_nmse_required": args.max_transport_nmse,
        "mean_symbol_confidence": (
            float(np.mean(confidence)) if confidence.size else None
        ),
        "gop_symbol_confidence": gop_confidence,
        "median_timing_ppm": float(np.median(timing)) if timing else None,
        "timing_ppm_range": [float(min(timing)), float(max(timing))]
        if timing
        else None,
        "tracking_realignments": len(realignments),
        "tracking_losses": len(losses),
        "source_discontinuities": len(source_gaps),
        "audio_errors": audio_errors,
        **receive_stats,
        "matches": matches,
        "transport_matches": transport_matches,
        "realignments": realignments,
        "accepted_gops": [
            {
                key: event.get(key)
                for key in (
                    "stream_sample",
                    "tracked",
                    "pilot_coherence",
                    "pilot_occupancy",
                    "payload_confidence",
                    "snr_db",
                    "pilot_evm_pct",
                    "pilot_timing_ppm",
                )
            }
            for event in accepted
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gops", type=int, default=30)
    parser.add_argument("--mode", choices=RELEASE_MODES)
    parser.add_argument("--settings", help="optional StationSettings JSON path")
    parser.add_argument("--tx-level", type=float)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--min-transport-correlation", type=float, default=0.95)
    parser.add_argument("--max-transport-nmse", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.gops < 1:
        parser.error("--gops must be at least 1")
    if not -1.0 <= args.min_transport_correlation <= 1.0:
        parser.error("--min-transport-correlation must be between -1 and 1")
    if args.max_transport_nmse < 0.0:
        parser.error("--max-transport-nmse must be non-negative")
    report = run(args)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if not args.quiet:
        print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
