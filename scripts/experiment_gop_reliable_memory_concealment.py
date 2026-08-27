#!/usr/bin/env python3
"""Evaluate bounded reliable-GOP memory and explicit erasure concealment.

This is a receiver-only follow-up to the V8 spatial reliability gate.  The
released codec, retained feature refiner, and selected spatial gate remain
unchanged.  A bounded receiver policy classifies each decoded GOP from pilot
SNR/coherence plus latent confidence, retains only the last reliable output,
and conceals at most two erased GOPs using either a last-frame hold or frozen
RAFT motion extrapolation.  Erased GOPs never update state; reacquisition uses
the released independent decode.

The controlled evaluation reports ordinary paired means and absolute p95/max
boundary outliers so a catastrophic semantic jump cannot hide behind averages.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES, AETVModeSpec  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    DEFAULT_CELLS,
    REPORT_METRICS,
    ChannelCell,
    SequenceCache,
    compare_reports,
    load_model,
    sequence_metrics,
    sha256_file,
)
from experiment_gop_feature_context import load_refiner  # noqa: E402
from experiment_gop_flow import RAFTAligner  # noqa: E402
from experiment_gop_reliability_gate import (  # noqa: E402
    MultiGOPSequenceCache,
    apply_reliability_sequence,
    cache_diagnostics_for_sequence,
    decode_received_features,
    join_many_gops,
    load_gate,
    multi_boundary_losses,
    prepare_event_batch,
    verify_standard_cache,
)


VARIANTS = ("released", "spatial", "memory_hold", "memory_flow")
SEAM_METRICS = (
    "boundary_excess",
    "boundary_error_ratio",
    "boundary_delta_lpips",
    "boundary_lowpass_step",
    "boundary_acceleration",
)


@dataclass(frozen=True)
class ConcealmentConfig:
    min_snr_db: float = -2.0
    min_pilot_coherence: float = 0.25
    min_mean_confidence: float = 0.45
    min_q10_confidence: float = 0.12
    strong_snr_db: float = 8.0
    strong_pilot_coherence: float = 0.75
    max_memory_gops: int = 2
    flow_decay: float = 0.82

    def validate(self) -> None:
        if not 0 <= self.min_pilot_coherence <= 1:
            raise ValueError("minimum pilot coherence must be in [0,1]")
        if not 0 <= self.min_mean_confidence <= 1:
            raise ValueError("minimum mean confidence must be in [0,1]")
        if not 0 <= self.min_q10_confidence <= 1:
            raise ValueError("minimum q10 confidence must be in [0,1]")
        if self.max_memory_gops < 1:
            raise ValueError("max memory GOPs must be positive")
        if not 0 < self.flow_decay <= 1:
            raise ValueError("flow decay must be in (0,1]")


def classify_reliable_gops(
    weights: torch.Tensor,
    snr_db: torch.Tensor,
    pilot_coherence: torch.Tensor,
    config: ConcealmentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fuse channel and confidence evidence while tolerating false-low confidence."""
    config.validate()
    if weights.ndim != 3 or snr_db.shape != weights.shape[:2]:
        raise ValueError("expected B,G,N weights and matching B,G diagnostics")
    mean = weights.mean(dim=2).clamp(0, 1)
    q10 = torch.quantile(weights.float(), 0.10, dim=2).to(weights.dtype).clamp(0, 1)
    channel_good = (
        (snr_db >= config.min_snr_db)
        & (pilot_coherence >= config.min_pilot_coherence)
    )
    confidence_good = (
        (mean >= config.min_mean_confidence)
        & (q10 >= config.min_q10_confidence)
    )
    channel_strong = (
        (snr_db >= config.strong_snr_db)
        & (pilot_coherence >= config.strong_pilot_coherence)
    )
    reliable = channel_good & (confidence_good | channel_strong)
    return reliable, {
        "mean_confidence": mean,
        "q10_confidence": q10,
        "channel_good": channel_good,
        "confidence_good": confidence_good,
        "channel_strong": channel_strong,
    }


def hold_concealment(memory_gop: torch.Tensor) -> torch.Tensor:
    if memory_gop.ndim != 5:
        raise ValueError("memory GOP must be BCTHW")
    return memory_gop[:, :, -1:].expand_as(memory_gop).clone()


def flow_concealment(
    memory_gop: torch.Tensor,
    aligner: RAFTAligner,
    *,
    decay: float,
) -> torch.Tensor:
    """Extrapolate the final reliable motion field without updating memory."""
    if memory_gop.ndim != 5 or memory_gop.shape[2] < 2:
        raise ValueError("motion concealment requires a BCTHW GOP with two frames")
    reference = memory_gop[:, :, -2]
    predicted = memory_gop[:, :, -1]
    with torch.no_grad():
        flow = aligner.estimate_flow(reference, predicted)
        frames = []
        for index in range(memory_gop.shape[2]):
            predicted = aligner.warp_with_flow(predicted, flow * (decay**index))
            frames.append(predicted)
    return torch.stack(frames, dim=2).clamp(0, 1)


def apply_bounded_memory_concealment(
    spatial_gops: torch.Tensor,
    reliable: torch.Tensor,
    aligner: RAFTAligner,
    config: ConcealmentConfig,
    *,
    mode: str,
) -> tuple[torch.Tensor, list[list[dict]]]:
    """Conceal unreliable GOPs from a bounded last-reliable or bootstrap state."""
    if mode not in {"hold", "flow"}:
        raise ValueError(f"unknown concealment mode: {mode}")
    if spatial_gops.ndim != 6 or reliable.shape != spatial_gops.shape[:2]:
        raise ValueError("expected BGCTHW GOPs and matching B,G reliability mask")
    config.validate()
    batch, count = spatial_gops.shape[:2]
    batch_outputs = []
    all_events: list[list[dict]] = []
    for row in range(batch):
        outputs = []
        events = []
        reliable_memory: torch.Tensor | None = None
        bootstrap_memory: torch.Tensor | None = None
        memory_age = 0
        for gop in range(count):
            current = spatial_gops[row : row + 1, gop]
            if bool(reliable[row, gop]):
                output = current
                reliable_memory = current
                bootstrap_memory = current
                memory_age = 0
                action = "reliable_update"
            else:
                memory = reliable_memory if reliable_memory is not None else bootstrap_memory
                if memory is None:
                    output = current
                    bootstrap_memory = current
                    action = "unreliable_bootstrap"
                elif memory_age < config.max_memory_gops:
                    if mode == "hold":
                        output = hold_concealment(memory)
                    else:
                        output = flow_concealment(memory, aligner, decay=config.flow_decay)
                    memory_age += 1
                    action = f"conceal_{mode}"
                else:
                    output = current
                    bootstrap_memory = current
                    action = "memory_expired_bypass"
            outputs.append(output)
            events.append(
                {
                    "gop": gop,
                    "reliable": bool(reliable[row, gop]),
                    "action": action,
                    "memory_age": memory_age,
                }
            )
        batch_outputs.append(torch.cat(outputs, dim=0))
        all_events.append(events)
    return torch.stack(batch_outputs, dim=0), all_events


def args_config(args: argparse.Namespace) -> ConcealmentConfig:
    return ConcealmentConfig(
        min_snr_db=args.min_snr_db,
        min_pilot_coherence=args.min_pilot_coherence,
        min_mean_confidence=args.min_mean_confidence,
        min_q10_confidence=args.min_q10_confidence,
        strong_snr_db=args.strong_snr_db,
        strong_pilot_coherence=args.strong_pilot_coherence,
        max_memory_gops=args.max_memory_gops,
        flow_decay=args.flow_decay,
    )


def spatial_decode(
    args: argparse.Namespace,
    model,
    refiner,
    gate,
    aligner,
    received: torch.Tensor,
    decode_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    snr: torch.Tensor,
    coherence: torch.Tensor,
    reliable: torch.Tensor,
    mode: AETVModeSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base, features, skips = decode_received_features(model, received, decode_weights, mode)
    reset = torch.zeros_like(reliable)
    spatial_all, _ = apply_reliability_sequence(
        refiner,
        gate,
        model.decoder,
        aligner,
        base,
        features,
        skips,
        gate_weights,
        snr,
        coherence,
        torch.ones_like(reliable),
        reset,
        photometric_threshold=args.photometric_threshold,
        photometric_softness=args.photometric_softness,
        scene_threshold_multiplier=args.scene_threshold_multiplier,
        scene_cut_threshold=args.scene_cut_threshold,
        min_previous_snr_db=args.spatial_min_previous_snr_db,
        min_previous_pilot_coherence=args.spatial_min_previous_pilot_coherence,
        output_flow_strength=args.output_flow_strength,
    )
    spatial_safe, _ = apply_reliability_sequence(
        refiner,
        gate,
        model.decoder,
        aligner,
        base,
        features,
        skips,
        gate_weights,
        snr,
        coherence,
        reliable,
        reset,
        photometric_threshold=args.photometric_threshold,
        photometric_softness=args.photometric_softness,
        scene_threshold_multiplier=args.scene_threshold_multiplier,
        scene_cut_threshold=args.scene_cut_threshold,
        min_previous_snr_db=args.spatial_min_previous_snr_db,
        min_previous_pilot_coherence=args.spatial_min_previous_pilot_coherence,
        output_flow_strength=args.output_flow_strength,
    )
    return base, spatial_all, spatial_safe


def standard_diagnostics(
    rx_cache: dict, cell: ChannelCell, index: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return cache_diagnostics_for_sequence(rx_cache, cell, index, device)


def evaluate_standard(
    args: argparse.Namespace,
    model,
    refiner,
    gate,
    aligner,
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> tuple[dict, dict]:
    config = args_config(args)
    rows = {
        variant: {cell.label: [] for cell in DEFAULT_CELLS}
        for variant in VARIANTS
    }
    detection = {cell.label: [] for cell in DEFAULT_CELLS}
    timings = {variant: [] for variant in VARIANTS if variant != "released"}
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
                weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
                snr, coherence = standard_diagnostics(rx_cache, cell, index, device)
                reliable, evidence = classify_reliable_gops(weights, snr, coherence, config)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                base, spatial, spatial_safe = spatial_decode(
                    args,
                    model,
                    refiner,
                    gate,
                    aligner,
                    received,
                    weights,
                    weights,
                    snr,
                    coherence,
                    reliable,
                    mode,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings["spatial"].append(time.perf_counter() - started)
                variant_gops = {"released": base, "spatial": spatial}
                for conceal_mode, label in (("hold", "memory_hold"), ("flow", "memory_flow")):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    concealed, events = apply_bounded_memory_concealment(
                        spatial_safe,
                        reliable,
                        aligner,
                        config,
                        mode=conceal_mode,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timings[label].append(time.perf_counter() - started)
                    variant_gops[label] = concealed
                    if conceal_mode == "hold":
                        detection[cell.label].append(
                            {
                                "sequence": index,
                                "reliable": reliable[0].tolist(),
                                "mean_confidence": evidence["mean_confidence"][0].tolist(),
                                "q10_confidence": evidence["q10_confidence"][0].tolist(),
                                "snr_db": snr[0].tolist(),
                                "pilot_coherence": coherence[0].tolist(),
                                "actions": events[0],
                            }
                        )
                for label, gops in variant_gops.items():
                    reconstruction = join_many_gops(gops)
                    rows[label][cell.label].append(
                        sequence_metrics(
                            reconstruction,
                            source,
                            mode.gop_frames,
                            device,
                            include_lpips=True,
                        )
                    )
            print(f"  standard sequence {index + 1:>2}/{len(dataset)}", flush=True)
    reports = {}
    for label in VARIANTS:
        reports[label] = {
            "cells": [asdict(cell) for cell in DEFAULT_CELLS],
            "sequences": rows[label],
        }
        if label in timings:
            reports[label]["runtime_ms_per_sequence"] = 1000 * st.mean(timings[label])
    return reports, detection


def _gop_l1(sequence: torch.Tensor, target: torch.Tensor, gop: int, frames: int) -> float:
    start = gop * frames
    return float(F.l1_loss(sequence[:, :, start : start + frames], target[:, :, start : start + frames]))


def evaluate_failure_cases(
    args: argparse.Namespace,
    model,
    refiner,
    gate,
    aligner,
    dataset: MultiGOPSequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    config = args_config(args)
    names = (
        "good_fade_good",
        "false_high_confidence",
        "false_low_confidence",
        "missing_gop",
        "random_reset",
        "scene_cut",
    )
    result = {name: {variant: [] for variant in VARIANTS} for name in names}
    with torch.inference_mode():
        for name in names:
            for start in range(0, len(dataset), args.batch):
                indices_list = list(range(start, min(start + args.batch, len(dataset))))
                if name == "scene_cut" and len(indices_list) < 2:
                    indices_list = [indices_list[0], (indices_list[0] + 1) % len(dataset)]
                indices = torch.tensor(indices_list, dtype=torch.long)
                source = torch.stack([dataset[index] for index in indices_list]).to(device)
                event = prepare_event_batch(name, source, indices, rx_cache, mode, device)
                reliable, _ = classify_reliable_gops(
                    event.gate_weights, event.snr, event.coherence, config
                )
                reliable = reliable & event.valid
                base, spatial, spatial_safe = spatial_decode(
                    args,
                    model,
                    refiner,
                    gate,
                    aligner,
                    event.received,
                    event.decode_weights,
                    event.gate_weights,
                    event.snr,
                    event.coherence,
                    reliable,
                    mode,
                )
                variants = {"released": base, "spatial": spatial}
                hold, _ = apply_bounded_memory_concealment(
                    spatial_safe, reliable, aligner, config, mode="hold"
                )
                flow, _ = apply_bounded_memory_concealment(
                    spatial_safe, reliable, aligner, config, mode="flow"
                )
                variants["memory_hold"] = hold
                variants["memory_flow"] = flow
                released_sequence = join_many_gops(base)
                for label, gops in variants.items():
                    sequence = join_many_gops(gops)
                    for row, sequence_index in enumerate(indices_list):
                        cross = multi_boundary_losses(
                            sequence[row : row + 1],
                            event.source[row : row + 1],
                            mode.gop_frames,
                        )
                        recovery = _gop_l1(
                            sequence[row : row + 1],
                            event.source[row : row + 1],
                            event.recovery_gop,
                            mode.gop_frames,
                        )
                        independent = _gop_l1(
                            released_sequence[row : row + 1],
                            event.source[row : row + 1],
                            event.recovery_gop,
                            mode.gop_frames,
                        )
                        result[name][label].append(
                            {
                                "sequence": sequence_index,
                                "recovery_error_ratio": recovery / max(independent, 1e-12),
                                "full_l1": float(F.l1_loss(sequence[row], event.source[row])),
                                "boundary_rgb_delta": float(cross["boundary_rgb_delta"]),
                                "boundary_lowpass_step": float(cross["boundary_lowpass_step"]),
                                "boundary_acceleration": float(cross["boundary_acceleration"]),
                                "reliable_gops": reliable[row].tolist(),
                            }
                        )
            print(f"  failure case {name}", flush=True)
    return result


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def outlier_summary(report: dict) -> dict:
    output = {}
    for cell, rows in report["sequences"].items():
        output[cell] = {}
        for metric in ("boundary_error_ratio", "boundary_delta_lpips", "boundary_excess"):
            values = [row[metric] for row in rows]
            output[cell][metric] = {
                "mean": st.mean(values),
                "p95": percentile(values, 0.95),
                "max": max(values),
                "max_sequence": values.index(max(values)),
            }
    return output


def significant_regression(metric: dict, tolerance: float = 0.0) -> bool:
    delta = metric["paired_delta"]
    return delta["mean"] - delta["two_se"] > tolerance


def triggered_quality_reasons(
    label: str,
    candidate: dict,
    reference: dict,
    detection: dict,
    *,
    ratio_tolerance: float = 0.02,
    lpips_tolerance: float = 0.001,
) -> list[str]:
    """Reject seam improvements that freeze/blur the sequences actually concealed."""
    reasons = []
    for cell, rows in detection.items():
        for detection_row in rows:
            if not any(
                item["action"].startswith("conceal_")
                for item in detection_row["actions"]
            ):
                continue
            sequence = detection_row["sequence"]
            before = reference["sequences"][cell][sequence]
            after = candidate["sequences"][cell][sequence]
            for metric, description in (
                ("within_motion_ratio", "motion ratio"),
                ("spatial_detail_ratio", "spatial detail ratio"),
            ):
                if (
                    after[metric] < 1.0
                    and after[metric] < before[metric] - ratio_tolerance
                ):
                    reasons.append(
                        f"{label} {cell} sequence {sequence} triggered {description} fell "
                        f"{before[metric]:.3f}->{after[metric]:.3f}"
                    )
            if after["lpips"] > before["lpips"] + lpips_tolerance:
                reasons.append(
                    f"{label} {cell} sequence {sequence} triggered LPIPS regressed "
                    f"{before['lpips']:.3f}->{after['lpips']:.3f}"
                )
    return reasons


def assess_variant(
    label: str,
    comparison: dict,
    outliers: dict,
    candidate: dict,
    reference: dict,
    detection: dict,
    *,
    max_ratio: float,
    max_boundary_lpips: float,
) -> dict:
    reasons = []
    for cell, metrics in comparison["cells"].items():
        if significant_regression(metrics["lpips"], tolerance=0.001):
            delta = metrics["lpips"]["paired_delta"]
            reasons.append(
                f"{cell} LPIPS regressed {delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
            )
        if significant_regression(metrics["within_gop_temporal_error"]):
            delta = metrics["within_gop_temporal_error"]["paired_delta"]
            reasons.append(
                f"{cell} within-GOP temporal error regressed "
                f"{delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
            )
        motion = metrics["within_motion_ratio"]["paired_delta"]
        if motion["mean"] + motion["two_se"] < -0.02:
            reasons.append(f"{cell} motion ratio fell {motion['mean']:+.6f}")
        detail = metrics["spatial_detail_ratio"]["paired_delta"]
        if detail["mean"] + detail["two_se"] < -0.02:
            reasons.append(f"{cell} spatial detail ratio fell {detail['mean']:+.6f}")
        if outliers[cell]["boundary_error_ratio"]["max"] > max_ratio:
            item = outliers[cell]["boundary_error_ratio"]
            reasons.append(
                f"{cell} max boundary ratio {item['max']:.3f} at sequence "
                f"{item['max_sequence']} exceeds {max_ratio:.3f}"
            )
        if outliers[cell]["boundary_delta_lpips"]["max"] > max_boundary_lpips:
            item = outliers[cell]["boundary_delta_lpips"]
            reasons.append(
                f"{cell} max boundary LPIPS {item['max']:.3f} at sequence "
                f"{item['max_sequence']} exceeds {max_boundary_lpips:.3f}"
            )
    reasons.extend(
        triggered_quality_reasons(label, candidate, reference, detection)
    )
    return {
        "variant": label,
        "accepted": not reasons,
        "decision": "retain" if not reasons else "reject",
        "reasons": reasons,
        "absolute_limits": {
            "boundary_error_ratio": max_ratio,
            "boundary_delta_lpips": max_boundary_lpips,
        },
    }


def write_report(
    destination: Path,
    args: argparse.Namespace,
    evaluations: dict,
    comparisons: dict,
    outliers: dict,
    decisions: dict,
    detection: dict,
    failure: dict,
) -> None:
    lines = [
        "# V8 bounded reliable-memory erasure concealment",
        "",
        f"Released checkpoint SHA-256: `{sha256_file(args.checkpoint)}`",
        f"Spatial gate SHA-256: `{sha256_file(args.gate)}`",
        "",
        (
            "Receiver-only policy; codec, wire contract, feature refiner, and spatial gate "
            "are unchanged. Memory is capped at two GOPs and erased GOPs never update state."
        ),
        "",
        "## Decisions",
        "",
        "| Variant | Decision | Reasons |",
        "|---|---|---|",
    ]
    for label, decision in decisions.items():
        reason = "; ".join(decision["reasons"]) or "all mean, outlier, LPIPS, motion, and detail gates passed"
        lines.append(f"| {label} | **{decision['decision'].upper()}** | {reason} |")

    metrics = tuple(dict.fromkeys(REPORT_METRICS))
    for cell in (item.label for item in DEFAULT_CELLS):
        lines.extend(
            (
                "",
                f"## {cell}",
                "",
                "| Model | " + " | ".join(metrics) + " |",
                "|---|" + "---:|" * len(metrics),
            )
        )
        for label in VARIANTS:
            rows = evaluations[label]["sequences"][cell]
            values = [st.mean(row[metric] for row in rows) for metric in metrics]
            lines.append(f"| {label} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
        lines.extend(
            (
                "",
                "| Model | ratio p95 | ratio max (seq) | boundary LPIPS p95 | boundary LPIPS max (seq) |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for label in VARIANTS:
            ratio = outliers[label][cell]["boundary_error_ratio"]
            lpips = outliers[label][cell]["boundary_delta_lpips"]
            lines.append(
                f"| {label} | {ratio['p95']:.4f} | {ratio['max']:.4f} ({ratio['max_sequence']}) "
                f"| {lpips['p95']:.4f} | {lpips['max']:.4f} ({lpips['max_sequence']}) |"
            )

    lines.extend(("", "## Concealment incidence", "", "| Cell | concealed sequences |", "|---|---:|"))
    for cell, rows in detection.items():
        concealed = sum(any(item["action"].startswith("conceal_") for item in row["actions"]) for row in rows)
        lines.append(f"| {cell} | {concealed}/{len(rows)} |")

    triggered = [
        (cell, row["sequence"])
        for cell, rows in detection.items()
        for row in rows
        if any(item["action"].startswith("conceal_") for item in row["actions"])
    ]
    if triggered:
        lines.extend(
            (
                "",
                "## Triggered-sequence audit",
                "",
                "| Cell / sequence | Model | boundary ratio | boundary LPIPS | low-pass step | acceleration | motion ratio | detail ratio | full LPIPS |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for cell, sequence in triggered:
            for label in VARIANTS:
                row = evaluations[label]["sequences"][cell][sequence]
                lines.append(
                    f"| {cell} / {sequence} | {label} | "
                    f"{row['boundary_error_ratio']:.6f} | "
                    f"{row['boundary_delta_lpips']:.6f} | "
                    f"{row['boundary_lowpass_step']:.6f} | "
                    f"{row['boundary_acceleration']:.6f} | "
                    f"{row['within_motion_ratio']:.6f} | "
                    f"{row['spatial_detail_ratio']:.6f} | {row['lpips']:.6f} |"
                )

    lines.extend(
        (
            "",
            "## Failure recovery",
            "",
            "| Event | Variant | mean recovery ratio | worst recovery ratio | mean full L1 |",
            "|---|---|---:|---:|---:|",
        )
    )
    for event, variants in failure.items():
        for label, rows in variants.items():
            recovery = [row["recovery_error_ratio"] for row in rows]
            full_l1 = [row["full_l1"] for row in rows]
            lines.append(
                f"| {event} | {label} | {st.mean(recovery):.6f} | {max(recovery):.6f} | {st.mean(full_l1):.6f} |"
            )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_comparisons(
    args: argparse.Namespace,
    model,
    refiner,
    gate,
    aligner,
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    config = args_config(args)
    destination = args.out / "renders"
    destination.mkdir(parents=True, exist_ok=True)
    sources = torch.cat([dataset[index] for index in range(len(dataset))], dim=1).unsqueeze(0)
    manifest = {
        "gui_boundary_blending": False,
        "frames_per_sequence": 12,
        "sequences": len(dataset),
        "total_frames_per_render": 12 * len(dataset),
        "fps": mode.fps,
        "files": {},
    }
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            clips = {variant: [] for variant in VARIANTS}
            for index in range(len(dataset)):
                received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
                weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
                snr, coherence = standard_diagnostics(rx_cache, cell, index, device)
                reliable, _ = classify_reliable_gops(weights, snr, coherence, config)
                base, spatial, spatial_safe = spatial_decode(
                    args,
                    model,
                    refiner,
                    gate,
                    aligner,
                    received,
                    weights,
                    weights,
                    snr,
                    coherence,
                    reliable,
                    mode,
                )
                hold, _ = apply_bounded_memory_concealment(
                    spatial_safe, reliable, aligner, config, mode="hold"
                )
                flow, _ = apply_bounded_memory_concealment(
                    spatial_safe, reliable, aligner, config, mode="flow"
                )
                current = {
                    "released": base,
                    "spatial": spatial,
                    "memory_hold": hold,
                    "memory_flow": flow,
                }
                for label, gops in current.items():
                    clips[label].append(join_many_gops(gops).cpu())
            panels = [("Source", sources)] + [
                (label, torch.cat(clips[label], dim=2)) for label in VARIANTS
            ]
            path = destination / f"full-paired-32-{cell.label}-no-gui-blend.mp4"
            write_labeled_grid_mp4(panels, path, fps=mode.fps, columns=3)
            manifest["files"][cell.label] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            print(f"Rendered {path}", flush=True)
    (destination / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "render", "all"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    parser.add_argument("--refiner", type=Path, default=Path("runs/gop-feature-context-v8/refiner.pt"))
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("runs/v8-reliability-gated-feature-context-safety-20260826/gate-spatial.pt"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/v8-bounded-memory-concealment-20260826")
    )
    parser.add_argument(
        "--standard-eval-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_3gop_eval")
    )
    parser.add_argument(
        "--standard-eval-rx-cache",
        type=Path,
        default=Path("runs/v8-two-gop-boundary-sweep-explicit-20260826/eval-runtime-rx.pt"),
    )
    parser.add_argument(
        "--failure-eval-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_5gop_eval")
    )
    parser.add_argument(
        "--failure-eval-rx-cache",
        type=Path,
        default=Path("runs/v8-reliability-gated-feature-context-20260826/failure-eval-runtime-rx-5gop.pt"),
    )
    parser.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--failure-eval-sequences", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--min-snr-db", type=float, default=-2.0)
    parser.add_argument("--min-pilot-coherence", type=float, default=0.25)
    parser.add_argument("--min-mean-confidence", type=float, default=0.45)
    parser.add_argument("--min-q10-confidence", type=float, default=0.12)
    parser.add_argument("--strong-snr-db", type=float, default=8.0)
    parser.add_argument("--strong-pilot-coherence", type=float, default=0.75)
    parser.add_argument("--max-memory-gops", type=int, default=2)
    parser.add_argument("--flow-decay", type=float, default=0.82)
    parser.add_argument("--max-boundary-ratio", type=float, default=3.0)
    parser.add_argument("--max-boundary-lpips", type=float, default=0.5)
    parser.add_argument("--output-flow-strength", type=float, default=0.10)
    parser.add_argument("--photometric-threshold", type=float, default=0.10)
    parser.add_argument("--photometric-softness", type=float, default=0.02)
    parser.add_argument("--scene-threshold-multiplier", type=float, default=1.5)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.15)
    parser.add_argument("--spatial-min-previous-snr-db", type=float, default=-2.0)
    parser.add_argument("--spatial-min-previous-pilot-coherence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode = AETV_MODES[args.mode]
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    standard_dataset = SequenceCache(args.standard_eval_cache, limit=args.eval_sequences)
    failure_dataset = MultiGOPSequenceCache(
        args.failure_eval_cache, 5, mode.gop_frames, args.failure_eval_sequences
    )
    standard_rx = verify_standard_cache(args, standard_dataset)
    failure_rx = torch.load(args.failure_eval_rx_cache, map_location="cpu", weights_only=False)
    model = load_model(args.checkpoint, mode, device).eval()
    refiner, refiner_payload = load_refiner(args.refiner, device)
    gate, gate_payload = load_gate(args.gate, device)
    if refiner_payload["source_sha256"] != sha256_file(args.checkpoint):
        raise SystemExit("refiner source checkpoint mismatch")
    if gate_payload["source_sha256"] != sha256_file(args.checkpoint):
        raise SystemExit("gate source checkpoint mismatch")
    for module in (model, refiner, gate):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    aligner = RAFTAligner(device).eval()
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "refiner": str(args.refiner.resolve()),
        "refiner_sha256": sha256_file(args.refiner),
        "gate": str(args.gate.resolve()),
        "gate_sha256": sha256_file(args.gate),
        "mode": args.mode,
        "concealment": asdict(args_config(args)),
        "absolute_promotion_limits": {
            "boundary_error_ratio": args.max_boundary_ratio,
            "boundary_delta_lpips": args.max_boundary_lpips,
        },
        "wire_contract_changed": False,
        "gui_blending": False,
    }
    (args.out / "experiment-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if args.command in {"evaluate", "all"}:
        evaluations, detection = evaluate_standard(
            args,
            model,
            refiner,
            gate,
            aligner,
            standard_dataset,
            standard_rx,
            mode,
            device,
        )
        comparisons = {
            label: {
                "vs_released": compare_reports(evaluations["released"], evaluations[label]),
                "vs_spatial": compare_reports(evaluations["spatial"], evaluations[label]),
            }
            for label in ("memory_hold", "memory_flow")
        }
        outliers = {label: outlier_summary(report) for label, report in evaluations.items()}
        decisions = {
            label: assess_variant(
                label,
                comparisons[label]["vs_released"],
                outliers[label],
                evaluations[label],
                evaluations["spatial"],
                detection,
                max_ratio=args.max_boundary_ratio,
                max_boundary_lpips=args.max_boundary_lpips,
            )
            for label in ("memory_hold", "memory_flow")
        }
        failure = evaluate_failure_cases(
            args,
            model,
            refiner,
            gate,
            aligner,
            failure_dataset,
            failure_rx,
            mode,
            device,
        )
        (args.out / "evaluation.json").write_text(
            json.dumps(evaluations, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "comparisons.json").write_text(
            json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "outliers.json").write_text(
            json.dumps(outliers, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "detection.json").write_text(
            json.dumps(detection, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "decisions.json").write_text(
            json.dumps(decisions, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "failure-evaluation.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        write_report(
            args.out / "report.md",
            args,
            evaluations,
            comparisons,
            outliers,
            decisions,
            detection,
            failure,
        )
        for label, decision in decisions.items():
            print(f"{label}: {decision['decision'].upper()}", flush=True)
            for reason in decision["reasons"]:
                print(f"  - {reason}", flush=True)
        if args.command == "evaluate":
            return

    if args.command in {"render", "all"}:
        render_comparisons(
            args,
            model,
            refiner,
            gate,
            aligner,
            standard_dataset,
            standard_rx,
            mode,
            device,
        )


if __name__ == "__main__":
    main()
