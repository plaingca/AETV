#!/usr/bin/env python3
"""Evaluate VQ-DeepVSC-style adaptive key frames over the V8 HF waveform.

Two native-12-fps source intervals are independently scheduled into the six
video slots of each one-second V8 GOP.  The released encoder/decoder, analog
latent budget, modem, channel emulator, and demodulator remain frozen.  Every
key-count arm sees the same source clips and deterministic channel seeds.

This is a wire-format experiment: the receiver is given each 11-bit key-frame
position mask out of band.  The report records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.adaptive_keyframes import (  # noqa: E402
    KeyframeSchedule,
    adaptive_schedule,
    collapse_repetitions,
    pack_keyframes,
    reconstruct_timeline,
    uniform_schedule,
)
from aetv.config import AETV_MODES  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    DEFAULT_CELLS,
    decode_independent_gops,
    load_model,
    runtime_channel_seed,
    runtime_retry_seed,
    runtime_transmit_gops,
    sequence_metrics,
    sha256_file,
)
from experiment_v8_receiver_postprocess import decode_high_fps, mean_metrics  # noqa: E402


SOURCE_FRAMES_PER_GOP = 11
SOURCE_GAP_FRAMES = 1


def schedules_for_clip(
    target: torch.Tensor,
    keyframes: int,
    codec_slots: int,
    *,
    uniform: bool = False,
) -> tuple[KeyframeSchedule, KeyframeSchedule]:
    schedules = []
    for start in (0, SOURCE_FRAMES_PER_GOP + SOURCE_GAP_FRAMES):
        segment = target[:, :, start : start + SOURCE_FRAMES_PER_GOP].permute(0, 2, 1, 3, 4)
        if uniform:
            schedules.append(uniform_schedule(SOURCE_FRAMES_PER_GOP, codec_slots))
        else:
            schedules.append(adaptive_schedule(segment, keyframes, codec_slots=codec_slots))
    return schedules[0], schedules[1]


def pack_clip(target: torch.Tensor, schedules: tuple[KeyframeSchedule, KeyframeSchedule]) -> torch.Tensor:
    packed = []
    for start, schedule in zip((0, SOURCE_FRAMES_PER_GOP + SOURCE_GAP_FRAMES), schedules):
        segment = target[:, :, start : start + SOURCE_FRAMES_PER_GOP].permute(0, 2, 1, 3, 4)
        packed.append(pack_keyframes(segment, schedule))
    return torch.stack(packed, dim=1)[0]


def reconstruct_clip(
    decoded: torch.Tensor,
    schedules: tuple[KeyframeSchedule, KeyframeSchedule],
) -> torch.Tensor:
    segments = []
    for index, schedule in enumerate(schedules):
        keys = collapse_repetitions(decoded[index : index + 1], schedule)
        segments.append(reconstruct_timeline(keys, schedule, SOURCE_FRAMES_PER_GOP))
    bridge = 0.5 * (segments[0][:, :, -1:] + segments[1][:, :, :1])
    return torch.cat((segments[0], bridge, segments[1]), dim=2)


def schedule_json(schedule: KeyframeSchedule) -> dict:
    return {
        "positions": list(schedule.positions),
        "repeats": list(schedule.repeats),
        "importance": list(schedule.importance),
    }


def paired_comparisons(rows: dict) -> dict:
    """Summarize paired arm-minus-uniform deltas and standard errors."""
    result = {}
    for cell, arms in rows.items():
        baseline = arms["uniform_k6"]
        result[cell] = {}
        for arm, values in arms.items():
            if arm == "uniform_k6":
                continue
            metrics = {}
            for metric in baseline[0]:
                deltas = [candidate[metric] - control[metric] for control, candidate in zip(baseline, values)]
                mean = st.mean(deltas)
                sem = st.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                metrics[metric] = {"mean_delta": mean, "sem": sem}
            result[cell][arm] = metrics
    return result


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    mode = AETV_MODES[args.mode]
    if mode.gop_frames != 6:
        raise ValueError("this experiment currently targets the six-slot V8 GOP")
    codec = load_model(args.checkpoint, mode, device).eval()
    key_counts = tuple(sorted(set(args.key_counts), reverse=True))
    if any(value < 2 or value > mode.gop_frames for value in key_counts):
        raise ValueError("every key count must be in [2, 6]")
    arm_specs = [("uniform_k6", mode.gop_frames, True)] + [
        (f"adaptive_k{value}", value, False) for value in key_counts
    ]
    arms = [name for name, _, _ in arm_specs]
    rows = {cell.label: {arm: [] for arm in arms} for cell in DEFAULT_CELLS}
    rendered = {cell.label: {arm: [] for arm in arms} for cell in DEFAULT_CELLS}
    targets = []
    schedule_records = []
    high_frames = 2 * (2 * mode.gop_frames) - 1

    for clip in range(args.clips):
        target = decode_high_fps(
            args.input,
            args.start + clip * args.clip_stride,
            high_frames,
            mode.fps * 2,
            mode.width,
            mode.height,
        ).unsqueeze(0).to(device)
        clip_schedules = {}
        clip_latents = {}
        with torch.inference_mode():
            for arm, keyframes, use_uniform in arm_specs:
                schedules = schedules_for_clip(
                    target,
                    keyframes,
                    mode.gop_frames,
                    uniform=use_uniform,
                )
                packed = pack_clip(target, schedules)
                latents = torch.stack(
                    [codec.encoder(packed[index : index + 1]) for index in range(2)], dim=1
                )[0]
                clip_schedules[arm] = schedules
                clip_latents[arm] = latents.float().cpu().numpy()
        schedule_records.append(
            {
                "clip": clip,
                "start_s": args.start + clip * args.clip_stride,
                "arms": {
                    arm: [schedule_json(value) for value in clip_schedules[arm]]
                    for arm in arms
                },
            }
        )

        for cell_index, cell in enumerate(DEFAULT_CELLS):
            initial_seed = runtime_channel_seed(args.seed, clip, cell_index)
            for arm in arms:
                for attempt in range(32):
                    try:
                        received, weights, _ = runtime_transmit_gops(
                            clip_latents[arm],
                            mode,
                            cell,
                            seed=runtime_retry_seed(initial_seed, attempt),
                        )
                        break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError(f"could not recover clip {clip} {arm} under {cell.label}")
                with torch.inference_mode():
                    decoded = decode_independent_gops(
                        codec,
                        torch.from_numpy(received).unsqueeze(0).to(device),
                        torch.from_numpy(weights).unsqueeze(0).to(device),
                        mode,
                    )
                    output = reconstruct_clip(decoded, clip_schedules[arm])
                    rows[cell.label][arm].append(
                        sequence_metrics(
                            output,
                            target,
                            mode.gop_frames * 2,
                            device,
                            include_lpips=not args.no_lpips,
                        )
                    )
                    rendered[cell.label][arm].append(output.cpu())
        targets.append(target.cpu())
        print(f"adaptive key frames {clip + 1:>2}/{args.clips}", flush=True)

    report = {
        "schema": 1,
        "paper": "VQ-DeepVSC, arXiv:2409.03393v1",
        "adaptation": {
            "temporal_stage": "greedy source-referenced interpolation-residual key-frame selection",
            "retransmission_proxy": "repeat important selected frames within the fixed six-slot V8 GOP",
            "receiver": "collapse repeated decoded slots, then variable-gap linear interpolation",
            "codec_changed": False,
            "waveform_changed": False,
            "wire_compatible": False,
            "wire_blocker": "receiver is given an 11-bit key-position mask per GOP out of band",
            "digital_vq_ldpc_adopted": False,
        },
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_fps": mode.fps * 2,
        "transmitted_slots_per_second": mode.gop_frames,
        "clips": args.clips,
        "frames_per_clip": high_frames,
        "seed": args.seed,
        "cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "schedules": schedule_records,
        "results": {
            cell: {
                arm: {"mean": mean_metrics(values), "sequences": values}
                for arm, values in arms_rows.items()
            }
            for cell, arms_rows in rows.items()
        },
        "comparisons_to_uniform": paired_comparisons(rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_render:
        source = torch.cat(targets, dim=2)
        for cell in DEFAULT_CELLS:
            panels = [("Source 12 fps", source)] + [
                (arm, torch.cat(rendered[cell.label][arm], dim=2)) for arm in arms
            ]
            write_labeled_grid_mp4(
                panels,
                args.report.parent / f"adaptive-keyframes-{cell.label}.mp4",
                fps=mode.fps * 2,
                columns=2,
            )
    print(f"saved {args.report}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    result.add_argument("--mode", default="V8")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--seed", type=int, default=240903393)
    result.add_argument("--start", type=float, default=0.0)
    result.add_argument("--clips", type=int, default=32)
    result.add_argument("--clip-stride", type=float, default=2.0)
    result.add_argument("--key-counts", type=int, nargs="+", default=(6, 5, 4, 3))
    result.add_argument("--report", type=Path, default=Path("runs/vq-deepvsc-hf-adaptation/paired-32.json"))
    result.add_argument("--no-lpips", action="store_true")
    result.add_argument("--no-render", action="store_true")
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
