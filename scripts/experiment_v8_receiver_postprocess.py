#!/usr/bin/env python3
"""Evaluate receiver-only RIFE interpolation and SNR-conditioned restoration.

The released V8 checkpoint, encoder, decoder, OFDM payload, and channel path
remain frozen.  Restoration is trained on clean-minus-decoded residuals from
the existing immutable runtime RX cache.  Interpolation is scored against
real 12 fps frames: only the even 6 fps frames are sent through V8.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES  # noqa: E402
from aetv.receiver_postprocess import (  # noqa: E402
    LinearFrameInterpolator,
    RIFEInterpolator,
    SNRConditionedDenoiser,
    SNRConditionedResidualDiffusion,
    interpolate_video,
)
from eval import write_labeled_grid_mp4  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    DEFAULT_CELLS,
    SequenceCache,
    compare_reports,
    decode_cached_sequence,
    decode_independent_gops,
    join_gops,
    load_model,
    runtime_retry_seed,
    runtime_transmit_gops,
    sequence_metrics,
    sha256_file,
)


def receiver_conditions(rx_cache: dict, cell: str, indices: list[int], device: torch.device):
    snr = []
    for index in indices:
        details = rx_cache["diagnostics"][cell][index]["gops"]
        snr.append(st.mean(float(item["snr_db"]) for item in details))
    confidence = rx_cache["weights"][cell][indices].mean(dim=(1, 2))
    return (
        torch.tensor(snr, device=device, dtype=torch.float32),
        confidence.to(device=device, dtype=torch.float32),
    )


def load_restorer(path: Path, device: torch.device) -> SNRConditionedResidualDiffusion:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SNRConditionedResidualDiffusion(
        SNRConditionedDenoiser(
            width=int(payload["config"]["width"]),
            condition_width=int(payload["config"]["condition_width"]),
        ),
        timesteps=int(payload["config"]["timesteps"]),
        max_correction=float(payload["config"]["max_correction"]),
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def train_restorer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    mode = AETV_MODES[args.mode]
    dataset = SequenceCache(args.train_cache, limit=args.limit)
    rx_cache = torch.load(args.train_rx_cache, map_location="cpu", weights_only=False)
    codec = load_model(args.checkpoint, mode, device).eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    restorer = SNRConditionedResidualDiffusion(
        SNRConditionedDenoiser(args.width, args.condition_width),
        timesteps=args.timesteps,
        max_correction=args.max_correction,
    ).to(device)
    optimizer = torch.optim.AdamW(restorer.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(
        list(range(len(dataset))),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = iter(loader)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            index_tensor = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            index_tensor = next(iterator)
        indices = [int(value) for value in index_tensor]
        cell = DEFAULT_CELLS[(step - 1) % len(DEFAULT_CELLS)]
        clean = torch.stack([dataset[index] for index in indices]).to(device)
        with torch.inference_mode():
            received = rx_cache["received"][cell.label][indices].to(device)
            weights = rx_cache["weights"][cell.label][indices].to(device)
            decoded = decode_independent_gops(codec, received, weights, mode)
            degraded = join_gops(decoded, len(indices), count=2)
            snr, confidence = receiver_conditions(rx_cache, cell.label, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = restorer.training_loss(clean, degraded, snr, confidence)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(restorer.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        row = {
            "step": step,
            "cell": cell.label,
            "loss": float(loss.detach()),
            "indices": indices,
        }
        history.append(row)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"step {step:>5}/{args.steps} {cell.label:<11} "
                f"loss={float(loss.detach()):.6f}",
                flush=True,
            )
    payload = {
        "kind": "aetv-v8-snr-conditioned-residual-diffusion",
        "model": {name: value.detach().cpu() for name, value in restorer.state_dict().items()},
        "config": {
            "width": args.width,
            "condition_width": args.condition_width,
            "timesteps": args.timesteps,
            "max_correction": args.max_correction,
            "schedule": "cosine-alpha-bar",
            "residual_domain": "clamp((clean-degraded)/max_correction,-1,1)",
        },
        "experiment": {
            "released_checkpoint": str(args.checkpoint.resolve()),
            "released_checkpoint_sha256": sha256_file(args.checkpoint),
            "runtime_rx_cache": str(args.train_rx_cache.resolve()),
            "runtime_rx_cache_sha256": sha256_file(args.train_rx_cache),
            "steps": args.steps,
            "seed": args.seed,
            "cells": [asdict(cell) for cell in DEFAULT_CELLS],
            "wire_contract_changed": False,
            "codec_frozen": True,
            "elapsed_s": time.perf_counter() - started,
        },
    }
    args.restorer.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.restorer)
    args.restorer.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {args.restorer}")


def evaluate_restorer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    mode = AETV_MODES[args.mode]
    dataset = SequenceCache(args.eval_cache, limit=args.limit)
    rx_cache = torch.load(args.eval_rx_cache, map_location="cpu", weights_only=False)
    codec = load_model(args.checkpoint, mode, device).eval()
    restorer = load_restorer(args.restorer, device)
    rows = {"baseline": {cell.label: [] for cell in DEFAULT_CELLS}, "restored": {cell.label: [] for cell in DEFAULT_CELLS}}
    with torch.inference_mode():
        for index in range(len(dataset)):
            target = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                degraded = decode_cached_sequence(codec, rx_cache, cell, index, mode, device)
                snr, confidence = receiver_conditions(rx_cache, cell.label, [index], device)
                restored = restorer.restore(
                    degraded, snr, confidence, steps=args.sample_steps, seed=args.seed + index
                )
                for label, value in (("baseline", degraded), ("restored", restored)):
                    rows[label][cell.label].append(
                        sequence_metrics(
                            value, target, mode.gop_frames, device, include_lpips=not args.no_lpips
                        )
                    )
            print(f"evaluated {index + 1:>2}/{len(dataset)}", flush=True)
    baseline = {"cells": [asdict(cell) for cell in DEFAULT_CELLS], "sequences": rows["baseline"]}
    restored = {"cells": [asdict(cell) for cell in DEFAULT_CELLS], "sequences": rows["restored"]}
    report = {"baseline": baseline, "restored": restored, "comparison": compare_reports(baseline, restored)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.report}")


def decode_high_fps(path: Path, start_s: float, frames: int, fps: float, width: int, height: int) -> torch.Tensor:
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{start_s:.6f}", "-i", str(path),
        "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    expected = frames * height * width * 3
    if proc.returncode or len(proc.stdout) != expected:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])
    array = np.frombuffer(proc.stdout, dtype=np.uint8).copy().reshape(frames, height, width, 3)
    return torch.from_numpy(array).permute(3, 0, 1, 2).float().div(255.0)


def frame_hold(video: torch.Tensor, factor: int = 2) -> torch.Tensor:
    return torch.repeat_interleave(video, factor, dim=2)[:, :, : (video.shape[2] - 1) * factor + 1]


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: st.mean(row[key] for row in rows) for key in rows[0]}


def evaluate_rife(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    mode = AETV_MODES[args.mode]
    codec = load_model(args.checkpoint, mode, device).eval()
    rife = RIFEInterpolator(args.rife_repo, args.rife_weights, device=device, scale=args.rife_scale)
    linear = LinearFrameInterpolator()
    arms = ("hold", "linear", "rife")
    rows = {cell.label: {arm: [] for arm in arms} for cell in DEFAULT_CELLS}
    rendered = {cell.label: {arm: [] for arm in arms} for cell in DEFAULT_CELLS}
    targets = []
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
        low = target[:, :, ::2]
        separated = low.reshape(1, 3, 2, mode.gop_frames, mode.height, mode.width).permute(0, 2, 1, 3, 4, 5)
        with torch.inference_mode():
            latents = torch.stack([codec.encoder(separated[:, index]) for index in range(2)], dim=1)[0]
        for cell_index, cell in enumerate(DEFAULT_CELLS):
            initial_seed = args.seed + 1009 * clip + 9176 * cell_index
            for attempt in range(32):
                try:
                    received, weights, _ = runtime_transmit_gops(
                        latents.float().cpu().numpy(), mode, cell, seed=runtime_retry_seed(initial_seed, attempt)
                    )
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(f"could not recover clip {clip} under {cell.label}")
            with torch.inference_mode():
                decoded = decode_independent_gops(
                    codec,
                    torch.from_numpy(received).unsqueeze(0).to(device),
                    torch.from_numpy(weights).unsqueeze(0).to(device),
                    mode,
                )
                degraded = join_gops(decoded, 1, count=2)
                outputs = {
                    "hold": frame_hold(degraded),
                    "linear": interpolate_video(degraded, linear, factor=2),
                    "rife": interpolate_video(degraded, rife, factor=2),
                }
                for arm, output in outputs.items():
                    rows[cell.label][arm].append(
                        sequence_metrics(
                            output, target, mode.gop_frames * 2, device, include_lpips=not args.no_lpips
                        )
                    )
                    rendered[cell.label][arm].append(output.cpu())
        targets.append(target.cpu())
        print(f"RIFE paired clip {clip + 1:>2}/{args.clips}", flush=True)
    report = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_fps": mode.fps * 2,
        "transmitted_fps": mode.fps,
        "output_fps": mode.fps * 2,
        "clips": args.clips,
        "frames_per_clip": high_frames,
        "cells": {
            cell: {arm: {"mean": mean_metrics(values), "sequences": values} for arm, values in arms_rows.items()}
            for cell, arms_rows in rows.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    source = torch.cat(targets, dim=2)
    for cell in DEFAULT_CELLS:
        panels = [("Source 12 fps", source)] + [
            (arm, torch.cat(rendered[cell.label][arm], dim=2)) for arm in arms
        ]
        write_labeled_grid_mp4(
            panels, args.report.parent / f"rife-paired-{cell.label}.mp4", fps=mode.fps * 2, columns=2
        )
    print(f"saved {args.report}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("train-restorer", "evaluate-restorer", "evaluate-rife"))
    result.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    result.add_argument("--mode", default="V8")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--seed", type=int, default=260826)
    result.add_argument("--limit", type=int)
    result.add_argument("--train-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_3gop_train"))
    result.add_argument("--eval-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    result.add_argument("--train-rx-cache", type=Path, default=Path("runs/v8-two-gop-boundary-sweep-explicit-20260826/train-runtime-rx.pt"))
    result.add_argument("--eval-rx-cache", type=Path, default=Path("runs/v8-two-gop-boundary-sweep-explicit-20260826/eval-runtime-rx.pt"))
    result.add_argument("--restorer", type=Path, default=Path("runs/v8-rife-diffusion-20260826/restorer.pt"))
    result.add_argument("--report", type=Path, default=Path("runs/v8-rife-diffusion-20260826/report.json"))
    result.add_argument("--steps", type=int, default=2000)
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--lr", type=float, default=2e-4)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--condition-width", type=int, default=128)
    result.add_argument("--timesteps", type=int, default=100)
    result.add_argument("--sample-steps", type=int, default=12)
    result.add_argument("--max-correction", type=float, default=0.25)
    result.add_argument("--log-interval", type=int, default=25)
    result.add_argument("--no-lpips", action="store_true")
    result.add_argument("--input", type=Path)
    result.add_argument("--start", type=float, default=0.0)
    result.add_argument("--clips", type=int, default=32)
    result.add_argument("--clip-stride", type=float, default=2.0)
    result.add_argument("--rife-repo", type=Path, default=Path("runs/v8-rife-diffusion-20260826/external/Practical-RIFE"))
    result.add_argument("--rife-weights", type=Path, default=Path("runs/v8-rife-diffusion-20260826/external/rife-v4.25-lite/train_log"))
    result.add_argument("--rife-scale", type=float, default=1.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train-restorer":
        train_restorer(args)
    elif args.command == "evaluate-restorer":
        evaluate_restorer(args)
    else:
        if args.input is None:
            raise SystemExit("evaluate-rife requires --input with native >=12 fps source")
        evaluate_rife(args)


if __name__ == "__main__":
    main()
