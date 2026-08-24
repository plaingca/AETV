#!/usr/bin/env python3
"""Benchmark release-model encode/decode latency on the current machine."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from aetv.codec import AETVCodec


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(call, device: torch.device, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        call()
        _sync(device)
        values.append(time.perf_counter() - started)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("V7", "V8"), default="V8")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch default")
    parser.add_argument("--channels-last", action="store_true", help="test channels-last 3D model layout")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, help="also write machine-readable results")
    args = parser.parse_args()

    if args.threads:
        os.environ["AETV_CPU_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)
    codec = AETVCodec(args.checkpoint, device=args.device, mode=args.mode)
    if args.channels_last:
        codec.model.to(memory_format=torch.channels_last_3d)
    rng = np.random.default_rng(20260824)
    frames = rng.integers(
        0,
        256,
        (codec.mode.gop_frames, codec.mode.height, codec.mode.width, 3),
        dtype=np.uint8,
    )
    latents = np.zeros(codec.mode.latents_per_gop, dtype=np.float32)
    weights = np.ones_like(latents)

    for _ in range(args.warmup):
        codec.encode_gop(frames)
        codec.decode_gop(latents, weights)
    if codec.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(codec.device)
    encode = _measure(lambda: codec.encode_gop(frames), codec.device, args.repeats)
    decode = _measure(lambda: codec.decode_gop(latents, weights), codec.device, args.repeats)
    cycle = [a + b for a, b in zip(encode, decode)]
    result = {
        "mode": codec.mode.name,
        "device": str(codec.device),
        "checkpoint": str(codec.checkpoint_path),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(codec.device) if codec.device.type == "cuda" else None,
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated(codec.device) / (1024 * 1024), 1)
            if codec.device.type == "cuda"
            else None
        ),
        "encode_median_ms": round(statistics.median(encode) * 1000, 2),
        "decode_median_ms": round(statistics.median(decode) * 1000, 2),
        "cycle_median_ms": round(statistics.median(cycle) * 1000, 2),
        "encode_realtime_headroom": round(1.0 / statistics.median(encode), 2),
        "decode_realtime_headroom": round(1.0 / statistics.median(decode), 2),
        "full_duplex_headroom": round(1.0 / statistics.median(cycle), 2),
        "repeats": args.repeats,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
