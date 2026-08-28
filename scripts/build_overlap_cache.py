#!/usr/bin/env python3
"""Build genuinely contiguous multi-GOP clips for the overlap experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aetv.config import AETV_MODES
from aetv.video_data import HFDatasetsVideoDataset, VideoClipSpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    ap.add_argument("--gops", type=int, default=5)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--dataset", default="lance-format/Openvid-1M")
    args = ap.parse_args()
    mode = AETV_MODES[args.mode]
    args.out.mkdir(parents=True, exist_ok=True)
    spec = VideoClipSpec(
        frames=args.gops * mode.gop_frames,
        fps=mode.fps,
        height=mode.height,
        width=mode.width,
    )
    stream = iter(HFDatasetsVideoDataset(
        dataset=args.dataset,
        split="train",
        spec=spec,
        epoch_size=args.count,
        seed=args.seed,
        shuffle_buffer=32,
    ))
    for index in range(args.count):
        clip = next(stream)
        torch.save((clip * 255).round().clamp(0, 255).byte(), args.out / f"sequence_{index:04d}.pt")
        print(f"{index + 1}/{args.count} {tuple(clip.shape)}", flush=True)


if __name__ == "__main__":
    main()
