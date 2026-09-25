#!/usr/bin/env python3
"""Fit the common-mode patch basis carried in AC2K's quantization slack.

The basis is the top two principal components of 4x4 patches of the temporal-mean
residual, estimated on the training split only. No gradient steps: this is the
linear code that the clean-channel slack can carry exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.slack_detail import CODE, SlackDetail, fit_patch_basis
from scripts.finetune_ac2k_psnr import load_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--fit-clips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ac2k", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--checkpoint", default="models/slack-detail-2.2khz-best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_clips, _val_clips, val_names = load_clips(Path(args.cache), args.val_clips, args.seed)
    model = SlackDetail(args.ac2k).to(device)
    print(f"fitting patch basis on {min(args.fit_clips, train_clips.shape[0])} train clips", flush=True)
    fit_patch_basis(model, train_clips, device, args.fit_clips)
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "ac2k_checkpoint": model.ac2k_path,
            "basis": model.basis.detach().cpu(),
            "patch_mean": model.patch_mean.detach().cpu(),
            "code_coordinates": CODE,
            "fit_clips": min(args.fit_clips, train_clips.shape[0]),
            "val_clips": val_names,
            "budget": 2816,
        },
        path,
    )
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
