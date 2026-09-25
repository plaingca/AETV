#!/usr/bin/env python3
"""Score the narrowband RVDJSCC checkpoint on the seed-2026 64-clip split.

The protocol matches the AC2K PSNR eval: three stateful 4-frame GOPs per
12-frame clip, clean PSNR, global SSIM, AlexNet LPIPS, and wire AWGN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.ac2k_psnr import load_psnr_checkpoint
from aetv.narrowband_jscc import global_ssim, impair_wire, psnr
from aetv.rvdjscc_narrowband import load_rvdjscc
from scripts.eval_ac2k_psnr import score_model, val_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/rvdjscc-narrowband-v2-2.2khz-best.pt")
    parser.add_argument("--baseline", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/rvdjscc-narrowband-v2/eval64.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    val, names = val_tensor(Path(args.cache), args.val_clips, args.seed)
    lpips_metric = None
    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}", flush=True)

    print("scoring rvdjscc", flush=True)
    model, payload = load_rvdjscc(args.checkpoint, device)
    improved = score_model(model, val, device, lpips_metric, (18.0, 15.0, 12.0, 6.0))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_report = None
    if args.baseline:
        print("scoring ac2k psnr baseline", flush=True)
        baseline = load_psnr_checkpoint(args.baseline, device)[0]
        baseline_report = score_model(baseline, val, device, lpips_metric, (18.0, 15.0, 12.0, 6.0))

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": payload.get("architecture"),
        "step": payload.get("step"),
        "bandwidth_khz": payload.get("bandwidth_khz"),
        "budget": payload.get("budget"),
        "key_len": (payload.get("model_config") or {}).get("key_len"),
        "interp_len": (payload.get("model_config") or {}).get("interp_len"),
        "seed": args.seed,
        "val_clips": names,
        "model": improved,
        "baseline_checkpoint": str(Path(args.baseline).resolve()) if args.baseline else "",
        "baseline": baseline_report,
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
