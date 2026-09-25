#!/usr/bin/env python3
"""Score the quantization-slack detail model against the AC2K PSNR fine-tune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.ac2k_psnr import load_psnr_checkpoint
from aetv.slack_detail import load_slack_detail
from scripts.eval_ac2k_psnr import score_model, val_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/slack-detail-2.2khz-best.pt")
    parser.add_argument("--baseline", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="runs/slack-detail/eval64.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    val, names = val_tensor(Path(args.cache), args.clips, args.seed)
    try:
        import lpips

        metric = lpips.LPIPS(net="alex").to(device)
    except Exception as exc:  # pragma: no cover - eval machines have lpips
        raise SystemExit(f"LPIPS is required for this protocol: {exc}") from exc
    snrs = (18.0, 15.0, 12.0, 6.0)
    model, _payload = load_slack_detail(args.checkpoint, device, ac2k_path=args.baseline)
    print("scoring slack-detail", flush=True)
    slack = score_model(model, val, device, metric, snrs)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    baseline, _ = load_psnr_checkpoint(args.baseline, device)
    print("scoring AC2K baseline", flush=True)
    base = score_model(baseline, val, device, metric, snrs)
    report = {"clips": names, "slack_detail": slack, "ac2k_psnr": base}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"slack_detail": slack, "ac2k_psnr": base}, indent=2), flush=True)


if __name__ == "__main__":
    main()
