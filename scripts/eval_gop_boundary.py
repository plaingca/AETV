#!/usr/bin/env python3
"""GOP-boundary report on cached V8 ``mpp12`` decodes, with paired standard errors.

Rows: the codec output as decoded (``current``) and, for each ``--refiner``,
the same decode passed through that receiver-side refiner. ``mpp12`` PSNR is
the shared scorer's statistic (mean of per-GOP PSNR per clip). Boundary
statistics are defined in ``aetv.gop_boundary``.

    python scripts/eval_gop_boundary.py --cache runs/gop-boundary/eval64.pt \
        --refiner fixed=runs/gop-refiner/best.pt --out runs/gop-boundary/eval64-report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.gop_boundary import boundary_record, load_refiner  # noqa: E402
from aetv.shared_eval import clip_psnr, paired, summarize  # noqa: E402

GOP = 6
SCALARS = ("mpp12", "lpips_mpp12", "boundary_tpsnr", "interior_tpsnr", "boundary_gap_db", "boundary_jump", "interior_jump")


def gop_confidence_frames(gop_conf: torch.Tensor) -> torch.Tensor:
    """(B, G) -> (B, G * GOP)."""
    return gop_conf.repeat_interleave(GOP, dim=-1)


@torch.no_grad()
def score_cache(cache: dict, refiners: dict, device, lpips_metric=None, draw: int = 0) -> dict:
    out = {name: [] for name in ["current", *refiners]}
    for i in range(cache["source"].shape[0]):
        source = cache["source"][i].float().div(255).to(device)
        decoded = cache["decoded"][i, draw].float().to(device)
        conf = gop_confidence_frames(cache["gop_confidence"][i, draw][None].float()).to(device)
        variants = {"current": decoded}
        for name, model in refiners.items():
            variants[name] = model(decoded[None], conf)[0]
        for name, recon in variants.items():
            gops = [source[None, :, g * GOP : (g + 1) * GOP] for g in range(2)]
            recons = [recon[None, :, g * GOP : (g + 1) * GOP] for g in range(2)]
            record = boundary_record(source, recon, GOP)
            record["mpp12"] = clip_psnr(gops, recons)
            if lpips_metric is not None:
                record["lpips_mpp12"] = float(np.mean([
                    float(lpips_metric(r[0].permute(1, 0, 2, 3) * 2 - 1, s[0].permute(1, 0, 2, 3) * 2 - 1).mean())
                    for s, r in zip(gops, recons)
                ]))
            out[name].append(record)
    return out


def summarise(records: dict) -> dict:
    base = records["current"]
    summary = {}
    for name, rows in records.items():
        entry = {}
        for key in SCALARS:
            if key not in rows[0]:
                continue
            series = [r[key] for r in rows]
            entry[key] = summarize(series)
            if name != "current":
                entry[key]["paired_vs_current"] = paired(series, [r[key] for r in base])
        entry["position_psnr"] = np.mean([r["position_psnr"] for r in rows], 0).tolist()
        entry["transition_tpsnr"] = np.mean([r["transition_tpsnr"] for r in rows], 0).tolist()
        entry["frame_psnr"] = np.mean([r["frame_psnr"] for r in rows], 0).tolist()
        summary[name] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refiner", action="append", default=[], help="label=path")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    refiners = {}
    for item in args.refiner:
        label, path = item.split("=", 1)
        refiners[label] = load_refiner(path, device)
    lpips_metric = None
    if not args.no_lpips:
        import lpips

        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    records = score_cache(cache, refiners, device, lpips_metric)
    summary = summarise(records)
    for name, entry in summary.items():
        line = [name]
        for key in ("mpp12", "lpips_mpp12", "boundary_tpsnr", "interior_tpsnr", "boundary_jump"):
            if key in entry:
                s = entry[key]
                text = f"{key} {s['mean']:.3f}"
                if "paired_vs_current" in s:
                    p = s["paired_vs_current"]
                    text += f" (Δ {p['mean']:+.3f} ± {p['se']:.3f})"
                line.append(text)
        print("  ".join(line))
        print("   position psnr " + " ".join(f"{v:.2f}" for v in entry["position_psnr"]))
        print("   transition tpsnr " + " ".join(f"{v:.2f}" for v in entry["transition_tpsnr"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "cache": args.cache, "model": cache["model"], "seed0": cache["seed0"], "clips": cache["names"],
        "failures": cache["failures"], "refiners": args.refiner, "summary": summary,
        "per_clip": {name: rows for name, rows in records.items()},
    }, indent=1, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v))


if __name__ == "__main__":
    main()
