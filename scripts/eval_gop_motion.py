#!/usr/bin/env python3
"""Motion preservation of V8 receivers on cached ``mpp12`` decodes.

For each transition t (frame t-1 to t) of every clip and receiver:

- **energy**: mean squared frame difference; the motion ratio is reconstructed
  over source energy, pooled as a ratio of sums.
- **flow**: Farneback optical-flow magnitude on luma; the flow ratio is
  reconstructed over source magnitude (ratio of sums), and ``epe`` is the mean
  endpoint error of the reconstructed flow against the source flow.
- **tpsnr**: PSNR of the reconstructed frame difference against the source's.

Transitions are split into inside-GOP / boundary and healthy / faded. A GOP is
faded when its mean modem confidence is below ``--faded-threshold``; a boundary
is healthy only if both GOPs are. ``high-motion`` restricts to clips whose
source flow is in the top quartile.

    python scripts/eval_gop_motion.py --cache runs/gop-boundary/eval64.pt \
        --refiner hq=models/v8-gop-refiner-hq.pt --out runs/gop-boundary/eval64-motion.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.gop_boundary import refiner_from_spec  # noqa: E402
from aetv.shared_eval import paired  # noqa: E402
from eval_gop_boundary import GOP, gop_confidence_frames  # noqa: E402

LUMA = torch.tensor([0.299, 0.587, 0.114])


def luma_u8(clip: torch.Tensor) -> np.ndarray:
    """(3, T, H, W) in [0, 1] -> (T, H, W) uint8 luma."""
    y = torch.einsum("c,cthw->thw", LUMA.to(clip), clip.float())
    return (y.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def flows(luma: np.ndarray) -> np.ndarray:
    """(T-1, H, W, 2) Farneback flow for consecutive frames."""
    return np.stack([
        cv2.calcOpticalFlowFarneback(luma[t - 1], luma[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for t in range(1, luma.shape[0])
    ])


def transitions(source: torch.Tensor, recon: torch.Tensor, flow_s: np.ndarray) -> list[dict]:
    ds = source[:, 1:].float() - source[:, :-1].float()
    dr = recon[:, 1:].float() - recon[:, :-1].float()
    es = ds.square().mean(dim=(0, 2, 3)).cpu().numpy()
    er = dr.square().mean(dim=(0, 2, 3)).cpu().numpy()
    terr = (dr - ds).square().mean(dim=(0, 2, 3)).cpu().numpy()
    flow_r = flows(luma_u8(recon))
    mag_s = np.linalg.norm(flow_s, axis=-1).mean(axis=(1, 2))
    mag_r = np.linalg.norm(flow_r, axis=-1).mean(axis=(1, 2))
    epe = np.linalg.norm(flow_r - flow_s, axis=-1).mean(axis=(1, 2))
    return [{"es": float(es[i]), "er": float(er[i]), "terr": float(terr[i]), "fs": float(mag_s[i]),
             "fr": float(mag_r[i]), "epe": float(epe[i])} for i in range(len(es))]


def label(t: int, gop_faded: list[bool]) -> tuple[str, str]:
    """Kind and health of transition t (frame t-1 -> t)."""
    if t % GOP == 0:
        faded = gop_faded[t // GOP - 1] or gop_faded[t // GOP]
        return "boundary", "faded" if faded else "healthy"
    return "inside", "faded" if gop_faded[t // GOP] else "healthy"


def psnr(mse: float) -> float:
    return 100.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse)


def pooled(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    s = {k: sum(r[k] for r in rows) for k in ("es", "er", "fs", "fr", "epe", "terr")}
    return {"n": len(rows), "energy_ratio": s["er"] / s["es"], "flow_ratio": s["fr"] / s["fs"],
            "epe": s["epe"] / len(rows), "tpsnr": float(np.mean([psnr(r["terr"]) for r in rows]))}


def clip_ratio(rows: list[dict], key_r: str, key_s: str) -> float | None:
    s = sum(r[key_s] for r in rows)
    return None if not rows or s <= 0 else sum(r[key_r] for r in rows) / s


@torch.no_grad()
def collect(cache: dict, refiners: dict, device, faded_threshold: float = 0.35) -> tuple[dict, list[float], int]:
    """Per-transition motion rows for ``current`` and each refiner, source flow per clip, faded GOP count."""
    rows = {name: [] for name in ["current", *refiners]}
    source_flow, faded_count = [], 0
    for i in range(cache["source"].shape[0]):
        source = cache["source"][i].float().div(255).to(device)
        decoded = cache["decoded"][i, 0].float().to(device)
        if cache["decoded"].dtype == torch.uint8:
            decoded = decoded / 255
        gconf = cache["gop_confidence"][i, 0].float()
        gop_faded = [bool(c < faded_threshold) for c in gconf]
        faded_count += sum(gop_faded)
        conf = gop_confidence_frames(gconf[None]).to(device)
        flow_s = flows(luma_u8(source))
        source_flow.append(float(np.linalg.norm(flow_s, axis=-1).mean()))
        variants = {"current": decoded, **{k: m(decoded[None], conf)[0] for k, m in refiners.items()}}
        for name, recon in variants.items():
            for t, rec in enumerate(transitions(source, recon, flow_s), start=1):
                kind, health = label(t, gop_faded)
                rows[name].append({**rec, "clip": i, "t": t, "kind": kind, "health": health})
    return rows, source_flow, faded_count


def healthy_motion_vs_current(rows: dict, name: str) -> dict:
    """Pooled inside-GOP healthy energy and flow ratios of ``name`` over ``current``."""
    keep = lambda r: r["kind"] == "inside" and r["health"] == "healthy"  # noqa: E731
    ours, base = pooled([r for r in rows[name] if keep(r)]), pooled([r for r in rows["current"] if keep(r)])
    return {"energy": ours["energy_ratio"] / base["energy_ratio"], "flow": ours["flow_ratio"] / base["flow_ratio"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refiner", action="append", default=[], help="label=path or label=path@healthy_gain")
    ap.add_argument("--faded-threshold", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    refiners = dict(refiner_from_spec(item, device) for item in args.refiner)
    names = ["current", *refiners]
    n = cache["source"].shape[0]
    rows, source_flow, faded_count = collect(cache, refiners, device, args.faded_threshold)
    high = set(np.argsort(source_flow)[-(n // 4):].tolist())

    groups = {
        "inside/healthy": lambda r: r["kind"] == "inside" and r["health"] == "healthy",
        "inside/faded": lambda r: r["kind"] == "inside" and r["health"] == "faded",
        "boundary/healthy": lambda r: r["kind"] == "boundary" and r["health"] == "healthy",
        "boundary/faded": lambda r: r["kind"] == "boundary" and r["health"] == "faded",
        "inside/healthy/high-motion": lambda r: r["kind"] == "inside" and r["health"] == "healthy" and r["clip"] in high,
        "all": lambda r: True,
    }
    summary = {}
    for name in names:
        entry = {}
        for g, keep in groups.items():
            sel = [r for r in rows[name] if keep(r)]
            entry[g] = pooled(sel)
            if name != "current" and sel:
                base = [r for r in rows["current"] if keep(r)]
                per = lambda rs, kr, ks: [clip_ratio([r for r in rs if r["clip"] == c], kr, ks) for c in range(n)]
                ours, ref = per(sel, "er", "es"), per(base, "er", "es")
                logs = [None if a is None or b is None or a <= 0 or b <= 0 else math.log(a / b) for a, b in zip(ours, ref)]
                entry[g]["energy_log_ratio_vs_current"] = paired(logs, [0.0] * n)
                ours, ref = per(sel, "fr", "fs"), per(base, "fr", "fs")
                logs = [None if a is None or b is None or a <= 0 or b <= 0 else math.log(a / b) for a, b in zip(ours, ref)]
                entry[g]["flow_log_ratio_vs_current"] = paired(logs, [0.0] * n)
        summary[name] = entry
    for g in groups:
        print(f"== {g} (n={summary['current'][g]['n']})")
        for name in names:
            e = summary[name][g]
            if not e["n"]:
                continue
            extra = ""
            if "energy_log_ratio_vs_current" in e:
                a, b = e["energy_log_ratio_vs_current"], e["flow_log_ratio_vs_current"]
                extra = (f"  vs current: energy x{math.exp(a['mean']):.3f} (±{a['se']:.3f} log) "
                         f"flow x{math.exp(b['mean']):.3f} (±{b['se']:.3f} log)")
            print(f"  {name:10s} energy {e['energy_ratio']:.3f} flow {e['flow_ratio']:.3f} "
                  f"epe {e['epe']:.3f} tpsnr {e['tpsnr']:.2f}{extra}")
    Path(args.out).write_text(json.dumps({
        "cache": args.cache, "refiners": args.refiner, "faded_threshold": args.faded_threshold,
        "faded_gops": faded_count, "high_motion_clips": sorted(high), "summary": summary,
        "per_transition": rows,
    }, indent=1))


if __name__ == "__main__":
    main()
