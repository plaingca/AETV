#!/usr/bin/env python3
"""Score release codecs on the shared protocol, with paired standard errors.

Eval split: first 64 clips of the seed-2026 shuffle, fade seeds 2026 + i*G + g.
``--split pool`` scores clips from the training pool instead (the clips after
the first 64), for choosing receiver settings without touching the eval split.

Receiver settings are confidence gains (see ``aetv.shared_eval.scale_confidence``).
All gains for one model are decoded from the same encode, AWGN draws and modem
waveforms, so ``paired`` deltas against the first gain are exact pairs.

    python scripts/eval_shared.py --models v8-face-gan v7-rxfix ac16 \
        --gains 1.0 --out runs/shared-eval/eval64.json
    python scripts/eval_shared.py --models v8-face-gan --split pool --clips 24 \
        --seed0 7300 --gains 1.0 1.25 1.5 1.75 2.0 --no-latent --out runs/shared-eval/pool.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.shared_eval import (  # noqa: E402
    DEFAULT_CACHE,
    FaceMasks,
    build_adapter,
    eval_split,
    load_clips,
    paired,
    score_clip,
    summarize,
    training_pool,
)

ROWS = ("clean", "awgn15", "awgn6", "modem_clean", "mpp12")


def gain_label(gain: float) -> str:
    return f"gain{gain:g}"


def collect(records: list[dict], label: str) -> dict:
    out: dict = {"rows": {}, "ssim": {}, "lpips": {}, "face": {}}
    for kind in out:
        keys = sorted({k for r in records for k in getattr(r[label], kind)})
        for key in keys:
            out[kind][key] = [getattr(r[label], kind).get(key) for r in records]
    out["failures"] = int(sum(r[label].failures for r in records))
    return out


def report(per_clip: dict, labels: list[str]) -> dict:
    base = per_clip[labels[0]]
    summary = {}
    for label in labels:
        values = per_clip[label]
        entry = {"failures": values["failures"]}
        for kind in ("rows", "ssim", "lpips", "face"):
            for key, series in values[kind].items():
                name = key if kind == "rows" else f"{kind}_{key}"
                entry[name] = summarize(series)
                if label != labels[0]:
                    entry[name]["paired_vs_" + labels[0]] = paired(series, base[kind][key])
        summary[label] = entry
    return summary


def save_frames(kept: dict, path: Path) -> None:
    torch.save(kept, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--gains", nargs="+", type=float, default=[1.0])
    parser.add_argument("--model-gain", action="append", default=[],
                        help="per-model extra gain, e.g. v8-face-gan=1.5 (added after --gains)")
    parser.add_argument("--split", choices=("eval", "pool"), default="eval")
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--pool-offset", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=2026)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--no-latent", action="store_true", help="skip clean / AWGN latent rows")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--frames-dir", help="save source and reconstructions for --frame-clips")
    parser.add_argument("--frame-clips", nargs="+", type=int, default=[0, 24, 48])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.split == "eval" and args.seed0 != 2026:
        parser.error("the eval split uses the published fade seeds (seed0 2026)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.split == "eval":
        paths = eval_split(args.cache, clips=args.clips)
    else:
        paths = training_pool(args.cache)[args.pool_offset : args.pool_offset + args.clips]
    clips = load_clips(paths)
    lpips_metric = None
    if not args.no_lpips:
        import lpips

        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    faces = None if args.no_face else FaceMasks()
    extra: dict[str, list[float]] = {}
    for item in args.model_gain:
        name, value = item.split("=")
        extra.setdefault(name, []).append(float(value))
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "split": args.split,
            "cache": args.cache,
            "clips": [p.name for p in paths],
            "seed0": args.seed0,
            "fade_seed": "seed0 + clip_index * gops_per_clip + gop_index",
            "pool_offset": args.pool_offset if args.split == "pool" else None,
        },
        "models": {},
    }
    for name in args.models:
        started = time.time()
        adapter = build_adapter(name, device)
        name = adapter.name
        gains = list(dict.fromkeys(list(args.gains) + extra.get(name, [])))
        labels = [gain_label(g) for g in gains]
        records = []
        for index in range(clips.shape[0]):
            mask = faces.mask(paths[index].name, clips[index]) if faces is not None else None
            keep = frames_dir is not None and index in args.frame_clips
            scored = score_clip(
                adapter, clips[index], index, device, dict(zip(labels, gains)), seed0=args.seed0,
                face_mask=mask, lpips_metric=lpips_metric, latent_rows=not args.no_latent, keep=keep,
            )
            if keep:
                scored, kept = scored
                save_frames(kept, frames_dir / f"{name}-{args.split}{index:02d}.pt")
            records.append(scored)
            if (index + 1) % 16 == 0:
                means = " ".join(f"{lb} {np.mean([r[lb].rows['mpp12'] for r in records]):.2f}" for lb in labels)
                print(f"  {name} {index + 1}/{clips.shape[0]} mpp12: {means}", flush=True)
        per_clip = {label: collect(records, label) for label in labels}
        result["models"][name] = {
            "gains": gains,
            "summary": report(per_clip, labels),
            "per_clip": per_clip,
            "seconds": time.time() - started,
        }
        best = max(labels, key=lambda lb: result["models"][name]["summary"][lb]["mpp12"]["mean"])
        result["models"][name]["best_mpp12"] = best
        for label in labels:
            s = result["models"][name]["summary"][label]["mpp12"]
            print(f"{name} {label}: mpp12 {s['mean']:.3f} ± {s['se']:.3f}", flush=True)
        print(f"{name}: best mpp12 setting {best}", flush=True)
        del adapter
        torch.cuda.empty_cache()
        out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
