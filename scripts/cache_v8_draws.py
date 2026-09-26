#!/usr/bin/env python3
"""Cache V8 decodes through the real modem + ``mpp12`` for GOP-boundary work.

Each 12-frame clip is two V8 GOPs. The encoder output goes through the V8 OFDM
modem and ``emulate(..., "mpp12")``; the frozen decoder output is cached with
the source and each GOP's mean modem confidence.

Fade seeds match the shared scorer: ``seed0 + clip * 2 + gop`` for draw 0, and
``seed0 + ((draw * clips) + clip) * 2 + gop`` in general.

    python scripts/cache_v8_draws.py --split eval --out runs/gop-boundary/eval64.pt
    python scripts/cache_v8_draws.py --split select --out runs/gop-boundary/select24.pt
    python scripts/cache_v8_draws.py --split train --draws 4 --out runs/gop-boundary/train.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.shared_eval import AutoencoderAdapter, eval_split, load_clips, modem_exchange, training_pool  # noqa: E402

SELECT_CLIPS = 24


def _single_thread():
    torch.set_num_threads(1)


def _exchange(job):
    wire, seed = job
    return modem_exchange(wire, "V8", seed, "mpp12")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("eval", "select", "train"), required=True)
    ap.add_argument("--model", default="models/v8-hf3k-mpp12-ft.pt")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--seed0", type=int)
    ap.add_argument("--clips", type=int)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.split == "eval":
        paths, seed0 = eval_split(clips=args.clips or 64), 2026
        if args.seed0 not in (None, 2026) or args.draws != 1:
            ap.error("the eval split uses one draw with the published fade seeds (seed0 2026)")
    elif args.split == "select":
        paths, seed0 = training_pool()[:SELECT_CLIPS], args.seed0 or 7300
    else:
        paths, seed0 = training_pool()[SELECT_CLIPS:], args.seed0 or 1_000_000
    if args.clips:
        paths = paths[: args.clips]
    device = torch.device("cuda")
    adapter = AutoencoderAdapter("v8", args.model, "V8", device)
    clips = load_clips(paths)
    n, gops = clips.shape[0], 2
    started = time.time()

    wires = []
    with torch.no_grad():
        for start in range(0, n, 32):
            video = clips[start : start + 32].float().div(255).to(device)
            wires.append(torch.stack([adapter.model.encoder(video[:, :, g * 6 : (g + 1) * 6]) for g in range(gops)], 1).cpu())
    wires = torch.cat(wires).numpy()
    print(f"encoded {n} clips in {time.time() - started:.1f}s", flush=True)

    jobs = [(wires[i, g], seed0 + ((d * n) + i) * gops + g) for d in range(args.draws) for i in range(n) for g in range(gops)]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(args.workers, initializer=_single_thread) as pool:
        results = pool.map(_exchange, jobs, chunksize=16)
    rx = torch.from_numpy(np.stack([r for r, _, _ in results])).view(args.draws, n, gops, -1)
    cf = torch.from_numpy(np.stack([c for _, c, _ in results])).view(args.draws, n, gops, -1)
    failures = int(sum(not ok for _, _, ok in results))
    print(f"modem {len(jobs)} GOPs in {time.time() - started:.1f}s, {failures} lost", flush=True)

    dtype = torch.float32 if args.split != "train" else torch.float16
    decoded = torch.empty(n, args.draws, 3, 12, 108, 192, dtype=dtype)
    shape = (6, 108, 192)
    with torch.no_grad():
        for d in range(args.draws):
            for start in range(0, n, 32):
                sl = slice(start, start + 32)
                parts = [
                    adapter.model.decoder(rx[d, sl, g].to(device), cf[d, sl, g].to(device), output_shape=shape).clamp(0, 1)
                    for g in range(gops)
                ]
                decoded[sl, d] = torch.cat(parts, 2).cpu().to(dtype)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "names": [p.name for p in paths],
        "model": args.model,
        "seed0": seed0,
        "source": clips,
        "decoded": decoded,
        "gop_confidence": cf.mean(-1).permute(1, 0, 2).contiguous(),
        "failures": failures,
    }, args.out)
    print(f"saved {args.out} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
