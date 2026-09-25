#!/usr/bin/env python3
"""Precompute VVC teacher signals at the 2.2 kHz rate for V8 distillation.

For every clip (default: the training pool, plus any extra clips listed), the
12-frame 192x108 6 fps clip is VVC-coded with a 1 s intra period (two
independent 6-frame GOPs, the V8 GOP). The QP is chosen per clip as the lowest
in ``--qps`` whose raw stream fits ``--kbps``. Stored per clip:

- ``decode``: the VVC decode at that rate, uint8 (3, 12, 108, 192).
- ``importance``: a bit-allocation map, uint8 (12, 108, 192) in units of 1/32.
  It is the local energy of (decode at the chosen QP - decode at ``--low-qp``),
  box-blurred over 8x8, normalized to mean 1 per GOP and clipped to 8. It marks
  where the extra bits changed the picture, a proxy for where VVC spent them.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

H, W, T, FPS, SECONDS = 108, 192, 12, 6, 2.0


def vvc_roundtrip(frames: np.ndarray, qp: int) -> tuple[float, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "s.vvc"
        enc = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libvvenc", "-preset", "slow", "-qp", str(qp),
               "-qpa", "0", "-period", "1", "-vvenc-params", "threads=1", "-pix_fmt", "yuv420p10le", "-f", "vvc",
               str(out)]
        subprocess.run(enc, input=frames.tobytes(), check=True, capture_output=True)
        kbps = out.stat().st_size * 8 / SECONDS / 1000
        dec = ["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(dec, check=True, capture_output=True).stdout
    return kbps, np.frombuffer(raw, np.uint8).reshape(T, H, W, 3)


def box_blur(x: np.ndarray, k: int = 8) -> np.ndarray:
    pad = np.pad(x, ((0, 0), (k // 2, k - k // 2 - 1), (k // 2, k - k // 2 - 1)), mode="edge")
    c = pad.cumsum(1).cumsum(2)
    c = np.pad(c, ((0, 0), (1, 0), (1, 0)))
    return (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]) / (k * k)


def process(job):
    path, out_dir, qps, kbps_target, low_qp = job
    import torch

    torch.set_num_threads(1)
    target = Path(out_dir) / (Path(path).stem + ".npz")
    if target.exists():
        return str(target), None
    clip = torch.load(path, map_location="cpu", weights_only=False)
    frames = clip.permute(1, 2, 3, 0).contiguous().numpy()
    chosen = None
    for qp in sorted(qps):
        kbps, dec = vvc_roundtrip(frames, qp)
        if kbps <= kbps_target:
            chosen = (qp, kbps, dec)
            break
    if chosen is None:
        qp = max(qps)
        chosen = (qp, *vvc_roundtrip(frames, qp))
    qp, kbps, dec = chosen
    _, low = vvc_roundtrip(frames, low_qp)
    energy = ((dec.astype(np.float32) - low.astype(np.float32)) / 255.0) ** 2
    energy = box_blur(energy.sum(-1))
    importance = np.zeros_like(energy)
    for start in (0, 6):
        e = energy[start : start + 6]
        importance[start : start + 6] = e / max(float(e.mean()), 1e-8)
    importance = np.clip(np.rint(np.clip(importance, 0, 8) * 32), 0, 255).astype(np.uint8)
    mse = float(((dec.astype(np.float64) - frames) / 255.0).__pow__(2).mean())
    np.savez(target, decode=dec.transpose(3, 0, 1, 2), importance=importance, qp=qp, kbps=kbps)
    return str(target), (qp, kbps, mse)


def main() -> None:
    from aetv.shared_eval import eval_split, training_pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/vvc_teacher")
    ap.add_argument("--kbps", type=float, default=4.5)
    ap.add_argument("--qps", nargs="+", type=int, default=[38, 41, 44, 47, 50, 53])
    ap.add_argument("--low-qp", type=int, default=62)
    ap.add_argument("--eval-clips", nargs="*", type=int, default=[0, 24, 48],
                    help="eval-split indices to encode for side-by-side frames only (never trained on)")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    evals = eval_split()
    jobs = [(str(p), str(out), a.qps, a.kbps, a.low_qp) for p in training_pool()]
    jobs += [(str(evals[i]), str(out / "eval"), a.qps, a.kbps, a.low_qp) for i in a.eval_clips]
    stats = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for n, (_, info) in enumerate(ex.map(process, jobs, chunksize=2)):
            if info is not None:
                stats.append(info)
            if (n + 1) % 200 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
    if stats:
        qps, rates, mses = zip(*stats)
        print(f"clips {len(stats)}: mean {np.mean(rates):.2f} kb/s, QP median {int(np.median(qps))}, "
              f"mean VVC PSNR {np.mean([10 * np.log10(1 / m) for m in mses]):.2f} dB", flush=True)


if __name__ == "__main__":
    main()
