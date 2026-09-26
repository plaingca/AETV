#!/usr/bin/env python3
"""Side-by-side frames and an MP4 of V8 receivers on ``mpp12`` (source | current | refiners...).

Uses a cache from ``scripts/cache_v8_draws.py`` (V8 modem + ``mpp12``) and one
or more ``--refiner label=path[@healthy_gain]``. The default grid rows are
frames 4-7, which straddle the boundary between GOP 0 (frames 0-5) and GOP 1
(frames 6-11). The MP4 plays every frame of each clip ``--loops`` times.

    python scripts/render_gop_boundary.py --cache runs/gop-boundary/eval64.pt \
        --refiner "old refiner=models/v8-gop-refiner-hq.pt" --clips 29 14 --out media/gop-boundary
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.gop_boundary import frame_psnr, refiner_from_spec  # noqa: E402
from eval_gop_boundary import GOP, gop_confidence_frames  # noqa: E402

SCALE = 2


def to_bgr(frame: torch.Tensor) -> np.ndarray:
    rgb = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    rgb = cv2.resize(rgb, (rgb.shape[1] * SCALE, rgb.shape[0] * SCALE), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def label(img: np.ndarray, text: str, y: int = 22, color=(255, 255, 255)) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (4, y - h - 4), (12 + w, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def panel_row(views: list[torch.Tensor], labels: list[str], t: int, psnrs: list[list[float] | None]) -> np.ndarray:
    tiles = []
    for view, name, ps in zip(views, labels, psnrs):
        img = to_bgr(view[:, t])
        label(img, name if ps is None else f"{name} {ps[t]:.1f} dB")
        gop, pos = divmod(t, GOP)
        tag = f"frame {t}  GOP {gop} pos {pos}"
        color = (80, 80, 255) if pos in (0, GOP - 1) else (255, 255, 255)
        label(img, tag, img.shape[0] - 10, color)
        tiles.append(img)
    sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
    return np.concatenate([x for tile in tiles for x in (tile, sep)][:-1], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refiner", action="append", required=True, help="label=path or label=path@healthy_gain")
    ap.add_argument("--clips", nargs="+", type=int, required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[4, 5, 6, 7], help="grid rows")
    ap.add_argument("--prefix", default="eval")
    ap.add_argument("--suffix", default="boundary-f04-f07")
    ap.add_argument("--name", default="gop-boundary-mpp12-side-by-side.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=6.0, help="MP4 frame rate (V8 plays at 6 fps)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    refiners = dict(refiner_from_spec(spec, device) for spec in args.refiner)
    labels = ["source", "current", *refiners]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_frames = []
    for i in args.clips:
        source = cache["source"][i].float().div(255).to(device)
        current = cache["decoded"][i, 0].float().to(device)
        conf = gop_confidence_frames(cache["gop_confidence"][i, 0][None].float()).to(device)
        with torch.no_grad():
            views = [source, current, *(m(current[None], conf)[0] for m in refiners.values())]
        psnrs = [None, *(frame_psnr(source, v) for v in views[1:])]
        rows = [panel_row(views, labels, t, psnrs) for t in args.frames]
        sep = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
        grid = np.concatenate([x for r in rows for x in (r, sep)][:-1], 0)
        cv2.imwrite(str(out / f"{args.prefix}{i:02d}-{args.suffix}.png"), grid)
        for _ in range(args.loops):
            for t in range(source.shape[1]):
                video_frames.append(panel_row(views, labels, t, psnrs))
        video_frames.extend([np.zeros_like(video_frames[-1])] * 3)
    h, w = video_frames[0].shape[:2]
    mp4 = out / args.name
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)],
        stdin=subprocess.PIPE,
    )
    for frame in video_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {len(args.clips)} grids and {mp4}")


if __name__ == "__main__":
    main()
