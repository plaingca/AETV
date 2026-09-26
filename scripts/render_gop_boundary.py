#!/usr/bin/env python3
"""Side-by-side frames and an MP4 around the V8 GOP boundary (source | current | fixed).

Uses a cache from ``scripts/cache_v8_draws.py`` (V8 modem + ``mpp12``) and a
refiner checkpoint. Frames 4-7 straddle the boundary between GOP 0 (frames
0-5) and GOP 1 (frames 6-11).

    python scripts/render_gop_boundary.py --cache runs/gop-boundary/eval64.pt \
        --refiner runs/gop-refiner-bidir/best.pt --clips 3 17 40 --out media/gop-boundary
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

from aetv.gop_boundary import frame_psnr, load_refiner  # noqa: E402
from eval_gop_boundary import GOP, gop_confidence_frames  # noqa: E402

SCALE = 2
LABELS = ("source", "current (mpp12)", "fixed (mpp12)")


def to_bgr(frame: torch.Tensor) -> np.ndarray:
    rgb = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    rgb = cv2.resize(rgb, (rgb.shape[1] * SCALE, rgb.shape[0] * SCALE), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def label(img: np.ndarray, text: str, y: int = 22, color=(255, 255, 255)) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (4, y - h - 4), (12 + w, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def panel_row(views: list[torch.Tensor], t: int, psnrs: list[list[float] | None]) -> np.ndarray:
    tiles = []
    for view, name, ps in zip(views, LABELS, psnrs):
        img = to_bgr(view[:, t])
        label(img, name if ps is None else f"{name} {ps[t]:.1f} dB")
        gop, pos = divmod(t, GOP)
        tag = f"frame {t}  GOP {gop} pos {pos}"
        color = (80, 80, 255) if pos in (0, GOP - 1) else (255, 255, 255)
        label(img, tag, img.shape[0] - 10, color)
        tiles.append(img)
    sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
    return np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refiner", required=True)
    ap.add_argument("--clips", nargs="+", type=int, required=True)
    ap.add_argument("--prefix", default="eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=3.0, help="MP4 frame rate (V8 is 6 fps; slower shows the jump)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    refiner = load_refiner(args.refiner, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_frames = []
    for i in args.clips:
        source = cache["source"][i].float().div(255).to(device)
        current = cache["decoded"][i, 0].float().to(device)
        conf = gop_confidence_frames(cache["gop_confidence"][i, 0][None].float()).to(device)
        with torch.no_grad():
            fixed = refiner(current[None], conf)[0]
        views = [source, current, fixed]
        psnrs = [None, frame_psnr(source, current), frame_psnr(source, fixed)]
        rows = [panel_row(views, t, psnrs) for t in range(GOP - 2, GOP + 2)]
        sep = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
        grid = np.concatenate([x for r in rows for x in (r, sep)][:-1], 0)
        cv2.imwrite(str(out / f"{args.prefix}{i:02d}-boundary-f04-f07.png"), grid)
        for t in range(source.shape[1]):
            video_frames.append(panel_row(views, t, psnrs))
        video_frames.extend([np.zeros_like(video_frames[-1])] * 2)
    h, w = video_frames[0].shape[:2]
    mp4 = out / "gop-boundary-mpp12-side-by-side.mp4"
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
