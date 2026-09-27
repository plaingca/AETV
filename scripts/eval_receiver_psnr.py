#!/usr/bin/env python3
"""Paired end-to-end PSNR of receiver arms on the shared protocol.

Same clips and fade seeds as ``scripts/eval_shared.py`` (64 eval clips, fade
seed ``2026 + clip * G + gop``). Each clip is encoded once, and every receiver
arm demodulates the identical impaired waveform, so per-clip differences are
exact pairs. ``pilot`` is the v0.1.23 receiver; ``both`` is LMMSE plus window
placement.

    python scripts/eval_receiver_psnr.py --model V8=models/v8-hf3k-mpp12-ft.pt \\
        --model V9=models/v9-wide4k.pt --out runs/receiver-psnr/eval64.json
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

from aetv import modem  # noqa: E402
from aetv.hfchannel import CHANNEL_PROFILES, RESEARCH_PROFILES, ChannelProfile  # noqa: E402
from aetv.shared_eval import (  # noqa: E402
    DEFAULT_CACHE,
    AutoencoderAdapter,
    FaceMasks,
    clip_psnr,
    eval_split,
    load_clips,
    masked_psnr,
    modem_exchange,
    paired,
    summarize,
)

ARMS = {"pilot": ("pilot", False), "lmmse": ("lmmse", False), "place": ("pilot", True), "both": ("lmmse", True)}
EXTRA = {"mpp3": ChannelProfile("mpp3", "MPP 3", 3.0, "mpp")}


def profile(name: str):
    return CHANNEL_PROFILES.get(name) or RESEARCH_PROFILES.get(name) or EXTRA[name]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="MODE=checkpoint")
    parser.add_argument("--arms", default="pilot,both")
    parser.add_argument("--profiles", default="mpp12")
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--seed0", type=int, default=2026)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--face-model", default="data/teachers/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--frames-dir", help="save source and reconstructions for --frame-clips")
    parser.add_argument("--frame-clips", nargs="+", type=int, default=[0, 24, 48])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = args.arms.split(",")
    profiles = args.profiles.split(",")
    paths = eval_split(args.cache, clips=args.clips)
    clips = load_clips(paths)
    faces = FaceMasks(args.face_model)
    lpips_metric = None
    if not args.no_lpips:
        import lpips

        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"protocol": {"clips": [p.name for p in paths], "seed0": args.seed0, "arms": arms, "profiles": profiles},
              "models": {}}

    for spec in args.model:
        mode_name, path = spec.split("=", 1)
        started = time.time()
        adapter = AutoencoderAdapter(mode_name, path, mode_name, device)
        per_clip = {(p, a, k): [] for p in profiles for a in arms for k in ("psnr", "lpips", "face", "failures")}
        for index in range(clips.shape[0]):
            clip = clips[index].float().div(255).unsqueeze(0).to(device)
            gops = adapter.gops(clip)
            wires = adapter.encode(gops)
            mask = faces.mask(paths[index].name, clips[index])[adapter.frame_index()]
            if (adapter.height, adapter.width) != tuple(mask.shape[-2:]):
                mask = torch.nn.functional.interpolate(
                    mask[None].float(), size=(adapter.height, adapter.width), mode="nearest")[0].bool()
            source = torch.cat(gops, 2)[0]
            kept = {"source": source.cpu().half()}
            for prof in profiles:
                for arm in arms:
                    modem.CHANNEL_ESTIMATOR, modem.WINDOW_PLACEMENT = ARMS[arm]
                    rx, cf, failures = [], [], 0
                    for g, wire in enumerate(wires):
                        seed = args.seed0 + index * len(wires) + g
                        r, c, ok = modem_exchange(wire.squeeze(0).float().cpu().numpy(), mode_name, seed, profile(prof))
                        failures += int(not ok)
                        rx.append(torch.from_numpy(r).to(device).unsqueeze(0))
                        cf.append(torch.from_numpy(c).to(device).unsqueeze(0))
                    recons = adapter.decode(rx, cf)
                    joined = torch.cat(recons, 2)[0]
                    per_clip[prof, arm, "psnr"].append(clip_psnr(gops, recons))
                    per_clip[prof, arm, "face"].append(masked_psnr(source, joined, mask.to(device)))
                    per_clip[prof, arm, "failures"].append(failures)
                    if lpips_metric is not None:
                        per_clip[prof, arm, "lpips"].append(float(np.mean([
                            float(lpips_metric(r[0].permute(1, 0, 2, 3) * 2 - 1, g[0].permute(1, 0, 2, 3) * 2 - 1).mean())
                            for g, r in zip(gops, recons)
                        ])))
                    if frames_dir is not None and index in args.frame_clips:
                        kept[(prof, arm)] = joined.cpu().half()
            if frames_dir is not None and index in args.frame_clips:
                torch.save(kept, frames_dir / f"{mode_name}-eval{index:02d}.pt")
            if (index + 1) % 16 == 0:
                line = " ".join(f"{p}/{a} {np.mean(per_clip[p, a, 'psnr']):.3f}" for p in profiles for a in arms)
                print(f"  {mode_name} {index + 1}/{clips.shape[0]} {line}", flush=True)
        summary = {}
        for prof in profiles:
            for arm in arms:
                entry = {k: summarize(per_clip[prof, arm, k]) for k in ("psnr", "lpips", "face")}
                entry["failures"] = int(sum(per_clip[prof, arm, "failures"]))
                if arm != arms[0]:
                    for k in ("psnr", "lpips", "face"):
                        entry[k]["paired_vs_" + arms[0]] = paired(per_clip[prof, arm, k], per_clip[prof, arms[0], k])
                summary[f"{prof}/{arm}"] = entry
                d = entry["psnr"].get("paired_vs_" + arms[0])
                print(f"{mode_name} {prof} {arm}: PSNR {entry['psnr']['mean']:.3f}"
                      + (f"  paired {d['mean']:+.3f} ± {d['se']:.3f}" if d else ""), flush=True)
        result["models"][mode_name] = {
            "checkpoint": path,
            "summary": summary,
            "per_clip": {f"{p}/{a}/{k}": v for (p, a, k), v in per_clip.items()},
            "seconds": time.time() - started,
        }
        out_path.write_text(json.dumps(result, indent=1))
        del adapter
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
