#!/usr/bin/env python3
"""Score the carrier fill on V8 multipath and on the unit-RMS AWGN table.

Multipath passes each GOP through the V8 modem and ``hfchannel.emulate``.
The same received coordinates are decoded by frozen AC2K and by the fill, so
the comparison does not depend on a second fade draw. The AWGN table is the
seed-2026 64-clip protocol used for the AC2K PSNR fine-tune.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv import AETV_MODES, demodulate_gop_stream, modulate_gop_stream
from aetv.ac2k_psnr import load_psnr_checkpoint
from aetv.hfchannel import emulate
from aetv.multipath_common import load_multipath_fill
from aetv.narrowband_jscc import global_ssim, psnr
from scripts.eval_ac2k_psnr import score_model, val_tensor


def _receive(wire: torch.Tensor, seed: int, profile: str) -> tuple[torch.Tensor, torch.Tensor]:
    mode = AETV_MODES["V8"]
    audio = modulate_gop_stream(
        [wire.squeeze(0).detach().float().cpu().numpy()], mode_name="V8", callsign="EVAL"
    )
    impaired = audio if profile == "clean" else emulate(audio, profile, seed=seed, fs=mode.geometry.fs)
    demod = demodulate_gop_stream(impaired, band=mode.band, drift_track="off")
    if not demod.gops_latents:
        return torch.zeros_like(wire), torch.zeros_like(wire)
    received = torch.as_tensor(demod.gops_latents[0], dtype=wire.dtype, device=wire.device).unsqueeze(0)
    confidence = (
        torch.as_tensor(demod.gops_weights[0], dtype=wire.dtype, device=wire.device).unsqueeze(0).clamp(0, 1)
    )
    return received, confidence


@torch.no_grad()
def score_profile(fill, baseline, val: torch.Tensor, profile: str, seed0: int) -> dict:
    device = next(fill.parameters()).device
    fill_scores, base_scores, fill_ssim = [], [], []
    for index in range(val.shape[0]):
        clip = val[index].float().div(255.0).unsqueeze(0).to(device)
        fill.reset()
        wires, gops = [], []
        for start in (0, 4, 8):
            gop = clip[:, :, start : start + 4]
            wires.append(fill.encode_gop(gop, retain_state=True))
            gops.append(gop)
        fill.reset()
        baseline.reset()
        for gop_index, (gop, wire) in enumerate(zip(gops, wires)):
            received, confidence = _receive(wire, seed0 + index * 3 + gop_index, profile)
            recon, _ = fill.decode_gop(received, confidence=confidence, retain_state=True)
            base, _ = baseline.decode_gop(received, confidence=confidence, retain_state=True)
            fill_scores.append(psnr(gop, recon))
            base_scores.append(psnr(gop, base))
            fill_ssim.append(global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3)))
        if (index + 1) % 8 == 0:
            print(
                f"  {profile} {index + 1}/{val.shape[0]}  fill {float(np.mean(fill_scores)):.2f}  "
                f"ac2k {float(np.mean(base_scores)):.2f}",
                flush=True,
            )
    return {
        "profile": profile,
        "seed0": seed0,
        "clips": int(val.shape[0]),
        "gops": len(fill_scores),
        "fill_psnr": float(np.mean(fill_scores)),
        "ac2k_psnr": float(np.mean(base_scores)),
        "delta_psnr": float(np.mean(fill_scores) - np.mean(base_scores)),
        "fill_ssim": float(np.mean(fill_ssim)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/multipath-fill-2.2khz-best.pt")
    parser.add_argument("--ac2k", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profiles", nargs="+", default=["mpp12"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/multipath-fill/eval64.json")
    parser.add_argument("--skip-awgn", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    val, names = val_tensor(Path(args.cache), args.val_clips, args.seed)
    fill, payload = load_multipath_fill(args.checkpoint, device, ac2k_path=args.ac2k)
    baseline, _base_payload = load_psnr_checkpoint(args.ac2k, device)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": payload.get("architecture"),
        "step": payload.get("step"),
        "budget": 2816,
        "bandwidth_khz": 2.2,
        "modem_mode": "V8",
        "seed": args.seed,
        "val_clips": names,
        "ac2k_checkpoint": str(Path(args.ac2k).resolve()),
        "multipath": {},
    }
    for profile in args.profiles:
        print(f"scoring {profile}", flush=True)
        report["multipath"][profile] = score_profile(fill, baseline, val, profile, args.seed)

    if not args.skip_awgn:
        lpips_metric = None
        try:
            import lpips

            lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:
            print(f"LPIPS unavailable: {exc}", flush=True)
        print("scoring fill AWGN table", flush=True)
        report["fill_awgn"] = score_model(fill, val, device, lpips_metric, (18.0, 15.0, 12.0, 6.0))
        print("scoring AC2K AWGN table", flush=True)
        report["ac2k_awgn"] = score_model(baseline, val, device, lpips_metric, (18.0, 15.0, 12.0, 6.0))

    text = json.dumps(report, indent=2)
    print(text, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
