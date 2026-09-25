#!/usr/bin/env python3
"""Score the codebook codec on V8 mpp12 and on the latent-AWGN table.

Multipath uses the same clip order, GOP cuts, and fade seeds as the AC2K
mpp12 baseline. Each model transmits its own wire. Latent AWGN is unit-RMS
noise at 18/15/12/6 dB plus a clean pass, with SSIM and AlexNet LPIPS.
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
from aetv.codebook_codec import load_codebook_codec
from aetv.hfchannel import emulate
from aetv.narrowband_jscc import global_ssim, impair_wire, psnr
from scripts.eval_ac2k_psnr import _lpips_pair, val_tensor


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
def score_profile(model, val: torch.Tensor, profile: str, seed0: int) -> dict:
    scores, ssim = [], []
    for index in range(val.shape[0]):
        clip = val[index].float().div(255.0).unsqueeze(0).to(next(model.parameters()).device)
        model.reset()
        wires, gops = [], []
        for start in (0, 4, 8):
            gop = clip[:, :, start : start + 4]
            wires.append(model.encode_gop(gop, retain_state=True))
            gops.append(gop)
        model.reset()
        for gop_index, (gop, wire) in enumerate(zip(gops, wires)):
            received, confidence = _receive(wire, seed0 + index * 3 + gop_index, profile)
            recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
            scores.append(psnr(gop, recon))
            ssim.append(global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3)))
        if (index + 1) % 8 == 0:
            print(f"  {profile} {index + 1}/{val.shape[0]}  {float(np.mean(scores)):.2f} dB", flush=True)
    return {
        "profile": profile,
        "seed0": seed0,
        "clips": int(val.shape[0]),
        "gops": len(scores),
        "psnr": float(np.mean(scores)),
        "ssim": float(np.mean(ssim)),
    }


@torch.no_grad()
def score_awgn(model, val: torch.Tensor, lpips_metric, snrs: tuple[float, ...]) -> dict:
    """Clean plus latent AWGN. PSNR, SSIM, and LPIPS at every SNR."""
    device = next(model.parameters()).device
    buckets = {"clean": {"psnr": [], "ssim": [], "lpips": []}}
    for snr in snrs:
        buckets[f"{snr:.0f}"] = {"psnr": [], "ssim": [], "lpips": []}
    for index in range(val.shape[0]):
        clip = val[index].float().div(255.0).unsqueeze(0).to(device)
        model.reset()
        wires, gops = [], []
        for start in (0, 4, 8):
            gop = clip[:, :, start : start + 4]
            wires.append(model.encode_gop(gop, retain_state=True))
            gops.append(gop)
        conditions = [("clean", None)] + [(f"{snr:.0f}", snr) for snr in snrs]
        for name, snr in conditions:
            model.reset()
            for gop, wire in zip(gops, wires):
                if snr is None:
                    received, confidence = wire, torch.ones_like(wire)
                else:
                    received, confidence = impair_wire(wire, snr)
                recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
                buckets[name]["psnr"].append(psnr(gop, recon))
                buckets[name]["ssim"].append(
                    global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3))
                )
                if lpips_metric is not None:
                    buckets[name]["lpips"].append(_lpips_pair(lpips_metric, gop, recon))
        if (index + 1) % 16 == 0:
            print(f"  awgn {index + 1}/{val.shape[0]} clean {float(np.mean(buckets['clean']['psnr'])):.2f} dB", flush=True)
    report = {"clips": int(val.shape[0]), "gops": len(buckets["clean"]["psnr"])}
    for name, values in buckets.items():
        key = "clean" if name == "clean" else f"awgn_{name}db"
        report[f"{key}_psnr"] = float(np.mean(values["psnr"]))
        report[f"{key}_ssim"] = float(np.mean(values["ssim"]))
        if values["lpips"]:
            report[f"{key}_lpips"] = float(np.mean(values["lpips"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/codebook-codec-2.2khz-best.pt")
    parser.add_argument("--ac2k", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/codebook-codec/eval64.json")
    parser.add_argument("--skip-awgn", action="store_true")
    parser.add_argument("--skip-ac2k-mpp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    val, names = val_tensor(Path(args.cache), args.val_clips, args.seed)
    model, payload = load_codebook_codec(args.checkpoint, device)
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
        "published_ac2k_mpp12": 19.62,
        "published_ac2k_clean": 24.80,
        "published_ac2k_awgn15": 24.37,
    }
    print("scoring codebook mpp12", flush=True)
    report["codebook_mpp12"] = score_profile(model, val, "mpp12", args.seed)
    if not args.skip_ac2k_mpp:
        baseline, _ = load_psnr_checkpoint(args.ac2k, device)
        print("scoring AC2K mpp12", flush=True)
        report["ac2k_mpp12"] = score_profile(baseline, val, "mpp12", args.seed)
        report["mpp12_delta"] = report["codebook_mpp12"]["psnr"] - report["ac2k_mpp12"]["psnr"]
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_awgn:
        lpips_metric = None
        try:
            import lpips

            lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:
            print(f"LPIPS unavailable: {exc}", flush=True)
        print("scoring codebook AWGN table", flush=True)
        report["codebook_awgn"] = score_awgn(model, val, lpips_metric, (18.0, 15.0, 12.0, 6.0))
        ac2k, _ = load_psnr_checkpoint(args.ac2k, device)
        print("scoring AC2K AWGN table", flush=True)
        report["ac2k_awgn"] = score_awgn(ac2k, val, lpips_metric, (18.0, 15.0, 12.0, 6.0))

    text = json.dumps(report, indent=2)
    print(text, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
