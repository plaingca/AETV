#!/usr/bin/env python3
"""Score the 2.2 kHz AC2K PSNR fine-tune against the fidelity checkpoint.

Uses the seed-2026 OpenVid cache split. Each 12-frame clip is three stateful
4-frame GOPs. Wire noise is unit-RMS AWGN. ``--modem-clips`` also loops the
first GOP through the V8 OFDM waveform.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv import AETV_MODES, demodulate_gop_stream, modulate_gop_stream
from aetv.ac2k_psnr import load_ac2k, load_psnr_checkpoint
from aetv.hfchannel import awgn
from aetv.narrowband_jscc import global_ssim, impair_wire, psnr


def val_tensor(cache: Path, val_clips: int, seed: int) -> tuple[torch.Tensor, list[str]]:
    files = sorted(cache.glob("*.pt"))
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    selected = [files[index] for index in order[:val_clips]]
    clips = [torch.load(path, map_location="cpu", weights_only=False) for path in selected]
    tensor = torch.stack(clips)
    if tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
    return tensor, [path.name for path in selected]


def _lpips_pair(metric, reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    ref = reference[0].permute(1, 0, 2, 3).mul(2).sub(1)
    hat = reconstruction[0].permute(1, 0, 2, 3).mul(2).sub(1)
    return float(metric(hat, ref).mean().item())


@torch.no_grad()
def score_model(model, val: torch.Tensor, device: torch.device, lpips_metric, snrs: tuple[float, ...]) -> dict:
    clean, ssim, lpips_clean = [], [], []
    noisy = {snr: [] for snr in snrs}
    for index in range(val.shape[0]):
        clip = val[index].float().div(255.0).unsqueeze(0).to(device)
        model.reset()
        wires, gops = [], []
        for start in (0, 4, 8):
            gop = clip[:, :, start : start + 4]
            wires.append(model.encode_gop(gop, retain_state=True))
            gops.append(gop)
        model.reset()
        for gop, wire in zip(gops, wires):
            recon, _ = model.decode_gop(wire, retain_state=True)
            clean.append(psnr(gop, recon))
            ssim.append(global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3)))
            if lpips_metric is not None:
                lpips_clean.append(_lpips_pair(lpips_metric, gop, recon))
        for snr in snrs:
            model.reset()
            for gop, wire in zip(gops, wires):
                received, confidence = impair_wire(wire, snr)
                recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
                noisy[snr].append(psnr(gop, recon))
        if (index + 1) % 16 == 0:
            print(f"  scored {index + 1}/{val.shape[0]} clips, clean {float(np.mean(clean)):.2f} dB", flush=True)
    report = {
        "clean_psnr": float(np.mean(clean)),
        "clean_ssim": float(np.mean(ssim)),
        "gops": len(clean),
        "clips": int(val.shape[0]),
    }
    if lpips_clean:
        report["clean_lpips"] = float(np.mean(lpips_clean))
    for snr, values in noisy.items():
        report[f"awgn_{snr:.0f}db_psnr"] = float(np.mean(values))
    return report


@torch.no_grad()
def score_modem(model, val: torch.Tensor, device: torch.device, clips: int) -> dict:
    mode = AETV_MODES["V8"]
    rows = {name: [] for name in ("clean", "awgn_18", "awgn_12", "awgn_6")}
    latent_mse = []
    for index in range(min(clips, val.shape[0])):
        gop = val[index, :, :4].float().div(255.0).unsqueeze(0).to(device)
        model.reset()
        wire = model.encode_gop(gop, retain_state=False).squeeze(0).detach().float().cpu().numpy()
        audio = modulate_gop_stream([wire], mode_name="V8", callsign="EVAL")
        for name, snr in (("clean", None), ("awgn_18", 18.0), ("awgn_12", 12.0), ("awgn_6", 6.0)):
            impaired = audio if snr is None else awgn(audio, snr_db=snr, seed=1000 + index, fs=mode.geometry.fs)
            demod = demodulate_gop_stream(impaired, band=mode.band, drift_track="off")
            if not demod.gops_latents:
                rows[name].append(float("nan"))
                continue
            received = torch.as_tensor(demod.gops_latents[0], dtype=torch.float32, device=device).unsqueeze(0)
            weights = torch.as_tensor(demod.gops_weights[0], dtype=torch.float32, device=device).unsqueeze(0)
            if name == "clean":
                latent_mse.append(float((received.squeeze(0).cpu() - torch.from_numpy(wire)).square().mean()))
            model.reset()
            recon, _ = model.decode_gop(received, confidence=weights.clamp(0, 1), retain_state=False)
            rows[name].append(psnr(gop, recon))
        print(f"  modem clip {index + 1}: clean {rows['clean'][-1]:.2f} dB", flush=True)
    return {
        "modem_clean_psnr": float(np.nanmean(rows["clean"])),
        "modem_awgn18_psnr": float(np.nanmean(rows["awgn_18"])),
        "modem_awgn12_psnr": float(np.nanmean(rows["awgn_12"])),
        "modem_awgn6_psnr": float(np.nanmean(rows["awgn_6"])),
        "modem_clean_latent_mse": float(np.mean(latent_mse)) if latent_mse else float("nan"),
        "modem_gops": int(min(clips, val.shape[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--baseline", default="models/ac2k-v2-fidelity-best-inference.pt")
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modem-clips", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/ac2k-psnr/eval64.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    val, names = val_tensor(Path(args.cache), args.val_clips, args.seed)
    lpips_metric = None
    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}", flush=True)

    print("scoring fine-tune", flush=True)
    model, payload = load_psnr_checkpoint(args.checkpoint, device)
    improved = score_model(model, val, device, lpips_metric, (18.0, 15.0, 12.0, 6.0))
    if args.modem_clips:
        improved.update(score_modem(model, val, device, args.modem_clips))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("scoring fidelity baseline", flush=True)
    baseline = load_ac2k(args.baseline, device).to(device).eval()
    baseline_report = score_model(baseline, val, device, lpips_metric, (18.0, 15.0, 12.0, 6.0))

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": payload.get("architecture"),
        "step": payload.get("step"),
        "bandwidth_khz": payload.get("bandwidth_khz", 2.2),
        "budget": payload.get("budget", 2816),
        "modem_mode": payload.get("modem_mode", "V8"),
        "seed": args.seed,
        "val_clips": names,
        "improved": improved,
        "baseline_checkpoint": str(Path(args.baseline).resolve()),
        "baseline": baseline_report,
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
