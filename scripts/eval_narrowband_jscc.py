#!/usr/bin/env python3
"""Score a narrowband JSCC checkpoint on held-out clips.

Reports the project's reconstruction metrics: PSNR, global SSIM, and LPIPS
when the training extra is installed. Wire AWGN uses the same 15 dB convention
as the AC6 runs. ``--modem`` also passes the latent vector through the
production OFDM waveform for that bandwidth.
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
from aetv.hfchannel import awgn
from aetv.narrowband_jscc import (
    NarrowbandJSCC,
    global_ssim,
    impair_wire,
    psnr,
)


def load_model(path: Path, device: torch.device) -> tuple[NarrowbandJSCC, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("model_config") or {}
    model = NarrowbandJSCC(
        bandwidth_khz=payload.get("bandwidth_khz", config.get("bandwidth_khz", 2.2)),
        width=config.get("width", 64),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def val_tensor(cache: Path, names: list[str] | None, val_clips: int, seed: int) -> tuple[torch.Tensor, list[str]]:
    files = sorted(cache.glob("*.pt"))
    if names:
        selected = [cache / name for name in names]
    else:
        order = list(range(len(files)))
        random.Random(seed).shuffle(order)
        selected = [files[index] for index in order[:val_clips]]
    clips = [torch.load(path, map_location="cpu", weights_only=False) for path in selected]
    tensor = torch.stack(clips)
    if tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
    return tensor, [path.name for path in selected]


def _lpips_pair(metric, reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    # LPIPS expects NCHW in [-1, 1]. Average the six frames.
    frames = reference.shape[2]
    ref = reference[0].permute(1, 0, 2, 3).mul(2).sub(1)
    hat = reconstruction[0].permute(1, 0, 2, 3).mul(2).sub(1)
    return float(metric(hat, ref).mean().item()) if frames else float("nan")


@torch.no_grad()
def score_codec(model: NarrowbandJSCC, val: torch.Tensor, device: torch.device, lpips_metric) -> dict:
    clean, ssim, lpips_clean = [], [], []
    noisy = {snr: [] for snr in (18.0, 15.0, 12.0, 6.0)}
    for index in range(val.shape[0]):
        clip = val[index].float().div(255.0)
        for start in (0, 6):
            gop = clip[:, start : start + 6].unsqueeze(0).to(device)
            recon = model(gop)
            clean.append(psnr(gop, recon))
            ssim.append(global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3)))
            if lpips_metric is not None:
                lpips_clean.append(_lpips_pair(lpips_metric, gop, recon))
            wire = model.encode(gop)
            for snr in noisy:
                received, confidence = impair_wire(wire, snr)
                noisy[snr].append(psnr(gop, model.decode(received, confidence)))
    report = {
        "clean_psnr": float(np.mean(clean)),
        "clean_ssim": float(np.mean(ssim)),
        "gops": len(clean),
    }
    if lpips_clean:
        report["clean_lpips"] = float(np.mean(lpips_clean))
    for snr, values in noisy.items():
        report[f"awgn_{snr:.0f}db_psnr"] = float(np.mean(values))
    return report


@torch.no_grad()
def score_modem(model: NarrowbandJSCC, val: torch.Tensor, device: torch.device, clips: int) -> dict:
    mode = AETV_MODES[model.modem_mode]
    rows = {name: [] for name in ("clean", "awgn_18", "awgn_12", "awgn_6")}
    latent_nmse = []
    for index in range(min(clips, val.shape[0])):
        gop = val[index, :, :6].float().div(255.0).unsqueeze(0).to(device)
        wire = model.encode(gop).squeeze(0).detach().float().cpu().numpy()
        audio = modulate_gop_stream([wire], mode_name=model.modem_mode, callsign="EVAL")
        for name, snr in (("clean", None), ("awgn_18", 18.0), ("awgn_12", 12.0), ("awgn_6", 6.0)):
            impaired = audio if snr is None else awgn(audio, snr_db=snr, seed=1000 + index, fs=mode.geometry.fs)
            demod = demodulate_gop_stream(impaired, band=mode.band, drift_track="off")
            if not demod.gops_latents:
                rows[name].append(float("nan"))
                continue
            received = torch.as_tensor(demod.gops_latents[0], dtype=torch.float32, device=device).unsqueeze(0)
            weights = torch.as_tensor(demod.gops_weights[0], dtype=torch.float32, device=device).unsqueeze(0)
            if name == "clean":
                latent_nmse.append(float((received.cpu() - torch.from_numpy(wire)).square().mean()))
            recon = model.decode(received, weights.clamp(0, 1))
            rows[name].append(psnr(gop, recon))
    return {
        "modem_clean_psnr": float(np.nanmean(rows["clean"])),
        "modem_awgn18_psnr": float(np.nanmean(rows["awgn_18"])),
        "modem_awgn12_psnr": float(np.nanmean(rows["awgn_12"])),
        "modem_awgn6_psnr": float(np.nanmean(rows["awgn_6"])),
        "modem_clean_latent_mse": float(np.mean(latent_nmse)) if latent_nmse else float("nan"),
        "modem_gops": int(min(clips, val.shape[0])),
    }


@torch.no_grad()
def score_ac6(path: Path, val: torch.Tensor, device: torch.device) -> dict:
    from aetv.ac6 import AC6Codec

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("model_config") or {}
    legacy = payload.get("architecture") == "ac6-gated-residual-v2-pixelshuffle"
    model = AC6Codec(
        spatial_pooling=False if legacy else None,
        width=config.get("width", config.get("model_width", 48)),
        features=config.get("features", 24 if legacy else 20),
        motion_width=config.get("motion_width", 48 if legacy else 40),
        refine_blocks=config.get("refine_blocks", 2),
        anchor_gate_max=config.get("anchor_gate_max", 0.35),
        anchor_style_max=config.get("anchor_style_max", 0.39),
        deep_tail=config.get("deep_tail", True),
        diagonal_gdn=config.get("diagonal_gdn", True),
        iframe_mem=config.get("iframe_mem", True),
        model_width=config.get("model_width"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    clean, noisy, ssim = [], [], []
    for index in range(val.shape[0]):
        clip = val[index].float().div(255.0).unsqueeze(0).to(device)
        gop0, gop1 = clip[:, :, :6], clip[:, :, 6:12]
        model.reset()
        wire0 = model.encode_gop(gop0, retain_state=True)
        wire1 = model.encode_gop(gop1, retain_state=True)
        model.reset()
        recon0, _ = model.decode_gop(wire0, retain_state=True)
        recon1, _ = model.decode_gop(wire1, retain_state=True)
        model.reset()
        received0, confidence0 = impair_wire(wire0, 15.0)
        received1, confidence1 = impair_wire(wire1, 15.0)
        noisy0, _ = model.decode_gop(received0, confidence=confidence0, retain_state=True)
        noisy1, _ = model.decode_gop(received1, confidence=confidence1, retain_state=True)
        for gop, recon, recon_n in (
            (gop0, recon0, noisy0),
            (gop1, recon1, noisy1),
        ):
            clean.append(psnr(gop, recon))
            noisy.append(psnr(gop, recon_n))
            ssim.append(global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3)))
    return {
        "checkpoint": str(path),
        "clean_psnr": float(np.mean(clean)),
        "clean_ssim": float(np.mean(ssim)),
        "awgn_15db_psnr": float(np.mean(noisy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modem-clips", type=int, default=0)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, payload = load_model(Path(args.checkpoint), device)
    names = payload.get("val_clips")
    val, used = val_tensor(Path(args.cache), names, args.val_clips, args.seed)
    lpips_metric = None
    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}", flush=True)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "bandwidth_khz": model.bandwidth_khz,
        "budget": model.budget,
        "modem_mode": model.modem_mode,
        "step": payload.get("step"),
        "val_clips": used,
        **score_codec(model, val, device, lpips_metric),
    }
    if args.modem_clips:
        report.update(score_modem(model, val, device, args.modem_clips))
    if args.baseline:
        report["baselines"] = [score_ac6(Path(path), val, device) for path in args.baseline]
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
