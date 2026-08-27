#!/usr/bin/env python3
"""Paired 32-clip evaluation of an LTX-V8 channel checkpoint versus V8."""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetv.ltx_channel import (  # noqa: E402
    LTX_REPO,
    LTXV8ChannelAdapter,
    finish_ltx_video,
    prepare_ltx_video,
)
from compare_checkpoints import CELLS, METRICS, load_clips, metric_values, paired  # noqa: E402
from train import AETV_MODES, AETVAutoencoder, simulate_transmission  # noqa: E402


def empty_results():
    return {label: {metric: [] for metric in METRICS} for label, _, _ in CELLS}


@torch.no_grad()
def evaluate_v8(checkpoint: Path, clips, device):
    mode = AETV_MODES["V8"]
    model = AETVAutoencoder(mode=mode, width=128, latent_channels=3, causal=False).to(device)
    model.load_pretrained_weights(str(checkpoint), device=device)
    model.eval()
    output = empty_results()
    shape = (mode.gop_frames, mode.height, mode.width)
    for index, clip in enumerate(clips):
        target = clip.to(device)
        latent = model.encoder(target)
        latent_numpy = latent[0].float().cpu().numpy()
        for label, snr, fading in CELLS:
            if snr is None:
                reconstruction = model.decoder(latent, torch.ones_like(latent), shape)
            else:
                received, confidence, _ = simulate_transmission(
                    latent_numpy, mode_name="V8", snr_db=snr, fading_preset=fading
                )
                reconstruction = model.decoder(
                    torch.from_numpy(received).unsqueeze(0).to(device),
                    torch.from_numpy(confidence).unsqueeze(0).to(device),
                    shape,
                )
            for metric, value in metric_values(reconstruction, target, device).items():
                output[label][metric].append(value)
        print(f"  V8 {index + 1:02d}/{len(clips)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return output


@torch.no_grad()
def evaluate_ltx(checkpoint: Path, clips, device):
    from diffusers import AutoencoderKLLTXVideo

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "aetv-ltx-v8-channel-v1":
        raise ValueError(f"not an LTX-V8 channel checkpoint: {checkpoint}")
    adapter = LTXV8ChannelAdapter().to(device).eval()
    adapter.load_state_dict(payload["adapter"])
    vae = AutoencoderKLLTXVideo.from_pretrained(
        payload.get("ltx_repo", LTX_REPO), subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    if payload.get("decoder_conv_in"):
        vae.decoder.conv_in.float()
        vae.decoder.conv_in.load_state_dict(payload["decoder_conv_in"])

    output = empty_results()
    for index, clip in enumerate(clips):
        target = clip.to(device)
        latent = vae.encode(prepare_ltx_video(target).to(torch.bfloat16)).latent_dist.mode()
        symbols = adapter.encode(latent.float())
        symbol_numpy = symbols[0].float().cpu().numpy()
        for label, snr, fading in CELLS:
            if snr is None:
                received = symbols
                confidence = torch.ones_like(symbols)
            else:
                received_numpy, confidence_numpy, _ = simulate_transmission(
                    symbol_numpy, mode_name="V8", snr_db=snr, fading_preset=fading
                )
                received = torch.from_numpy(received_numpy).unsqueeze(0).to(device)
                confidence = torch.from_numpy(confidence_numpy).unsqueeze(0).to(device)
            restored = adapter.decode(received, confidence)
            # The fine-tuned input convolution keeps fp32 master weights while
            # the frozen remainder of the VAE is bf16. Autocast is therefore
            # part of the checkpoint's inference contract.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                reconstruction = finish_ltx_video(vae.decode(restored).sample)
            for metric, value in metric_values(reconstruction, target, device).items():
                output[label][metric].append(value)
        print(f"  LTX {index + 1:02d}/{len(clips)}", flush=True)
    return output


def print_comparison(results, reference: str, candidate: str) -> None:
    base = results[reference]
    test = results[candidate]
    for metric, better in METRICS.items():
        print(f"\n=== {metric.upper()} ({better} is better) ===")
        print(f"{'cell':>12} {'V8 mean':>10} {'LTX mean':>10} {'delta +- SE':>22}")
        for label, _, _ in CELLS:
            reference_mean = st.mean(base[label][metric])
            candidate_mean = st.mean(test[label][metric])
            delta, error = paired(test[label][metric], base[label][metric])
            mark = "*" if error > 0 and abs(delta) > 2 * error else ""
            print(
                f"{label:>12} {reference_mean:10.4f} {candidate_mean:10.4f} "
                f"{delta:+9.4f} +- {error:<7.4f}{mark}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--v8", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/openvid-cache-5fps-eval-v68"))
    parser.add_argument("--clips", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    clips = load_clips(args.cache_dir, AETV_MODES["V8"], args.clips)
    if len(clips) != args.clips:
        raise SystemExit(f"wanted {args.clips} held-out clips, found {len(clips)}")
    print(f"Paired evaluation: {len(clips)} clips x {len(CELLS)} cells", flush=True)
    started = time.time()
    results = {
        "v8-released": evaluate_v8(args.v8, clips, device),
        "ltx-v8": evaluate_ltx(args.candidate, clips, device),
    }
    print_comparison(results, "v8-released", "ltx-v8")
    payload = {
        "reference": "v8-released",
        "candidate": "ltx-v8",
        "clips": len(clips),
        "cells": [
            {"label": label, "snr_db": snr, "fading": fading}
            for label, snr, fading in CELLS
        ],
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
