#!/usr/bin/env python3
"""Score the fresh codec and the frozen downsample reference on V8 mpp12.

The reference config is read from the training run. It is not retuned on the
64-clip val split. Val clips are the seed-2026 holdout of the OpenVid cache,
resized to the codec contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.deepstream_hf import (
    GOP_FRAMES,
    HEIGHT,
    WIDTH,
    DownsampleReference,
    load_deepstream_hf,
    psnr,
    v8_exchange,
)
from aetv.narrowband_jscc import global_ssim
from scripts.train_deepstream_hf import load_split, resize_gop


def _score_model(model, clips: torch.Tensor, profile: str, seed0: int) -> dict:
    device = next(model.parameters()).device
    scores, ssim = [], []
    for index in range(clips.shape[0]):
        clip = resize_gop(clips[index : index + 1].to(device))
        for gop_index, start in enumerate(range(0, clip.shape[2], GOP_FRAMES)):
            gop = clip[:, :, start : start + GOP_FRAMES]
            wire = model.encode_gop(gop, retain_state=False)
            received_np, confidence_np = v8_exchange(
                wire.squeeze(0).detach().float().cpu().numpy(), seed0 + index * 6 + gop_index, profile
            )
            received = torch.as_tensor(received_np, device=device, dtype=wire.dtype).unsqueeze(0)
            confidence = torch.as_tensor(confidence_np, device=device, dtype=wire.dtype).unsqueeze(0)
            recon, _confidence = model.decode_gop(received, confidence=confidence, retain_state=False)
            scores.append(psnr(gop, recon))
            ssim.append(global_ssim(gop[0].permute(1, 0, 2, 3), recon[0].permute(1, 0, 2, 3)))
        if (index + 1) % 8 == 0:
            print(f"  model {profile} {index + 1}/{clips.shape[0]}  {float(np.mean(scores)):.2f} dB", flush=True)
    return {"profile": profile, "psnr": float(np.mean(scores)), "ssim": float(np.mean(ssim)), "gops": len(scores)}


def _score_reference(reference: DownsampleReference, clips: torch.Tensor, profile: str, seed0: int) -> dict:
    scores, ssim = [], []
    for index in range(clips.shape[0]):
        clip = resize_gop(clips[index : index + 1]).squeeze(0).cpu().numpy()
        for gop_index, start in enumerate(range(0, clip.shape[1], GOP_FRAMES)):
            gop = np.transpose(clip[:, start : start + GOP_FRAMES], (1, 0, 2, 3))
            wire = reference.transmit(gop)
            received, confidence = v8_exchange(wire, seed0 + index * 6 + gop_index, profile)
            recon = np.clip(reference.receive(received, confidence), 0.0, 1.0)
            scores.append(psnr(torch.from_numpy(gop), torch.from_numpy(recon.copy())))
            ssim.append(
                global_ssim(
                    torch.from_numpy(gop).permute(1, 0, 2, 3),
                    torch.from_numpy(recon).permute(1, 0, 2, 3),
                )
            )
        if (index + 1) % 8 == 0:
            print(f"  reference {profile} {index + 1}/{clips.shape[0]}  {float(np.mean(scores)):.2f} dB", flush=True)
    return {"profile": profile, "psnr": float(np.mean(scores)), "ssim": float(np.mean(ssim)), "gops": len(scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/deepstream-hf-2.2khz-best.pt")
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--scales", default="runs/deepstream-hf/dct_scales.npy")
    parser.add_argument("--baseline", default="runs/deepstream-hf/baseline.json")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/deepstream-hf/eval64.json")
    parser.add_argument("--skip-clean", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    _train, val, names = load_split(Path(args.cache), args.val_clips, args.seed)
    del _train
    model, payload = load_deepstream_hf(args.checkpoint, device)
    baseline_payload = json.loads(Path(args.baseline).read_text())
    config = payload.get("baseline", {}).get("config", baseline_payload["config"])
    scales = np.load(args.scales) if Path(args.scales).exists() else None
    reference = DownsampleReference(config, scales=scales)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": payload.get("architecture"),
        "step": payload.get("step"),
        "budget": 2816,
        "bandwidth_khz": 2.2,
        "modem_mode": "V8",
        "contract": {"width": WIDTH, "height": HEIGHT, "frames": GOP_FRAMES, "fps": float(GOP_FRAMES)},
        "seed": args.seed,
        "val_clips": names,
        "baseline_config": config,
        "baseline_holdout_mpp12": baseline_payload.get("holdout_mpp12"),
    }
    print("scoring model mpp12", flush=True)
    report["model_mpp12"] = _score_model(model, val, "mpp12", args.seed)
    print("scoring reference mpp12", flush=True)
    report["baseline_mpp12"] = _score_reference(reference, val, "mpp12", args.seed)
    report["mpp12_delta"] = report["model_mpp12"]["psnr"] - report["baseline_mpp12"]["psnr"]
    if not args.skip_clean:
        print("scoring model clean", flush=True)
        report["model_clean"] = _score_model(model, val, "clean", args.seed)
        print("scoring reference clean", flush=True)
        report["baseline_clean"] = _score_reference(reference, val, "clean", args.seed)
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
