#!/usr/bin/env python3
"""Train the fresh DeepStream HF codec from random weights.

The fair baseline is chosen on training-holdout clips only. The 64-clip val
split is never used here. Checkpoints are written only when real V8 + mpp12
PSNR on that holdout improves.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.deepstream_hf import (
    GOP_FRAMES,
    HEIGHT,
    WIDTH,
    REFERENCE_CANDIDATES,
    DeepStreamHF,
    DownsampleReference,
    apply_hf_fade,
    dct_scales,
    load_deepstream_hf,
    psnr,
    reference_fits,
    v8_exchange,
)


def load_split(cache: Path, val_clips: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    files = sorted(cache.glob("*.pt"))
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    val_ids, train_ids = order[:val_clips], order[val_clips:]

    def stack(ids: list[int]) -> torch.Tensor:
        clips = [torch.load(files[index], map_location="cpu", weights_only=False) for index in ids]
        tensor = torch.stack(clips)
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
        return tensor

    return stack(train_ids), stack(val_ids), [files[index].name for index in val_ids]


def resize_gop(video: torch.Tensor) -> torch.Tensor:
    """(B, 3, 2, H, W) uint or float 0..1 -> float (B, 3, 2, 18, 32)."""
    if video.dtype == torch.uint8:
        video = video.float().div(255.0)
    batch, channels, frames, _height, _width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, _height, _width)
    small = F.interpolate(flat, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
    return small.reshape(batch, frames, channels, HEIGHT, WIDTH).permute(0, 2, 1, 3, 4).contiguous()


def sample_gop(clips: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    index = torch.randint(0, clips.shape[0], (batch,))
    start = random.randrange(0, clips.shape[2] - GOP_FRAMES + 1, GOP_FRAMES)
    video = clips[index, :, start : start + GOP_FRAMES]
    video = resize_gop(video)
    if random.random() < 0.5:
        video = video.flip(-1)
    return video.to(device)


def save_checkpoint(model: DeepStreamHF, path: Path, metrics: dict, step: int, baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metrics": metrics,
            "step": step,
            "budget": 2816,
            "bandwidth_khz": 2.2,
            "modem_mode": "V8",
            "channel": "mpp12",
            "contract": {"width": WIDTH, "height": HEIGHT, "frames": GOP_FRAMES, "fps": float(GOP_FRAMES)},
            "baseline": baseline,
        },
        path,
    )


def _modem_gop(wire: torch.Tensor, seed: int, profile: str) -> tuple[torch.Tensor, torch.Tensor]:
    received, confidence = v8_exchange(wire.detach().float().cpu().numpy().reshape(-1), seed, profile)
    return (
        torch.as_tensor(received, dtype=wire.dtype, device=wire.device).view(1, -1),
        torch.as_tensor(confidence, dtype=wire.dtype, device=wire.device).view(1, -1),
    )


@torch.no_grad()
def score_split(model: DeepStreamHF, clips: torch.Tensor, count: int, seed0: int, profile: str) -> float:
    was = model.training
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for index in range(count):
        clip = resize_gop(clips[index : index + 1].to(device))
        for gop_index, start in enumerate(range(0, clip.shape[2], GOP_FRAMES)):
            gop = clip[:, :, start : start + GOP_FRAMES]
            wire = model.encode_gop(gop, retain_state=False)
            received, confidence = _modem_gop(wire, seed0 + index * 6 + gop_index, profile)
            recon, _confidence = model.decode_gop(received, confidence=confidence, retain_state=False)
            scores.append(psnr(gop, recon))
    if was:
        model.train()
    return float(np.mean(scores))


@torch.no_grad()
def score_reference(reference: DownsampleReference, clips: torch.Tensor, count: int, seed0: int, profile: str) -> float:
    scores = []
    for index in range(count):
        clip = resize_gop(clips[index : index + 1]).squeeze(0).cpu().numpy()
        for gop_index, start in enumerate(range(0, clip.shape[1], GOP_FRAMES)):
            gop = np.transpose(clip[:, start : start + GOP_FRAMES], (1, 0, 2, 3))
            wire = reference.transmit(gop)
            received, confidence = v8_exchange(wire, seed0 + index * 6 + gop_index, profile)
            recon = np.clip(reference.receive(received, confidence), 0.0, 1.0)
            scores.append(psnr(torch.from_numpy(gop), torch.from_numpy(recon)))
    return float(np.mean(scores))


def choose_baseline(train_clips: torch.Tensor, holdout: torch.Tensor, count: int) -> dict:
    videos = []
    for index in range(min(32, train_clips.shape[0])):
        clip = resize_gop(train_clips[index : index + 1]).squeeze(0).cpu().numpy()
        videos.append(np.transpose(clip[:, :GOP_FRAMES], (1, 0, 2, 3)))
    scales = dct_scales(np.stack(videos, axis=0))
    best = None
    for config in REFERENCE_CANDIDATES:
        if not reference_fits(config):
            continue
        reference = DownsampleReference(config, scales=scales)
        score = score_reference(reference, holdout, count, seed0=8100, profile="mpp12")
        record = {"config": config, "holdout_mpp12": score}
        print(f"  baseline {config}  mpp12 {score:.3f} dB", flush=True)
        if best is None or score > best["holdout_mpp12"]:
            best = record
    best["scales"] = scales.tolist()
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--holdout-clips", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("--out", default="runs/deepstream-hf")
    parser.add_argument("--checkpoint", default="models/deepstream-hf-2.2khz-best.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_clips, _val, _names = load_split(Path(args.cache), args.val_clips, args.seed)
    holdout = train_clips[-args.holdout_clips :]
    train_clips = train_clips[: -args.holdout_clips]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("choosing downsample/upsample baseline on training holdout", flush=True)
    baseline = choose_baseline(train_clips, holdout, args.holdout_clips)
    (out / "baseline.json").write_text(json.dumps({k: v for k, v in baseline.items() if k != "scales"}) + "\n")
    np.save(out / "dct_scales.npy", np.asarray(baseline["scales"], dtype=np.float32))
    print(f"frozen baseline {baseline['config']} at {baseline['holdout_mpp12']:.3f} dB", flush=True)

    if args.resume:
        model, resumed = load_deepstream_hf(args.resume, device)
        best = float(resumed.get("metrics", {}).get("holdout_mpp12", -1.0))
        print(f"resumed {args.resume} at holdout {best:.3f} dB", flush=True)
    else:
        model = DeepStreamHF().to(device)
        best = -1.0
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters {params / 1e6:.2f}M | train clips {train_clips.shape[0]}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.jsonl"
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        progress = step / args.steps
        warm = min(1.0, step / 200)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.lr * warm * (0.1 + 0.9 * cosine)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Learn the picture under a light fade, then match the mpp surrogate.
        sigma = args.sigma * (0.15 + 0.85 * min(1.0, step / (0.35 * args.steps)))
        video = sample_gop(train_clips, args.batch, device)
        wire = model.encode_gop(video, retain_state=False)
        received, confidence = apply_hf_fade(wire, sigma)
        if step % 4 == 0:
            received_m, confidence_m = _modem_gop(wire[:1], seed=50_000 + step, profile="mpp12")
            received = torch.cat([received[:1] + (received_m - received[:1]).detach(), received[1:]], dim=0)
            confidence = torch.cat([confidence_m, confidence[1:]], dim=0)
        recon, _confidence = model.decode_gop(received, confidence=confidence, retain_state=False)
        loss = F.mse_loss(recon, video)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            was = model.training
            model.eval()
            with torch.no_grad():
                clean_wire = model.encode_gop(video[:1], retain_state=False)
                clean = psnr(video[:1], model.decode_gop(clean_wire, retain_state=False)[0])
            if was:
                model.train()
            print(
                f"step {step} loss {float(loss.detach()):.4f} lr {lr:.6f} sigma {sigma:.2f} clean {clean:.2f}",
                flush=True,
            )
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = score_split(model, holdout, args.holdout_clips, seed0=7300, profile="mpp12")
            record = {"step": step, "loss": float(loss.detach()), "holdout_mpp12": score, "sigma": sigma}
            print(f"  holdout mpp12 {score:.3f} dB", flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if score > best:
                best = score
                save_checkpoint(model, Path(args.checkpoint), record, step, {k: v for k, v in baseline.items() if k != "scales"})
                print(f"  saved {args.checkpoint}", flush=True)
    print(f"best holdout mpp12 {best:.3f} dB in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
