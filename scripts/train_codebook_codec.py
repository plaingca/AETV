#!/usr/bin/env python3
"""Train the codebook JSCC codec from scratch against a two-path fade.

Checkpoints are kept when real V8 + mpp12 PSNR on held-out training clips
improves. The 64-clip eval set is not used here.
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

from aetv import AETV_MODES, demodulate_gop_stream, modulate_gop_stream
from aetv.codebook_codec import CodebookCodec, apply_carrier_fade
from aetv.hfchannel import emulate
from aetv.narrowband_jscc import psnr
from scripts.finetune_ac2k_psnr import load_clips


def _modem(wire: torch.Tensor, seed: int, profile: str) -> tuple[torch.Tensor, torch.Tensor]:
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


def sample_clip(clips: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    index = torch.randint(0, clips.shape[0], (batch,))
    start = random.choice((0, 4))
    video = clips[index, :, start : start + 8].float().div(255.0)
    if random.random() < 0.5:
        video = video.flip(-1)
    return video.to(device)


def save_checkpoint(model: CodebookCodec, path: Path, metrics: dict, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "codes": model.codes,
            "dim": model.dim,
            "width": model.width,
            "metrics": metrics,
            "step": step,
            "budget": 2816,
            "bandwidth_khz": 2.2,
            "modem_mode": "V8",
            "channel": "mpp12",
        },
        path,
    )


@torch.no_grad()
def score_mpp(model: CodebookCodec, clips: torch.Tensor, count: int, seed0: int) -> float:
    was = model.training
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for index in range(count):
        clip = clips[index].float().div(255.0).unsqueeze(0).to(device)
        model.reset()
        wires, gops = [], []
        for start in (0, 4, 8):
            gop = clip[:, :, start : start + 4]
            wires.append(model.encode_gop(gop, retain_state=True))
            gops.append(gop)
        model.reset()
        for gop_index, (gop, wire) in enumerate(zip(gops, wires)):
            received, confidence = _modem(wire, seed0 + index * 3 + gop_index, "mpp12")
            recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
            scores.append(psnr(gop, recon))
    if was:
        model.train()
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--codes", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--holdout-clips", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="runs/codebook-codec")
    parser.add_argument("--checkpoint", default="models/codebook-codec-2.2khz-best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_clips, _val, _names = load_clips(Path(args.cache), args.val_clips, args.seed)
    holdout = train_clips[-args.holdout_clips :]
    train_clips = train_clips[: -args.holdout_clips]
    model = CodebookCodec(codes=args.codes, width=args.width).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters {params / 1e6:.2f}M | train clips {train_clips.shape[0]}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    best = -1.0
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        progress = step / args.steps
        # Learn the picture first, then spend more of the step on deep fades.
        sigma = 0.10 + 0.65 * progress
        fade_prob = 0.25 + 0.55 * progress
        warm = min(1.0, step / 200)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.lr * warm * (0.1 + 0.9 * cosine)
        for group in optimizer.param_groups:
            group["lr"] = lr
        video = sample_clip(train_clips, args.batch, device)
        model.reset()
        loss = video.new_zeros(())
        for part in (video[:, :, :4], video[:, :, 4:]):
            wire = model.encode_gop(part, retain_state=True)
            if random.random() < fade_prob:
                received, confidence = apply_carrier_fade(wire, sigma)
            else:
                received, confidence = wire, torch.ones_like(wire)
            recon, _outage = model.decode_gop(received, confidence=confidence, retain_state=True)
            loss = loss + F.mse_loss(recon, part)
        loss = loss / 2 + 0.05 * model.aux_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            commit = model.codebook.last_commit
            entropy = model.codebook.last_entropy
            print(
                f"step {step} loss {float(loss.detach()):.4f} lr {lr:.6f} sigma {sigma:.2f} "
                f"commit {float(commit.detach()) if commit is not None else 0:.4f} "
                f"usage {float(entropy.detach()) if entropy is not None else 0:.3f} "
                f"token_h {float(model.codebook.last_token_h.detach()) if model.codebook.last_token_h is not None else 0:.3f} "
                f"codes {model.codebook.last_unique}",
                flush=True,
            )
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = score_mpp(model, holdout, args.holdout_clips, seed0=7300)
            record = {"step": step, "loss": float(loss.detach()), "holdout_mpp12": score, "sigma": sigma}
            print(f"  holdout mpp12 {score:.3f} dB", flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if score > best:
                best = score
                save_checkpoint(model, Path(args.checkpoint), record, step)
                print(f"  saved {args.checkpoint}", flush=True)
    print(f"best holdout mpp12 {best:.3f} dB in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
