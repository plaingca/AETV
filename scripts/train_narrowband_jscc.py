#!/usr/bin/env python3
"""Train the narrowband JSCC autoencoder against an existing RF latent budget.

The budget is selected with ``--bandwidth`` and is never larger than the
modem GOP already defined for that channel (2.2, 8, or 16 kHz). The objective
is reconstruction PSNR: pixel MSE, a light SSIM term, and AWGN on the unit-RMS
wire so the 15 dB operating point stays close to the clean loopback.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.narrowband_jscc import (
    NarrowbandJSCC,
    global_ssim,
    impair_wire,
    psnr,
    resolve_bandwidth,
)


def pyramid_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(prediction, target)
    pred, ref = prediction, target
    weight = 1.0
    for _ in range(2):
        pred = F.avg_pool3d(pred, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ref = F.avg_pool3d(ref, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        weight *= 0.5
        loss = loss + weight * F.mse_loss(pred, ref)
    return loss


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1, c2 = 0.01**2, 0.03**2
    batch, channels, frames, height, width = prediction.shape
    pred = prediction.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    ref = target.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    mu_p = F.avg_pool2d(pred, 3, stride=1, padding=1)
    mu_r = F.avg_pool2d(ref, 3, stride=1, padding=1)
    var_p = (F.avg_pool2d(pred.square(), 3, stride=1, padding=1) - mu_p.square()).clamp_min(0)
    var_r = (F.avg_pool2d(ref.square(), 3, stride=1, padding=1) - mu_r.square()).clamp_min(0)
    cov = F.avg_pool2d(pred * ref, 3, stride=1, padding=1) - mu_p * mu_r
    score = ((2 * mu_p * mu_r + c1) * (2 * cov + c2)) / (
        (mu_p.square() + mu_r.square() + c1) * (var_p + var_r + c2)
    )
    return 1.0 - score.clamp(0, 1).mean()


def load_split(cache: Path, val_clips: int, seed: int, overfit: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    files = sorted(cache.glob("*.pt"))
    if len(files) < val_clips + 8:
        raise FileNotFoundError(f"need at least {val_clips + 8} clips in {cache}, found {len(files)}")
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    val_ids = order[:val_clips]
    train_ids = order[val_clips:]
    if overfit:
        train_ids = train_ids[:overfit]
        val_ids = train_ids[: min(4, len(train_ids))]

    def stack(ids: list[int]) -> torch.Tensor:
        clips = [torch.load(files[index], map_location="cpu", weights_only=False) for index in ids]
        tensor = torch.stack(clips)
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
        return tensor

    train = stack(train_ids)
    val = stack(val_ids)
    names = [files[index].name for index in val_ids]
    return train, val, names


def sample_gop(clips: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    count = clips.shape[0]
    index = torch.randint(0, count, (batch,))
    start = int(torch.randint(0, clips.shape[2] - 5, ()).item())
    gop = clips[index, :, start : start + 6].float().div(255.0)
    if random.random() < 0.5:
        gop = gop.flip(-1)
    return gop.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: NarrowbandJSCC, val: torch.Tensor, device: torch.device, clips: int) -> dict[str, float]:
    model.eval()
    clean, noisy, ssim_clean = [], [], []
    limit = min(clips, val.shape[0])
    for index in range(limit):
        clip = val[index].float().div(255.0)
        scores_c, scores_n, scores_s = [], [], []
        for start in (0, 6):
            gop = clip[:, start : start + 6].unsqueeze(0).to(device)
            recon = model(gop)
            received, confidence = impair_wire(model.encode(gop), 15.0)
            recon_n = model.decode(received, confidence)
            scores_c.append(psnr(gop, recon))
            scores_n.append(psnr(gop, recon_n))
            flat_ref = gop[0].permute(1, 0, 2, 3)
            flat_hat = recon[0].permute(1, 0, 2, 3)
            scores_s.append(global_ssim(flat_ref, flat_hat))
        clean.append(sum(scores_c) / len(scores_c))
        noisy.append(sum(scores_n) / len(scores_n))
        ssim_clean.append(sum(scores_s) / len(scores_s))
    model.train()
    return {
        "clean_psnr": float(sum(clean) / len(clean)),
        "noisy_15db_psnr": float(sum(noisy) / len(noisy)),
        "clean_ssim": float(sum(ssim_clean) / len(ssim_clean)),
        "clips": float(limit),
    }


def save_inference(model: NarrowbandJSCC, path: Path, metrics: dict, step: int, val_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": model.architecture,
        "bandwidth_khz": model.bandwidth_khz,
        "budget": model.budget,
        "modem_mode": model.modem_mode,
        "model_config": model.config,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metrics": metrics,
        "step": step,
        "val_clips": val_names,
    }
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    khz, budget, mode = resolve_bandwidth(args.bandwidth)
    device = torch.device(args.device)
    cache = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_clips, val_clips, val_names = load_split(cache, args.val_clips, args.seed, args.overfit)
    print(
        f"bandwidth {khz} kHz | budget {budget} | modem {mode} | "
        f"train clips {train_clips.shape[0]} | val clips {val_clips.shape[0]}",
        flush=True,
    )
    model = NarrowbandJSCC(khz, width=args.width).to(device)
    if args.init:
        payload = torch.load(args.init, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
        print(f"warm start {args.init}: missing {len(missing)} unexpected {len(unexpected)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    best = -1e9
    log_path = out / "train_log.jsonl"
    t0 = time.time()
    for step in range(1, args.steps + 1):
        gop = sample_gop(train_clips, args.batch, device)
        scale = 0.85 + 0.3 * torch.rand(gop.shape[0], 1, 1, 1, 1, device=device)
        shift = (torch.rand(gop.shape[0], 1, 1, 1, 1, device=device) - 0.5) * 0.08
        gop = (gop * scale + shift).clamp(0, 1)
        snr = None
        if random.random() < args.channel_prob:
            snr = torch.empty(gop.shape[0], device=device).uniform_(args.snr_min, args.snr_max)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            recon = model(gop, snr_db=snr)
            loss = pyramid_mse(recon, gop) + args.ssim_weight * ssim_loss(recon, gop)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_interval == 0 or step == 1:
            mse = F.mse_loss(recon.detach().float(), gop.float()).item()
            train_psnr = 10.0 * math_log10(mse)
            print(
                f"step {step:05d}/{args.steps} loss {loss.item():.4f} "
                f"train_psnr {train_psnr:.2f} lr {scheduler.get_last_lr()[0]:.2e} "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(model, val_clips, device, args.eval_clips)
            score = 0.5 * metrics["clean_psnr"] + 0.5 * metrics["noisy_15db_psnr"]
            record = {"step": step, "score": score, **metrics}
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"eval step {step}: clean {metrics['clean_psnr']:.2f} dB | "
                f"15 dB {metrics['noisy_15db_psnr']:.2f} dB | SSIM {metrics['clean_ssim']:.4f}",
                flush=True,
            )
            if score > best:
                best = score
                save_inference(model, Path(args.checkpoint), metrics, step, val_names)
                save_inference(model, out / "best.pt", metrics, step, val_names)
                print(f"saved {args.checkpoint}", flush=True)


def math_log10(mse: float) -> float:
    import math

    if mse <= 1e-10:
        return 100.0
    return math.log10(1.0 / mse) * 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandwidth", type=float, required=True, choices=[2.2, 8.0, 16.0])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument("--channel-prob", type=float, default=0.55)
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=28.0)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--eval-clips", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--overfit", type=int, default=0, help="train on only this many clips")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
