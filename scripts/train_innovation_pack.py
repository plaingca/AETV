#!/usr/bin/env python3
"""Train the conditional innovation packed into AC2K's spare 2.2 kHz coordinates."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.innovation_pack import INDEX_PATH, InnovationPack, compute_weak_index
from aetv.narrowband_jscc import global_ssim, impair_wire, psnr
from scripts.finetune_ac2k_psnr import load_clips, sample_pair


def ensure_index(train_clips: torch.Tensor, device: torch.device) -> None:
    if INDEX_PATH.exists():
        return
    print(f"measuring spare coordinates -> {INDEX_PATH}", flush=True)
    index = compute_weak_index(train_clips[:40], device)
    torch.save(index, INDEX_PATH)
    print(f"spare index {tuple(index.shape)}", flush=True)


@torch.no_grad()
def evaluate(model: InnovationPack, val: torch.Tensor, device: torch.device, clips: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    clean, noisy, ssim = [], [], []
    for index in range(min(clips, val.shape[0])):
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
        model.reset()
        for gop, wire in zip(gops, wires):
            received, confidence = impair_wire(wire, 15.0)
            recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
            noisy.append(psnr(gop, recon))
    if was_training:
        model.train()
    return {
        "clean_psnr": float(sum(clean) / len(clean)),
        "noisy_15db_psnr": float(sum(noisy) / len(noisy)),
        "clean_ssim": float(sum(ssim) / len(ssim)),
    }


def save_checkpoint(model: InnovationPack, path: Path, metrics: dict, step: int, val_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "ac2k_checkpoint": model.ac2k_path,
            "index": model.index.detach().cpu(),
            "pack": {key: value.detach().cpu() for key, value in model.pack.state_dict().items()},
            "metrics": metrics,
            "step": step,
            "val_clips": val_names,
            "budget": 2816,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise-start", type=int, default=400)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--eval-clips", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ac2k", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--out", default="runs/innovation-pack")
    parser.add_argument("--checkpoint", default="models/innovation-pack-2.2khz-best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_clips, val_clips, val_names = load_clips(Path(args.cache), args.val_clips, args.seed)
    ensure_index(train_clips, device)
    model = InnovationPack(args.ac2k).to(device)
    params = sum(item.numel() for item in model.pack.parameters())
    frozen = sum(item.numel() for item in model.base.parameters())
    print(
        f"innovation {params/1e6:.2f}M trainable | frozen AC2K {frozen/1e6:.2f}M | "
        f"spare {model.index.numel()} of 2816",
        flush=True,
    )
    baseline = evaluate(model.base, val_clips, device, args.eval_clips)
    print(
        f"frozen baseline clean {baseline['clean_psnr']:.2f} dB | "
        f"15 dB {baseline['noisy_15db_psnr']:.2f} dB",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.pack.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    best = -1e9
    log_path = Path(args.out)
    log_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        gop0, gop1 = sample_pair(train_clips, args.batch, device)
        use_noise = step > args.noise_start and random.random() < 0.5
        snr_db = None if not use_noise else float(torch.empty(()).uniform_(6.0, 20.0))
        optimizer.zero_grad(set_to_none=True)
        model.reset()
        wire0 = model.encode_gop(gop0, retain_state=True)
        wire1 = model.encode_gop(gop1, retain_state=True)
        model.reset()
        if snr_db is None:
            recon0, _ = model.decode_gop(wire0, retain_state=True)
            recon1, _ = model.decode_gop(wire1, retain_state=True)
        else:
            received0, confidence0 = impair_wire(wire0, snr_db)
            received1, confidence1 = impair_wire(wire1, snr_db)
            recon0, _ = model.decode_gop(received0, confidence=confidence0, retain_state=True)
            recon1, _ = model.decode_gop(received1, confidence=confidence1, retain_state=True)
        recon = torch.cat([recon0, recon1], dim=2)
        target = torch.cat([gop0, gop1], dim=2)
        loss = F.mse_loss(recon.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.pack.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            mse = float(loss.detach())
            batch_psnr = 100.0 if mse <= 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
            print(
                f"step {step:05d}/{args.steps} loss {mse:.4f} batch_psnr {batch_psnr:.2f} "
                f"snr {snr_db if snr_db is not None else 'clean'} elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(model, val_clips, device, args.eval_clips)
            print(
                f"eval {step}: clean {metrics['clean_psnr']:.2f} dB | "
                f"15 dB {metrics['noisy_15db_psnr']:.2f} dB | SSIM {metrics['clean_ssim']:.4f} "
                f"| baseline {baseline['clean_psnr']:.2f}",
                flush=True,
            )
            with (log_path / "train_log.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, "baseline": baseline, **metrics}) + "\n")
            if metrics["clean_psnr"] > best:
                best = metrics["clean_psnr"]
                save_checkpoint(model, Path(args.checkpoint), metrics, step, val_names)
                print(f"saved {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
