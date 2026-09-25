#!/usr/bin/env python3
"""Fine-tune AC2K for reconstruction PSNR at the existing 2,816-coordinate budget."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.ac2k_psnr import AC2KPSNR, load_ac2k
from aetv.narrowband_jscc import global_ssim, impair_wire, psnr
from scripts.train_narrowband_jscc import pyramid_mse, ssim_loss


def load_clips(cache: Path, val_clips: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
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


def sample_pair(clips: torch.Tensor, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.randint(0, clips.shape[0], (batch,))
    start = 0 if random.random() < 0.5 else 4
    pair = clips[index, :, start : start + 8].float().div(255.0)
    if random.random() < 0.5:
        pair = pair.flip(-1)
    pair = pair.to(device)
    return pair[:, :, :4], pair[:, :, 4:]


@torch.no_grad()
def evaluate(model: AC2KPSNR, val: torch.Tensor, device: torch.device, clips: int) -> dict[str, float]:
    model.eval()
    clean, noisy, ssim = [], [], []
    for index in range(min(clips, val.shape[0])):
        clip = val[index].float().div(255.0).unsqueeze(0).to(device)
        model.reset()
        wires = []
        gops = []
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
    model.train()
    return {
        "clean_psnr": float(sum(clean) / len(clean)),
        "noisy_15db_psnr": float(sum(noisy) / len(noisy)),
        "clean_ssim": float(sum(ssim) / len(ssim)),
    }


def save_checkpoint(model: AC2KPSNR, path: Path, metrics: dict, step: int, val_names: list[str], init: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "base_architecture": model.base.architecture,
            "base_config": model.base.config,
            "init_checkpoint": init,
            "bandwidth_khz": 2.2,
            "budget": 2816,
            "modem_mode": "V8",
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metrics": metrics,
            "step": step,
            "val_clips": val_names,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="models/ac2k-v2-fidelity-best-inference.pt")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--tail-lr", type=float, default=2e-4)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--eval-clips", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="runs/ac2k-psnr")
    parser.add_argument("--checkpoint", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_clips, val_clips, val_names = load_clips(Path(args.cache), args.val_clips, args.seed)
    base = load_ac2k(args.init, device).to(device)
    model = AC2KPSNR(base).to(device)
    before = evaluate(model, val_clips, device, args.eval_clips)
    print(f"init clean {before['clean_psnr']:.2f} dB | 15 dB {before['noisy_15db_psnr']:.2f} dB", flush=True)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.base.parameters(), "lr": args.lr},
            {"params": model.tail.parameters(), "lr": args.tail_lr},
        ],
        weight_decay=1e-5,
    )
    best = 0.5 * before["clean_psnr"] + 0.5 * before["noisy_15db_psnr"]
    save_checkpoint(model, Path(args.checkpoint), before, 0, val_names, args.init)
    log_path = Path(args.out)
    log_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        gop0, gop1 = sample_pair(train_clips, args.batch, device)
        model.reset()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            wire0 = model.encode_gop(gop0, retain_state=True)
            wire1 = model.encode_gop(gop1, retain_state=True)
            model.reset()
            if random.random() < 0.4:
                snr = float(torch.empty(()).uniform_(12.0, 28.0))
                received0, confidence0 = impair_wire(wire0, snr)
                received1, confidence1 = impair_wire(wire1, snr)
                recon0, _ = model.decode_gop(received0, confidence=confidence0, retain_state=True)
                recon1, _ = model.decode_gop(received1, confidence=confidence1, retain_state=True)
            else:
                recon0, _ = model.decode_gop(wire0, retain_state=True)
                recon1, _ = model.decode_gop(wire1, retain_state=True)
            target = torch.cat([gop0, gop1], dim=2)
            recon = torch.cat([recon0, recon1], dim=2)
            loss = pyramid_mse(recon.float(), target) + 0.02 * ssim_loss(recon.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(
                f"step {step:05d}/{args.steps} loss {loss.item():.4f} "
                f"batch_psnr {psnr(target, recon.detach()):.2f} elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(model, val_clips, device, args.eval_clips)
            score = 0.5 * metrics["clean_psnr"] + 0.5 * metrics["noisy_15db_psnr"]
            print(
                f"eval {step}: clean {metrics['clean_psnr']:.2f} dB | "
                f"15 dB {metrics['noisy_15db_psnr']:.2f} dB | SSIM {metrics['clean_ssim']:.4f}",
                flush=True,
            )
            with (log_path / "train_log.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, "score": score, **metrics}) + "\n")
            if score > best:
                best = score
                save_checkpoint(model, Path(args.checkpoint), metrics, step, val_names, args.init)
                print(f"saved {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
