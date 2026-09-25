#!/usr/bin/env python3
"""Train the narrowband RVDJSCC adaptation at a fixed existing budget."""

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

from aetv.narrowband_jscc import global_ssim, impair_wire, psnr, resolve_bandwidth, unit_rms
from aetv.rvdjscc_narrowband import RVDJSCCNarrowband, save_fp16_shards


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


def key_only_loss(
    model: RVDJSCCNarrowband, gop: torch.Tensor, snr_db: float | None, latent_weight: float
) -> torch.Tensor:
    """Paper's first stage: train the key codec before interpolation depends on it."""
    frame = gop[:, :, 3]
    nominal = model._snr(frame.shape[0], model.nominal_snr_db, frame.device, frame.dtype)
    code = unit_rms(model.key_codec.encode(frame, nominal))
    if snr_db is None:
        received = code
        snr = model._snr(frame.shape[0], model.clean_snr_db, frame.device, torch.float32)
    else:
        received, confidence = impair_wire(code, snr_db)
        snr = model._snr_from_confidence(confidence, frame.shape[0], frame.device, frame.dtype)
    denoised = model._denoise_frame(received, snr)
    recon = model._decode_key(denoised, snr)
    return F.mse_loss(recon.float(), frame) + latent_weight * F.mse_loss(denoised.float(), code.float())


def reconstruct_pair(model: RVDJSCCNarrowband, gop0: torch.Tensor, gop1: torch.Tensor, snr_db: float | None):
    model.reset()
    wire0 = model.encode_gop(gop0, retain_state=True)
    wire1 = model.encode_gop(gop1, retain_state=True)
    model.reset()
    if snr_db is None:
        received0, received1 = wire0, wire1
        recon0, _ = model.decode_gop(wire0, retain_state=True)
        recon1, _ = model.decode_gop(wire1, retain_state=True)
        snr = model._snr(gop0.shape[0], model.clean_snr_db, gop0.device, torch.float32)
    else:
        received0, confidence0 = impair_wire(wire0, snr_db)
        received1, confidence1 = impair_wire(wire1, snr_db)
        recon0, _ = model.decode_gop(received0, confidence=confidence0, retain_state=True)
        recon1, _ = model.decode_gop(received1, confidence=confidence1, retain_state=True)
        snr = model._snr_from_confidence(confidence0, gop0.shape[0], gop0.device, gop0.dtype)
    latent = 0.5 * (
        model.denoising_loss(wire0, received0, snr) + model.denoising_loss(wire1, received1, snr)
    )
    recon = torch.cat([recon0, recon1], dim=2)
    target = torch.cat([gop0, gop1], dim=2)
    return recon, target, latent


@torch.no_grad()
def evaluate(model: RVDJSCCNarrowband, val: torch.Tensor, device: torch.device, clips: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    clean, noisy, ssim, key_scores, interp_scores = [], [], [], [], []
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
            key_scores.append(psnr(gop[:, :, 3:], recon[:, :, 3:]))
            interp_scores.append(psnr(gop[:, :, :3], recon[:, :, :3]))
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
        "key_psnr": float(sum(key_scores) / len(key_scores)),
        "interp_psnr": float(sum(interp_scores) / len(interp_scores)),
    }


def save_checkpoint(model: RVDJSCCNarrowband, path: Path, metrics: dict, step: int, val_names: list[str]) -> None:
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
        "paper": "arXiv:2601.01729",
    }
    torch.save(payload, path)
    save_fp16_shards(payload, path.with_suffix(""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandwidth", type=float, default=2.2, choices=[2.2, 8.0, 16.0])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--key-steps", type=int, default=800)
    parser.add_argument("--lambda-latent", type=float, default=0.7)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--context-channels", type=int, default=64)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--eval-clips", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="runs/rvdjscc-narrowband-v2")
    parser.add_argument("--checkpoint", default="models/rvdjscc-narrowband-v2-2.2khz-best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    khz, budget, mode = resolve_bandwidth(args.bandwidth)
    train_clips, val_clips, val_names = load_clips(Path(args.cache), args.val_clips, args.seed)
    model = RVDJSCCNarrowband(
        args.bandwidth, width=args.width, context_channels=args.context_channels
    ).to(device)
    params = sum(item.numel() for item in model.parameters())
    print(
        f"bandwidth {khz} kHz | budget {budget} | key {model.key_len} x{model.key_codec.latent_channels} | "
        f"interp {model.interp_len} x{model.interp_codec.latent_channels} | "
        f"modem {mode} | {model.architecture} | params {params/1e6:.2f}M",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    best = -1e9
    joint_scoring = False
    log_path = Path(args.out)
    log_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        gop0, gop1 = sample_pair(train_clips, args.batch, device)
        snr_db = None if random.random() < 0.3 else float(torch.empty(()).uniform_(0.0, 20.0))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if step <= args.key_steps:
                loss = 0.5 * (
                    key_only_loss(model, gop0, snr_db, args.lambda_latent)
                    + key_only_loss(model, gop1, snr_db, args.lambda_latent)
                )
                frame = loss
                latent = loss.new_zeros(())
                recon = target = gop0
            else:
                recon, target, latent = reconstruct_pair(model, gop0, gop1, snr_db)
                frame = F.mse_loss(recon.float(), target)
                loss = frame + args.lambda_latent * latent.float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            if step <= args.key_steps:
                batch_psnr = float("nan")
            else:
                batch_psnr = psnr(target, recon.detach())
            phase = "key" if step <= args.key_steps else "joint"
            print(
                f"step {step:05d}/{args.steps} {phase} loss {loss.item():.4f} frame {frame.item():.4f} "
                f"latent {float(latent):.4f} batch_psnr {batch_psnr:.2f} "
                f"snr {snr_db if snr_db is not None else 'clean'} elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(model, val_clips, device, args.eval_clips)
            if step <= args.key_steps:
                score = metrics["key_psnr"]
            else:
                if not joint_scoring:
                    best = -1e9
                    joint_scoring = True
                score = 0.5 * metrics["clean_psnr"] + 0.5 * metrics["noisy_15db_psnr"]
            print(
                f"eval {step}: clean {metrics['clean_psnr']:.2f} dB | "
                f"key {metrics['key_psnr']:.2f} | interp {metrics['interp_psnr']:.2f} | "
                f"15 dB {metrics['noisy_15db_psnr']:.2f} dB | SSIM {metrics['clean_ssim']:.4f}",
                flush=True,
            )
            with (log_path / "train_log.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, "score": score, **metrics}) + "\n")
            if score > best:
                best = score
                save_checkpoint(model, Path(args.checkpoint), metrics, step, val_names)
                print(f"saved {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
