#!/usr/bin/env python3
"""Train an LTX-Video VAE transport adapter for AETV V8.

The run is deliberately staged for throughput:

1. Cache the frozen LTX posterior modes for every local OpenVid V8 clip.
2. Train the 6,144 -> 2,816 -> 6,144 adapter mostly in latent space.
3. Finish with source-referenced video losses through the LTX decoder and a
   low-rate update of its latent input convolution.

All training samples come from the existing local V8 OpenVid cache.  The
separate 32-clip SSTVAE cache is never sampled by this trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aetv.channel import AETVChannelConfig, AETVLatentChannel, AETVWaveformChannel  # noqa: E402
from aetv.ltx_channel import (  # noqa: E402
    LTX_REPO,
    LTXV8ChannelAdapter,
    finish_ltx_video,
    latent_loss,
    prepare_ltx_video,
)


def load_vae(device: torch.device):
    from diffusers import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_pretrained(
        LTX_REPO, subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    vae.requires_grad_(False)
    return vae


def load_video(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.ndim != 4 or tuple(value.shape) != (3, 6, 108, 192):
        raise ValueError(f"unexpected cached clip {path}: {tuple(value.shape)}")
    return value.float().div_(255.0) if value.dtype == torch.uint8 or value.max() > 1 else value.float()


@torch.no_grad()
def build_latent_cache(args, device: torch.device, cache_path: Path) -> dict:
    files = sorted(Path(args.train_cache).glob("*.pt"))
    if not files:
        raise SystemExit(f"no training clips in {args.train_cache}")
    if args.max_clips:
        files = files[: args.max_clips]
    print(f"Caching LTX latents for {len(files)} OpenVid clips...", flush=True)
    vae = load_vae(device)
    batches: list[torch.Tensor] = []
    started = time.time()
    for offset in range(0, len(files), args.cache_batch):
        paths = files[offset : offset + args.cache_batch]
        video = torch.stack([load_video(path) for path in paths]).to(device)
        prepared = prepare_ltx_video(video).to(torch.bfloat16)
        latent = vae.encode(prepared).latent_dist.mode()
        batches.append(latent.cpu().to(torch.float16))
        if offset == 0 or (offset + len(paths)) % 1000 == 0:
            rate = (offset + len(paths)) / max(time.time() - started, 1e-6)
            print(f"  {offset + len(paths):5d}/{len(files)} ({rate:.1f} clips/s)", flush=True)
    latents = torch.cat(batches)
    stats_value = latents.float()
    mean = stats_value.mean(dim=(0, 2, 3, 4))
    std = stats_value.std(dim=(0, 2, 3, 4)).clamp_min(1e-4)
    payload = {
        "repo": LTX_REPO,
        "files": [str(path) for path in files],
        "latents": latents,
        "mean": mean,
        "std": std,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(
        f"Wrote {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB) in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    del vae
    torch.cuda.empty_cache()
    return payload


def next_indexes(count: int, batch: int, generator: torch.Generator):
    order = torch.randperm(count, generator=generator)
    cursor = 0
    while True:
        if cursor + batch > count:
            order = torch.randperm(count, generator=generator)
            cursor = 0
        selected = order[cursor : cursor + batch]
        cursor += batch
        yield selected


def save_checkpoint(path: Path, adapter, decoder_conv, optimizer, step: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "aetv-ltx-v8-channel-v1",
        "step": step,
        "ltx_repo": LTX_REPO,
        "adapter": adapter.state_dict(),
        "decoder_conv_in": decoder_conv.state_dict() if decoder_conv is not None else None,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def configure_decoder_input_finetune(vae):
    module = vae.decoder.conv_in
    module.float().requires_grad_(True)
    return module


def video_losses(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(reconstruction, target) + 0.5 * F.mse_loss(reconstruction, target)
    grad_r_x = reconstruction[..., 1:] - reconstruction[..., :-1]
    grad_t_x = target[..., 1:] - target[..., :-1]
    grad_r_y = reconstruction[..., 1:, :] - reconstruction[..., :-1, :]
    grad_t_y = target[..., 1:, :] - target[..., :-1, :]
    delta_r = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
    delta_t = target[:, :, 1:] - target[:, :, :-1]
    return (
        loss
        + 0.25 * (F.l1_loss(grad_r_x, grad_t_x) + F.l1_loss(grad_r_y, grad_t_y))
        + 0.5 * F.l1_loss(delta_r, delta_t)
    )


def perceptual_video_loss(model, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = reconstruction.shape
    predicted = reconstruction.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    reference = target.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    return model(predicted * 2 - 1, reference * 2 - 1).mean()


def decode_video(vae, latent: torch.Tensor) -> torch.Tensor:
    return finish_ltx_video(vae.decode(latent).sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ltx-v8-openvid-full-20260826"))
    parser.add_argument("--train-cache", type=Path, default=Path("data/openvid_aetv_cache/mode_v8_192x108_6fps"))
    parser.add_argument("--steps", type=int, default=125_000)
    parser.add_argument("--latent-steps", type=int, default=100_000)
    parser.add_argument("--waveform-start", type=int, default=85_000)
    parser.add_argument("--channel-start", type=int, default=15_000)
    parser.add_argument("--channel-ramp", type=int, default=10_000)
    parser.add_argument("--batch-latent", type=int, default=192)
    parser.add_argument("--batch-video", type=int, default=4)
    parser.add_argument("--cache-batch", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--video-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-6)
    parser.add_argument("--clean-batch-prob", type=float, default=0.25)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=16.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.latent_steps > args.steps:
        raise SystemExit("--latent-steps cannot exceed --steps")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)

    latent_cache_path = args.run_dir / "openvid-ltx-latents.pt"
    if args.rebuild_cache or not latent_cache_path.exists():
        cache = build_latent_cache(args, device, latent_cache_path)
    else:
        print(f"Loading {latent_cache_path}...", flush=True)
        cache = torch.load(latent_cache_path, map_location="cpu", weights_only=False)
    latents = cache["latents"]
    files = cache["files"]
    print(f"Training set: {len(files)} clips, latent tensor {tuple(latents.shape)}", flush=True)

    adapter = LTXV8ChannelAdapter().to(device)
    adapter.set_latent_stats(cache["mean"], cache["std"])
    latent_channel = AETVLatentChannel(
        AETVChannelConfig(
            snr_db_range=(args.snr_min, args.snr_max),
            p_truncate=0.10,
            erasure_rate_max=0.03,
        )
    ).to(device)
    waveform_channel = AETVWaveformChannel(
        "W",
        AETVChannelConfig(
            snr_db_range=(args.snr_min, args.snr_max),
            p_fading=0.50,
            p_measured_path=0.40,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    start_step = 0
    decoder_conv = None
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        adapter.load_state_dict(resume_payload["adapter"])
        start_step = int(resume_payload["step"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        print(f"Resuming from step {start_step}", flush=True)

    writer = SummaryWriter(args.run_dir / "tensorboard")
    cpu_generator = torch.Generator().manual_seed(args.seed)
    latent_indexes = next_indexes(len(files), args.batch_latent, cpu_generator)
    video_indexes = next_indexes(len(files), args.batch_video, cpu_generator)
    vae = None
    perceptual = None
    started = time.time()
    interval_started = started

    for step in range(start_step + 1, args.steps + 1):
        video_stage = step > args.latent_steps
        if video_stage and vae is None:
            print(f"Loading LTX decoder for source-referenced stage at step {step}...", flush=True)
            vae = load_vae(device)
            decoder_conv = configure_decoder_input_finetune(vae)
            import lpips

            perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
            if resume_payload and resume_payload.get("decoder_conv_in"):
                decoder_conv.load_state_dict(resume_payload["decoder_conv_in"])
            optimizer = torch.optim.AdamW(
                [
                    {"params": adapter.parameters(), "lr": args.video_lr},
                    {"params": decoder_conv.parameters(), "lr": args.decoder_lr},
                ],
                weight_decay=1e-4,
                fused=True,
            )
            torch.cuda.empty_cache()

        indexes = next(video_indexes if video_stage else latent_indexes)
        target_latent = latents[indexes].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            symbols = adapter.encode(target_latent)
            if step < args.channel_start:
                mix = 0.0
            else:
                mix = min(1.0, (step - args.channel_start) / max(args.channel_ramp, 1))
            if mix:
                channel = waveform_channel if step >= args.waveform_start else latent_channel
                impaired, confidence = channel(symbols.float())
                received = symbols.lerp(impaired.to(symbols), mix)
                confidence = torch.ones_like(confidence).lerp(confidence.to(symbols), mix)
                if video_stage and random.random() < args.clean_batch_prob:
                    received = symbols
                    confidence = torch.ones_like(symbols)
            else:
                received, confidence = symbols, torch.ones_like(symbols)

            restored = adapter.decode(received, confidence)
            restored_clean = adapter.decode(symbols, torch.ones_like(symbols))
            loss_noisy_latent = latent_loss(restored, target_latent)
            loss_clean_latent = latent_loss(restored_clean, target_latent)
            loss_consistency = F.smooth_l1_loss(restored.float(), restored_clean.float().detach(), beta=0.1)
            clean_latent_weight = 1.0 if video_stage else 0.5
            loss = loss_noisy_latent + clean_latent_weight * loss_clean_latent + 0.1 * loss_consistency

            loss_video = torch.zeros((), device=device)
            loss_video_clean = torch.zeros((), device=device)
            if video_stage:
                target_video = torch.stack([load_video(files[int(index)]) for index in indexes]).to(device)
                reconstruction = decode_video(vae, restored.to(torch.bfloat16))
                reconstruction_clean = decode_video(vae, restored_clean.to(torch.bfloat16))
                loss_video = video_losses(reconstruction, target_video)
                loss_video_clean = video_losses(reconstruction_clean, target_video)
                loss_perceptual = perceptual_video_loss(perceptual, reconstruction, target_video)
                loss_perceptual_clean = perceptual_video_loss(
                    perceptual, reconstruction_clean, target_video
                )
                loss = (
                    loss
                    + 2.0 * loss_video
                    + 2.0 * loss_video_clean
                    + 0.15 * loss_perceptual
                    + 0.10 * loss_perceptual_clean
                )

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if decoder_conv is not None:
            torch.nn.utils.clip_grad_norm_(decoder_conv.parameters(), 0.1)
        optimizer.step()

        with torch.no_grad():
            nmse = (
                (restored.float() - target_latent).square().mean()
                / target_latent.square().mean().clamp_min(1e-8)
            )
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/latent_nmse", nmse.item(), step)
        writer.add_scalar("train/channel_mix", mix, step)
        if video_stage:
            writer.add_scalar("train/video_loss", loss_video.item(), step)
            writer.add_scalar("train/video_clean_loss", loss_video_clean.item(), step)

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - interval_started
            rate = args.log_every / elapsed if step > 1 else 1 / max(elapsed, 1e-6)
            print(
                f"step {step:6d}/{args.steps} loss={loss.item():.5f} "
                f"nmse={nmse.item():.5f} mix={mix:.2f} "
                f"stage={'video' if video_stage else 'latent'} {rate:.1f} step/s",
                flush=True,
            )
            interval_started = time.time()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_checkpoint(args.run_dir / f"checkpoint-{step:06d}.pt", adapter, decoder_conv, optimizer, step, args)
            save_checkpoint(args.run_dir / "checkpoint.pt", adapter, decoder_conv, optimizer, step, args)
            numbered = sorted(
                args.run_dir.glob("checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9].pt"),
                key=lambda path: path.stat().st_mtime,
            )
            for obsolete in numbered[: -args.keep_checkpoints] if args.keep_checkpoints > 0 else []:
                obsolete.unlink()

    writer.close()
    print(f"Completed {args.steps} steps in {(time.time() - started) / 3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
