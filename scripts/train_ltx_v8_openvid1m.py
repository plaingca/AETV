#!/usr/bin/env python3
"""One full 1M-clip OpenVid epoch for the LTX-V8 channel pipeline.

Unlike the cache-backed pilot, this trainer streams distinct clips from the
native Hugging Face Lance dataset.  It performs source-referenced training
through the frozen LTX encoder, the 2,816-value RF adapter, and the lightly
adapted LTX decoder input convolution.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetv.channel import AETVChannelConfig, AETVLatentChannel, AETVWaveformChannel  # noqa: E402
from aetv.ltx_channel import (  # noqa: E402
    LTXV8ChannelAdapter,
    finish_ltx_video,
    latent_loss,
    prepare_ltx_video,
)
from aetv.video_data import HFDatasetsVideoDataset, VideoClipSpec  # noqa: E402
from compare_checkpoints import load_clips, metric_values  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402
from train import (  # noqa: E402
    AETV_MODES,
    dwt3d_loss,
    simulate_transmission,
    spatial_gradient_loss,
    temporal_acceleration_loss,
    temporal_cosine_loss,
    temporal_delta_loss,
    temporal_energy_loss,
)
from train_ltx_v8_channel import (  # noqa: E402
    configure_decoder_input_finetune,
    decode_video,
    load_vae,
    perceptual_video_loss,
    save_checkpoint,
    video_losses,
)


# Match the fixed five-clip evaluations used by the V8 trainer.  These exercise
# the actual V8 OFDM modem rather than the differentiable training surrogate.
EVAL_CELLS: list[tuple[str, float | None, str | None]] = [
    ("clean", None, None),
    ("18db", 18.0, None),
    ("12db", 12.0, None),
    ("10db", 10.0, None),
    ("9db", 9.0, None),
    ("6db", 6.0, None),
    ("0db", 0.0, None),
    ("minus2db", -2.0, None),
    ("mpg12", 12.0, "mpg"),
    ("mpp12", 12.0, "mpp"),
    ("mpp6", 6.0, "mpp"),
    ("mpp0", 0.0, "mpp"),
]


@torch.no_grad()
def run_heldout_evaluation(
    adapter: LTXV8ChannelAdapter,
    vae,
    clips: list[torch.Tensor],
    step: int,
    run_dir: Path,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    """Evaluate fixed held-out clips through LTX, V8 OFDM, and the decoder."""
    adapter_was_training = adapter.training
    adapter.eval()
    step_dir = run_dir / f"eval_step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    results = {
        label: {metric: [] for metric in ("psnr", "ssim", "lpips", "delta_l1", "accel_l1", "delta_lpips")}
        for label, _, _ in EVAL_CELLS
    }

    print(f"\n--- Held-out full-pipeline evaluation at step {step} ({len(clips)} clips) ---", flush=True)
    for clip_index, clip in enumerate(clips):
        target = clip.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latent = vae.encode(prepare_ltx_video(target).to(torch.bfloat16)).latent_dist.mode()
            symbols = adapter.encode(latent.float())
        symbol_numpy = symbols[0].float().cpu().numpy()
        panels: list[tuple[str, torch.Tensor]] = [("Source", target.cpu())]
        for label, snr, fading in EVAL_CELLS:
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                reconstruction = finish_ltx_video(vae.decode(restored).sample)
            values = metric_values(reconstruction, target, device)
            for metric, value in values.items():
                results[label][metric].append(value)
            panels.append((f"{label} {values['psnr']:.1f}dB", reconstruction.cpu()))
        write_labeled_grid_mp4(
            panels, step_dir / f"eval_clip_{clip_index:02d}.mp4", fps=6.0, columns=3
        )
        print(f"  clip {clip_index + 1}/{len(clips)}", flush=True)

    means = {
        label: {metric: statistics.fmean(values) for metric, values in metrics.items()}
        for label, metrics in results.items()
    }
    payload = {
        "step": step,
        "clip_count": len(clips),
        "clip_files": [f"clip_{index:05d}.pt" for index in range(len(clips))],
        "cells": [
            {"label": label, "snr_db": snr, "fading": fading}
            for label, snr, fading in EVAL_CELLS
        ],
        "means": means,
        "per_clip": results,
    }
    (step_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    for label, metrics in means.items():
        for metric, value in metrics.items():
            writer.add_scalar(f"eval/{label}_{metric}", value, step)
    print(
        "  means: "
        + " | ".join(
            f"{label}={metrics['psnr']:.2f}dB/{metrics['lpips']:.3f} LPIPS"
            for label, metrics in means.items()
        ),
        flush=True,
    )
    if adapter_was_training:
        adapter.train()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ltx-v8-openvid1m-full-20260826"))
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore optimizer and global step from --init, then finish --samples total clips.",
    )
    parser.add_argument("--dataset", default="lance-format/Openvid-1M")
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Lance workers; one traverses the complete stream without physical-shard imbalance.",
    )
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--lance-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-6)
    parser.add_argument(
        "--objective",
        choices=("baseline", "v8-perceptual", "perceptual-only"),
        default="baseline",
        help=(
            "Training loss stack; v8-perceptual ports the released V8 reference losses, "
            "while perceptual-only optimizes spatial and temporal LPIPS only."
        ),
    )
    parser.add_argument("--latent-noisy-weight", type=float, default=0.25)
    parser.add_argument("--latent-clean-weight", type=float, default=0.25)
    parser.add_argument("--latent-consistency-weight", type=float, default=0.10)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--l1-weight", type=float, default=0.80)
    parser.add_argument("--dwt-weight", type=float, default=3.0)
    parser.add_argument("--dwt-levels", type=int, default=3)
    parser.add_argument("--grad-weight", type=float, default=1.5)
    parser.add_argument("--temporal-weight", type=float, default=1.5)
    parser.add_argument("--temporal-accel-weight", type=float, default=0.3)
    parser.add_argument("--temporal-energy-weight", type=float, default=2.0)
    parser.add_argument("--temporal-cosine-weight", type=float, default=0.2)
    parser.add_argument("--vgg-weight", type=float, default=0.15)
    parser.add_argument("--clean-vgg-weight", type=float, default=0.20)
    parser.add_argument("--temporal-vgg-weight", type=float, default=0.05)
    parser.add_argument("--render-consistency-weight", type=float, default=1.0)
    parser.add_argument("--clean-anchor-weight", type=float, default=1.0)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=16.0)
    parser.add_argument("--channel-ramp", type=int, default=12_500)
    parser.add_argument("--waveform-start", type=int, default=25_000)
    parser.add_argument("--clean-batch-prob", type=float, default=0.25)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=2_000)
    parser.add_argument("--eval-clips", type=int, default=5)
    parser.add_argument(
        "--eval-cache", type=Path, default=Path("runs/openvid-cache-5fps-eval-v68")
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-save", action="store_true", help="Skip checkpoints (benchmark/smoke runs).")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples % args.batch:
        raise SystemExit("--samples must divide evenly by --batch")
    if args.workers > 2:
        raise SystemExit("OpenVid-1M exposes two Lance iterable shards; --workers must be <= 2")
    steps = args.samples // args.batch
    args.steps = steps
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

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    start_step = int(payload.get("step", 0)) if args.resume else 0
    if start_step >= steps:
        raise SystemExit(f"checkpoint step {start_step} has already reached target step {steps}")
    adapter = LTXV8ChannelAdapter().to(device)
    adapter.load_state_dict(payload["adapter"])
    vae = load_vae(device)
    decoder_conv = configure_decoder_input_finetune(vae)
    if payload.get("decoder_conv_in"):
        decoder_conv.load_state_dict(payload["decoder_conv_in"])

    if args.objective == "v8-perceptual":
        from aetv.models import MultiLayerVGGPerceptualLoss

        perceptual = MultiLayerVGGPerceptualLoss().to(device).eval()
    else:
        import lpips

        perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
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
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter.parameters(), "lr": args.lr},
            {"params": decoder_conv.parameters(), "lr": args.decoder_lr},
        ],
        weight_decay=1e-4,
        fused=True,
    )
    if args.resume:
        optimizer_state = payload.get("optimizer")
        if not optimizer_state:
            raise RuntimeError(f"resume checkpoint has no optimizer state: {args.init}")
        optimizer.load_state_dict(optimizer_state)
        print(f"Resuming checkpoint step {start_step:,}; optimizer state restored", flush=True)

    remaining_samples = (steps - start_step) * args.batch
    loader_kwargs = {}
    if args.workers > 0:
        # Hugging Face's Lance reader owns native threads and is not fork-safe.
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch,
            multiprocessing_context="spawn",
        )
    def make_loader(sample_count: int, cycle: int) -> DataLoader:
        dataset = HFDatasetsVideoDataset(
            dataset=args.dataset,
            spec=VideoClipSpec(frames=6, fps=6, height=108, width=192),
            epoch_size=sample_count,
            # Each pass uses a new deterministic shuffle. OpenVid's usable
            # Lance stream is smaller than its 1M nominal name after duration,
            # decode, and corrupt-video filtering, so a sample-budgeted epoch
            # may legitimately need more than one pass.
            seed=args.seed + start_step + 1_000_003 * cycle,
            shuffle_buffer=32,
            lance_batch_size=args.lance_batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            **loader_kwargs,
        )

    stream_cycle = 0
    loader = make_loader(remaining_samples, stream_cycle)
    iterator = iter(loader)
    writer = SummaryWriter(args.run_dir / "tensorboard")
    eval_clips: list[torch.Tensor] = []
    if args.eval_interval > 0:
        eval_clips = load_clips(args.eval_cache, AETV_MODES["V8"], args.eval_clips)
        if len(eval_clips) != args.eval_clips:
            raise RuntimeError(
                f"wanted {args.eval_clips} fixed eval clips in {args.eval_cache}, "
                f"found {len(eval_clips)}"
            )
        run_heldout_evaluation(
            adapter, vae, eval_clips, start_step, args.run_dir, writer, device
        )
    started = time.time()
    interval_started = started

    print(
        f"Streaming {remaining_samples:,} remaining OpenVid clips: steps "
        f"{start_step + 1:,}..{steps:,} x batch {args.batch}, {args.workers} workers; "
        f"objective={args.objective}",
        flush=True,
    )
    for step in range(start_step + 1, steps + 1):
        try:
            video = next(iterator)
        except StopIteration:
            completed = (step - start_step - 1) * args.batch
            still_needed = remaining_samples - completed
            stream_cycle += 1
            print(
                f"OpenVid stream pass {stream_cycle} exhausted after {completed:,} resumed "
                f"clips; starting deterministic pass {stream_cycle + 1} for "
                f"{still_needed:,} remaining clips",
                flush=True,
            )
            del iterator, loader
            loader = make_loader(still_needed, stream_cycle)
            iterator = iter(loader)
            try:
                video = next(iterator)
            except StopIteration as error:
                raise RuntimeError("OpenVid replacement stream yielded no complete batch") from error
        video = video.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            target_latent = vae.encode(prepare_ltx_video(video).to(torch.bfloat16)).latent_dist.mode().float()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            symbols = adapter.encode(target_latent)
            mix = min(1.0, step / max(args.channel_ramp, 1))
            channel = waveform_channel if step >= args.waveform_start else latent_channel
            impaired, confidence = channel(symbols.float())
            received = symbols.lerp(impaired.to(symbols), mix)
            confidence = torch.ones_like(confidence).lerp(confidence.to(symbols), mix)
            clean_batch = random.random() < args.clean_batch_prob
            if clean_batch:
                received = symbols
                confidence = torch.ones_like(symbols)

            restored = adapter.decode(received, confidence)
            reconstruction = decode_video(vae, restored)
            if args.objective == "perceptual-only":
                # Deliberately exclude latent NMSE, MSE/L1, wavelet, gradient,
                # clean-anchor, and render-consistency terms. This branch is
                # allowed to sacrifice clean reconstruction for maximum
                # channel-impaired spatial and temporal LPIPS quality.
                reconstruction_delta = (
                    0.5 * (reconstruction[:, :, 1:] - reconstruction[:, :, :-1]) + 0.5
                ).clamp(0, 1)
                video_delta = (
                    0.5 * (video[:, :, 1:] - video[:, :, :-1]) + 0.5
                ).clamp(0, 1)
                loss_vgg = perceptual_video_loss(perceptual, reconstruction, video)
                loss_temporal_vgg = perceptual_video_loss(
                    perceptual, reconstruction_delta, video_delta
                )
                loss = (
                    args.vgg_weight * loss_vgg
                    + args.temporal_vgg_weight * loss_temporal_vgg
                )
                loss_dwt = reconstruction.new_zeros(())
                loss_clean_vgg = reconstruction.new_zeros(())
                loss_render_consistency = reconstruction.new_zeros(())
                loss_clean_anchor = reconstruction.new_zeros(())
            else:
                restored_clean = adapter.decode(symbols, torch.ones_like(symbols))
                loss_noisy_latent = latent_loss(restored, target_latent)
                loss_clean_latent = latent_loss(restored_clean, target_latent)
                loss_consistency = F.smooth_l1_loss(
                    restored.float(), restored_clean.float().detach(), beta=0.1
                )
                reconstruction_clean = decode_video(vae, restored_clean)
                loss_video = video_losses(reconstruction, video)
                loss_video_clean = video_losses(reconstruction_clean, video)
            if args.objective == "baseline":
                loss_perceptual = perceptual_video_loss(
                    perceptual, reconstruction, video
                )
                loss_perceptual_clean = perceptual_video_loss(
                    perceptual, reconstruction_clean, video
                )
                loss = (
                    loss_noisy_latent
                    + loss_clean_latent
                    + 0.1 * loss_consistency
                    + 2.0 * loss_video
                    + 2.0 * loss_video_clean
                    + 0.15 * loss_perceptual
                    + 0.10 * loss_perceptual_clean
                )
                loss_dwt = reconstruction.new_zeros(())
                loss_vgg = loss_perceptual
                loss_temporal_vgg = reconstruction.new_zeros(())
                loss_render_consistency = reconstruction.new_zeros(())
                loss_clean_anchor = loss_video_clean
            elif args.objective == "v8-perceptual":
                loss_mse = F.mse_loss(reconstruction, video)
                loss_l1 = F.l1_loss(reconstruction, video)
                loss_grad = spatial_gradient_loss(reconstruction, video)
                loss_temporal = temporal_delta_loss(reconstruction, video)
                loss_temporal_accel = temporal_acceleration_loss(reconstruction, video)
                loss_temporal_energy = temporal_energy_loss(reconstruction, video)
                loss_temporal_cosine = temporal_cosine_loss(reconstruction, video)
                loss_dwt = dwt3d_loss(reconstruction, video, levels=args.dwt_levels)
                loss_vgg = perceptual(reconstruction, video)
                loss_clean_vgg = perceptual(reconstruction_clean, video)
                reconstruction_delta = (
                    0.5 * (reconstruction[:, :, 1:] - reconstruction[:, :, :-1]) + 0.5
                ).clamp(0, 1)
                video_delta = (
                    0.5 * (video[:, :, 1:] - video[:, :, :-1]) + 0.5
                ).clamp(0, 1)
                loss_temporal_vgg = perceptual(reconstruction_delta, video_delta)
                loss_render_consistency = F.l1_loss(
                    reconstruction, reconstruction_clean.detach()
                )
                loss_clean_anchor = (
                    args.mse_weight * F.mse_loss(reconstruction_clean, video)
                    + args.l1_weight * F.l1_loss(reconstruction_clean, video)
                    + args.grad_weight * spatial_gradient_loss(reconstruction_clean, video)
                    + args.temporal_weight
                    * temporal_delta_loss(reconstruction_clean, video)
                    + args.temporal_accel_weight
                    * temporal_acceleration_loss(reconstruction_clean, video)
                )
                loss = (
                    args.latent_noisy_weight * loss_noisy_latent
                    + args.latent_clean_weight * loss_clean_latent
                    + args.latent_consistency_weight * loss_consistency
                    + args.mse_weight * loss_mse
                    + args.l1_weight * loss_l1
                    + args.dwt_weight * loss_dwt
                    + args.grad_weight * loss_grad
                    + args.temporal_weight * loss_temporal
                    + args.temporal_accel_weight * loss_temporal_accel
                    + args.temporal_energy_weight * loss_temporal_energy
                    + args.temporal_cosine_weight * loss_temporal_cosine
                    + args.vgg_weight * loss_vgg
                    + args.clean_vgg_weight * loss_clean_vgg
                    + args.temporal_vgg_weight * loss_temporal_vgg
                    + args.render_consistency_weight * loss_render_consistency
                    + args.clean_anchor_weight * loss_clean_anchor
                )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
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
        writer.add_scalar("train/clean_batch", float(clean_batch), step)
        writer.add_scalar("train/samples", step * args.batch, step)
        if args.objective in ("v8-perceptual", "perceptual-only"):
            writer.add_scalar("train/loss_dwt", loss_dwt.item(), step)
            writer.add_scalar("train/loss_vgg", loss_vgg.item(), step)
            writer.add_scalar("train/loss_clean_vgg", loss_clean_vgg.item(), step)
            writer.add_scalar("train/loss_temporal_vgg", loss_temporal_vgg.item(), step)
            writer.add_scalar(
                "train/loss_render_consistency", loss_render_consistency.item(), step
            )
            writer.add_scalar("train/loss_clean_anchor", loss_clean_anchor.item(), step)

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - interval_started
            interval_steps = 1 if step == 1 else args.log_every
            rate = interval_steps / max(elapsed, 1e-6)
            resumed_clips = (step - start_step) * args.batch
            total_rate = resumed_clips / max(time.time() - started, 1e-6)
            print(
                f"step {step:6d}/{steps} samples={step * args.batch:7d}/{args.samples} "
                f"loss={loss.item():.4f} nmse={nmse.item():.4f} mix={mix:.2f} "
                f"clean={int(clean_batch)} {rate:.2f} step/s {total_rate:.1f} clips/s",
                flush=True,
            )
            if args.objective in ("v8-perceptual", "perceptual-only"):
                print(
                    f"  dwt={loss_dwt.item():.4f} vgg={loss_vgg.item():.3f}/"
                    f"{loss_clean_vgg.item():.3f}clean "
                    f"tvgg={loss_temporal_vgg.item():.3f} "
                    f"render_cons={loss_render_consistency.item():.4f} "
                    f"clean_anchor={loss_clean_anchor.item():.4f} "
                    f"vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                    flush=True,
                )
            interval_started = time.time()
        if not args.no_save and (step % args.checkpoint_every == 0 or step == steps):
            save_checkpoint(
                args.run_dir / f"checkpoint-{step:06d}.pt",
                adapter, decoder_conv, optimizer, step, args,
            )
            save_checkpoint(
                args.run_dir / "checkpoint.pt", adapter, decoder_conv, optimizer, step, args
            )
            numbered = sorted(
                args.run_dir.glob("checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9].pt"),
                key=lambda path: path.stat().st_mtime,
            )
            for obsolete in numbered[: -args.keep_checkpoints] if args.keep_checkpoints > 0 else []:
                obsolete.unlink()
        if args.eval_interval > 0 and (step % args.eval_interval == 0 or step == steps):
            run_heldout_evaluation(
                adapter, vae, eval_clips, step, args.run_dir, writer, device
            )

    writer.close()
    print(
        f"Completed target of {args.samples:,} OpenVid clips in "
        f"{(time.time() - started) / 3600:.2f} hours",
        flush=True,
    )


if __name__ == "__main__":
    main()
