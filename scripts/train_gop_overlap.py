#!/usr/bin/env python3
"""Full-train the fixed-rate centered-window AETV GOP codec.

The encoder transmits one 2,816-value V8 latent per six-frame GOP.  The default
decoder buffers five adjacent latent vectors and emits the centered three GOPs,
adding one GOP of receive lookahead without changing symbols per second.  This
amortizes decoding while temporal and boundary losses score adjacent output GOPs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.channel import (  # noqa: E402
    AETVChannelConfig,
    AETVLatentChannel,
    AETVWaveformChannel,
)
from aetv.config import AETV_MODES  # noqa: E402
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from aetv.overlap_models import OverlappingGOPAutoencoder  # noqa: E402
from aetv.video_data import HFDatasetsVideoDataset, VideoClipSpec  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    SequenceCache,
    boundary_losses,
    cache_name,
    sequence_metrics,
)
from train import (  # noqa: E402
    dwt3d_loss,
    spatial_gradient_loss,
    temporal_acceleration_loss,
    temporal_delta_loss,
)


def initialize_from_released(model: OverlappingGOPAutoencoder, checkpoint: Path) -> None:
    """Transplant the released V8 encoder/decoder and retain its latent order."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    released = payload["model_state_dict"]
    state = model.state_dict()
    copied = 0
    for key in ("encoder.encoder.", "decoder.decoder."):
        for name in list(state):
            if name.startswith(key) and name in released and state[name].shape == released[name].shape:
                state[name].copy_(released[name])
                copied += 1
    # The released non-compact V8 layout is the first 2,808 grid coordinates;
    # the final eight wire values are intentionally unused by its decoder.
    pack = state["encoder.latent_pack.weight"]
    unpack = state["decoder.latent_unpack.weight"]
    if unpack.shape[1] != pack.shape[0]:
        raise RuntimeError("unexpected V8 overlap latent geometry")
    pack.zero_()
    unpack.zero_()
    pack_diagonal = min(pack.shape[0], pack.shape[1])
    unpack_diagonal = min(unpack.shape[0], unpack.shape[1])
    pack[:pack_diagonal, :pack_diagonal].copy_(torch.eye(pack_diagonal, dtype=pack.dtype))
    unpack[:unpack_diagonal, :unpack_diagonal].copy_(torch.eye(unpack_diagonal, dtype=unpack.dtype))
    model.load_state_dict(state, strict=True)
    print(f"initialized overlap model from {checkpoint} ({copied} released tensors)", flush=True)


def cached_loader(args: argparse.Namespace, mode, device: torch.device):
    dataset = SequenceCache(
        args.train_cache
        if args.train_cache is not None
        else args.data_dir / cache_name(mode, args.train_gops, "train"),
        max_frames=args.train_gops * mode.gop_frames,
    )
    generator = torch.Generator().manual_seed(args.seed)
    while True:
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            drop_last=True,
            generator=generator,
            num_workers=0,
        )
        for batch in loader:
            yield batch.to(device, non_blocking=True)


def streaming_loader(args: argparse.Namespace, mode, device: torch.device):
    spec = VideoClipSpec(
        frames=args.train_gops * mode.gop_frames,
        fps=mode.fps,
        height=mode.height,
        width=mode.width,
    )
    restart = 0
    while True:
        loader = None
        try:
            dataset = HFDatasetsVideoDataset(
                dataset=args.hf_dataset,
                split="train",
                spec=spec,
                epoch_size=max(args.steps * args.batch, 1000),
                seed=args.seed + restart * 1009,
                shuffle_buffer=32,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch,
                num_workers=args.workers,
                multiprocessing_context="spawn" if args.workers > 0 else None,
                pin_memory=True,
                persistent_workers=args.workers > 0,
                prefetch_factor=2 if args.workers > 0 else None,
                drop_last=True,
            )
            for batch in loader:
                yield batch.to(device, non_blocking=True)
            restart += 1
        except Exception as error:
            message = str(error).lower()
            transient = any(
                marker in message
                for marker in (
                    "status: 429",
                    "rate limit",
                    "timed out",
                    "connection reset",
                    "service unavailable",
                    "status: 502",
                    "status: 503",
                    "status: 504",
                )
            )
            if not transient:
                raise
            restart += 1
            delay = min(300, 30 * 2 ** min(restart - 1, 4))
            print(
                f"streaming source transient failure; retrying in {delay}s: "
                f"{type(error).__name__}",
                flush=True,
            )
            # Dropping the loader shuts down failed persistent workers before
            # opening a fresh dataset iterator with a different shuffle seed.
            del loader
            time.sleep(delay)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[OverlappingGOPAutoencoder, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != OverlappingGOPAutoencoder.checkpoint_kind:
        raise ValueError(f"{path} is not an overlapping-GOP checkpoint")
    config = payload["model_config"]
    model = OverlappingGOPAutoencoder(
        mode=config["mode"],
        width=config["width"],
        latent_channels=config["latent_channels"],
        window_gops=config["window_gops"],
        emit_gops=config.get("emit_gops", 1),
        synthesis_halo_frames=config.get("synthesis_halo_frames", 2),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def save_checkpoint(
    path: Path,
    model: OverlappingGOPAutoencoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "kind": model.checkpoint_kind,
            "step": step,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_args": vars(args),
            "wire_contract": {
                "mode": model.mode.name,
                "latents_per_gop": model.latent_budget,
                "gop_seconds": model.mode.gop_frames / model.mode.fps,
                "lookahead_gops": model.lookahead_gops,
                "transmitted_values_per_second": model.latent_budget,
            },
        },
        temporary,
    )
    temporary.replace(path)


def update_latest_and_prune(out: Path, checkpoint: Path, keep: int) -> None:
    latest = out / "latest.pt"
    latest.unlink(missing_ok=True)
    latest.symlink_to(checkpoint.name)
    checkpoints = sorted(out.glob("checkpoint_step_*.pt"))
    for stale in checkpoints[:-max(1, keep)]:
        stale.unlink()


def corrupt_latent_sequence(
    channel: AETVLatentChannel | AETVWaveformChannel,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, gops, budget = latents.shape
    noisy, weights = channel(latents.reshape(batch * gops, budget).float())
    return noisy.reshape_as(latents).to(latents.dtype), weights.reshape_as(latents).to(
        latents.dtype
    )


def compute_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    perceptual: MultiLayerVGGPerceptualLoss | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    frames_per_gop = AETV_MODES[args.mode].gop_frames
    delta_error = (recon[:, :, 1:] - recon[:, :, :-1]) - (
        target[:, :, 1:] - target[:, :, :-1]
    )
    boundaries = list(range(frames_per_gop, recon.shape[2], frames_per_gop))
    if not boundaries:
        raise ValueError("overlap training requires at least one GOP boundary")
    boundary_delta = torch.stack(
        [delta_error[:, :, boundary - 1] for boundary in boundaries], dim=2
    ).abs().mean()
    acceleration_error = delta_error[:, :, 1:] - delta_error[:, :, :-1]
    acceleration = torch.stack(
        [acceleration_error[:, :, boundary - 2:boundary] for boundary in boundaries], dim=2
    ).abs().mean()
    lowpass = F.avg_pool2d(
        delta_error.permute(0, 2, 1, 3, 4).flatten(0, 1), 9, 1, 4
    ).reshape(delta_error.shape[0], delta_error.shape[2], delta_error.shape[1], *delta_error.shape[-2:]).permute(0, 2, 1, 3, 4)
    boundary_lowpass = torch.stack(
        [lowpass[:, :, boundary - 1] for boundary in boundaries], dim=2
    ).abs().mean()
    values = {
        "mse": F.mse_loss(recon, target),
        "l1": F.l1_loss(recon, target),
        "dwt": dwt3d_loss(recon, target, levels=args.dwt_levels),
        "gradient": spatial_gradient_loss(recon, target),
        "temporal": temporal_delta_loss(recon, target),
        "acceleration": temporal_acceleration_loss(recon, target),
        "boundary": boundary_delta,
        "boundary_lowpass": boundary_lowpass,
        "boundary_acceleration": acceleration,
    }
    if perceptual is None:
        values["perceptual"] = recon.new_zeros(())
    else:
        stride = max(1, args.perceptual_stride)
        values["perceptual"] = perceptual(
            recon[:, :, ::stride], target[:, :, ::stride]
        )
    total = (
        args.mse_weight * values["mse"]
        + args.l1_weight * values["l1"]
        + args.dwt_weight * values["dwt"]
        + args.gradient_weight * values["gradient"]
        + args.temporal_weight * values["temporal"]
        + args.acceleration_weight * values["acceleration"]
        + args.boundary_weight * values["boundary"]
        + args.boundary_lowpass_weight * values["boundary_lowpass"]
        + args.boundary_acceleration_weight * values["boundary_acceleration"]
        + args.perceptual_weight * values["perceptual"]
    )
    return total, values


def evaluate(
    model: OverlappingGOPAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
) -> dict | None:
    mode = model.mode
    cache = (
        args.eval_cache
        if args.eval_cache is not None
        else args.data_dir / cache_name(mode, args.train_gops, "eval")
    )
    try:
        dataset = SequenceCache(cache, max_frames=args.train_gops * mode.gop_frames)
    except ValueError:
        print(f"evaluation skipped: no fixed cache at {cache}", flush=True)
        return None
    rows = []
    model.eval()
    with torch.inference_mode():
        for index in range(min(args.eval_sequences, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            latents = model.encode_sequence(source)
            recon = model.decode_sequence(latents)
            target = model.target_for_sequence(source)
            rows.append(
                sequence_metrics(
                    recon,
                    target,
                    mode.gop_frames,
                    device,
                    include_lpips=args.eval_lpips,
                )
            )
    return {
        "step": args._current_step,
        "sequences": rows,
        "means": {
            key: sum(row[key] for row in rows) / len(rows) for key in rows[0]
        },
    }


def train(args: argparse.Namespace) -> None:
    if args.train_gops < args.window_gops:
        raise SystemExit("--train-gops must be at least --window-gops")
    if (args.train_gops - args.window_gops) % args.emit_gops:
        raise SystemExit("training GOPs must tile windows at --emit-gops stride")
    emitted_gops = (
        (args.train_gops - args.window_gops) // args.emit_gops + 1
    ) * args.emit_gops
    if emitted_gops < 3:
        raise SystemExit("configuration must emit at least three GOPs per sample")
    mode = AETV_MODES[args.mode]
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out / "tensorboard")

    if args.resume:
        model, payload = load_checkpoint(args.resume, device)
        start_step = int(payload["step"])
    else:
        model = OverlappingGOPAutoencoder(
            mode=mode,
            width=args.model_width,
            latent_channels=args.latent_channels,
            window_gops=args.window_gops,
            emit_gops=args.emit_gops,
            synthesis_halo_frames=args.synthesis_halo_frames,
        ).to(device)
        payload = None
        start_step = 0
        if args.init_checkpoint:
            initialize_from_released(model, args.init_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4
    )
    if payload is not None and not args.reset_optimizer:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.min_lr
    )
    # The scheduler advances only when the optimizer does, once per accumulation
    # group. Reconstruct that update count for older checkpoints that predate a
    # serialized scheduler state.
    for _ in range(start_step // args.accum):
        scheduler.step()

    perceptual = (
        MultiLayerVGGPerceptualLoss().to(device).eval()
        if args.perceptual_weight > 0
        else None
    )
    channel_config = AETVChannelConfig(
        snr_db_range=(args.snr_min, args.snr_max),
        p_fading=args.p_fading if args.channel_kind == "waveform" else 0.0,
        p_measured_path=args.p_measured_path,
        p_truncate=args.p_truncate if args.channel_kind == "latent" else 0.0,
        erasure_rate_max=(
            args.erasure_rate_max if args.channel_kind == "latent" else 0.0
        ),
    )
    channel = (
        AETVWaveformChannel(band=mode.band, cfg=channel_config)
        if args.channel_kind == "waveform"
        else AETVLatentChannel(channel_config)
    ).to(device)
    batches = (
        cached_loader(args, mode, device)
        if args.data_source == "cache"
        else streaming_loader(args, mode, device)
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "model": model.config(),
                "parameters": parameter_count,
                "train_gops": args.train_gops,
                "emitted_gops_per_sample": emitted_gops,
                "source": args.data_source,
                "channel": args.channel_kind,
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        args._current_step = step
        model.train()
        source = next(batches)
        if source.max() > 1:
            source = source.float().div_(255)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            latents = model.encode_sequence(
                source, checkpoint_encoder=args.gradient_checkpointing
            )
            if step <= args.clean_warmup:
                channel_mix = 0.0
            elif step < args.clean_warmup + args.channel_ramp:
                channel_mix = (step - args.clean_warmup) / max(1, args.channel_ramp)
            else:
                channel_mix = 1.0
            if channel_mix > 0:
                noisy, noisy_weights = corrupt_latent_sequence(channel, latents)
                received = latents + channel_mix * (noisy - latents)
                weights = torch.ones_like(latents) + channel_mix * (
                    noisy_weights - 1.0
                )
            else:
                received = latents
                weights = torch.ones_like(latents)
            recon = model.decode_sequence(
                received,
                weights,
                checkpoint_windows=args.gradient_checkpointing,
            )
            target = model.target_for_sequence(source)
            total, losses = compute_losses(recon, target, perceptual, args)
            scaled = total / args.accum
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}")
        scaled.backward()
        if step % args.accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        for key, value in losses.items():
            writer.add_scalar(f"train/{key}", float(value.detach()), step)
        writer.add_scalar("train/total", float(total.detach()), step)
        writer.add_scalar("train/channel_mix", channel_mix, step)
        if step == 1 or step % args.log_interval == 0:
            memory = (
                torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            )
            print(
                f"step {step:>7}/{args.steps} total={float(total.detach()):.5f} "
                f"l1={float(losses['l1'].detach()):.5f} "
                f"boundary={float(losses['boundary'].detach()):.5f} "
                f"perceptual={float(losses['perceptual'].detach()):.5f} "
                f"channel={channel_mix:.2f} vram={memory:.2f}GiB "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True,
            )
        if step % args.checkpoint_interval == 0 or step == args.steps:
            checkpoint = out / f"checkpoint_step_{step:07d}.pt"
            save_checkpoint(checkpoint, model, optimizer, step, args)
            update_latest_and_prune(out, checkpoint, args.keep_checkpoints)
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            report = evaluate(model, args, device)
            if report is not None:
                path = out / f"eval_step_{step:07d}.json"
                path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
                print(json.dumps(report["means"], sort_keys=True), flush=True)
    writer.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("train", "eval"))
    value.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    value.add_argument("--out", type=Path, default=Path("runs/v8-overlap-full"))
    value.add_argument("--resume", type=Path)
    value.add_argument("--init-checkpoint", type=Path)
    value.add_argument("--reset-optimizer", action="store_true")
    value.add_argument("--data-source", choices=("stream", "cache"), default="stream")
    value.add_argument("--data-dir", type=Path, default=Path("runs/gop-boundary-data"))
    value.add_argument("--train-cache", type=Path)
    value.add_argument("--eval-cache", type=Path)
    value.add_argument("--hf-dataset", default="lance-format/Openvid-1M")
    value.add_argument("--train-gops", type=int, default=5)
    value.add_argument("--window-gops", type=int, default=5)
    value.add_argument("--emit-gops", type=int, default=3)
    value.add_argument("--synthesis-halo-frames", type=int, default=2)
    value.add_argument("--model-width", type=int, default=256)
    value.add_argument("--latent-channels", type=int, default=8)
    value.add_argument("--steps", type=int, default=250000)
    value.add_argument("--batch", type=int, default=1)
    value.add_argument("--accum", type=int, default=8)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--lr", type=float, default=1e-4)
    value.add_argument("--min-lr", type=float, default=1e-6)
    value.add_argument("--clean-warmup", type=int, default=20000)
    value.add_argument("--channel-ramp", type=int, default=20000)
    value.add_argument("--snr-min", type=float, default=0.0)
    value.add_argument("--snr-max", type=float, default=18.0)
    value.add_argument("--channel-kind", choices=("latent", "waveform"), default="waveform")
    value.add_argument("--p-fading", type=float, default=0.70)
    value.add_argument("--p-measured-path", type=float, default=0.40)
    value.add_argument("--p-truncate", type=float, default=0.15)
    value.add_argument("--erasure-rate-max", type=float, default=0.04)
    value.add_argument("--mse-weight", type=float, default=2.0)
    value.add_argument("--l1-weight", type=float, default=1.5)
    value.add_argument("--dwt-weight", type=float, default=0.5)
    value.add_argument("--dwt-levels", type=int, default=3)
    value.add_argument("--gradient-weight", type=float, default=0.5)
    value.add_argument("--temporal-weight", type=float, default=1.0)
    value.add_argument("--acceleration-weight", type=float, default=1.0)
    value.add_argument("--boundary-weight", type=float, default=8.0)
    value.add_argument("--boundary-lowpass-weight", type=float, default=4.0)
    value.add_argument("--boundary-acceleration-weight", type=float, default=2.0)
    value.add_argument("--perceptual-weight", type=float, default=0.08)
    value.add_argument("--perceptual-stride", type=int, default=2)
    value.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--grad-clip", type=float, default=1.0)
    value.add_argument("--checkpoint-interval", type=int, default=2000)
    value.add_argument("--keep-checkpoints", type=int, default=3)
    value.add_argument("--eval-interval", type=int, default=2000)
    value.add_argument("--eval-sequences", type=int, default=8)
    value.add_argument("--eval-lpips", action="store_true")
    value.add_argument("--log-interval", type=int, default=25)
    value.add_argument("--seed", type=int, default=20260825)
    value.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        train(args)
        return
    if not args.resume:
        raise SystemExit("eval requires --resume")
    device = torch.device(args.device)
    model, payload = load_checkpoint(args.resume, device)
    args._current_step = int(payload["step"])
    report = evaluate(model, args, device)
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
