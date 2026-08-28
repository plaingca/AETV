#!/usr/bin/env python3
"""End-to-end zero-rate-change recurrent V8 multi-GOP training.

This is the promotion-candidate trainer requested after receiver-only methods
plateaued.  It warm-starts released V8, trains every current-GOP encoder
parameter, carries a causal decoder bottleneck through 5--16 GOPs, and jointly
trains the synthesis tail.  The differentiable channel is deliberately only a
curriculum; final evidence must come from exact continuous runtime evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aetv.multigop_channel import MultiGOPChannelCurriculum
from aetv.recurrent_feature_tcm import V8RecurrentFeatureTCM
from aetv.recurrent_joint_codec import V8RecurrentJointCodec
from aetv.transition_anchor_codec import V8TransitionAnchorCodec
from scripts.experiment_gop_boundaries import SequenceCache


class EventSequenceDataset(Dataset):
    """Return a real contiguous clip plus a deterministic cut donor."""

    def __init__(self, root: Path, max_gops: int):
        self.source = SequenceCache(root, max_frames=max_gops * 6)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        donor = (index * 37 + 11) % len(self.source)
        if donor == index:
            donor = (donor + 1) % len(self.source)
        return self.source[index], self.source[donor]


class IdentityMultiGOPChannel(torch.nn.Module):
    """Exact clean branch used to enforce the released quality envelope."""

    def forward(self, values: torch.Tensor, *, progress: float = 1.0):
        batch, count, _ = values.shape
        flags = torch.zeros(batch, count, dtype=torch.bool, device=values.device)
        return values, torch.ones_like(values), {
            "missing": flags,
            "fade": flags,
            "fade_end": flags,
            "snr_db": values.new_full((batch, count), float("inf")),
            "gain": values.new_ones((batch, count)),
        }


def select_unroll(
    primary: torch.Tensor,
    donor: torch.Tensor,
    *,
    step: int,
    steps: int,
    min_gops: int,
    max_gops: int,
    cut_probability: float,
    reset_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
    """Crop 5--16 GOPs and add supervised cuts/resets only at GOP boundaries."""
    progress = step / max(steps, 1)
    curriculum_max = min(max_gops, min_gops + int((max_gops - min_gops) * progress * 1.5))
    count = random.randint(min_gops, max(min_gops, curriculum_max))
    available = primary.shape[2] // 6
    start = random.randint(0, available - count)
    video = primary[:, :, start * 6 : (start + count) * 6].clone()
    donor_video = donor[:, :, start * 6 : (start + count) * 6]
    reset = torch.zeros(video.shape[0], count, dtype=torch.bool)
    cut = torch.zeros_like(reset)

    for item in range(video.shape[0]):
        if count > 2 and random.random() < cut_probability:
            index = random.randint(1, count - 1)
            video[item, :, index * 6 :] = donor_video[item, :, index * 6 :]
            reset[item, index] = True
            cut[item, index] = True
        if count > 2 and random.random() < reset_probability:
            reset[item, random.randint(1, count - 1)] = True
    return video, reset, {"gops": count, "cut": cut}


def rgb_to_ycbcr_delta(value: torch.Tensor) -> torch.Tensor:
    red, green, blue = value.unbind(dim=1)
    return torch.stack(
        (
            0.299 * red + 0.587 * green + 0.114 * blue,
            -0.168736 * red - 0.331264 * green + 0.5 * blue,
            0.5 * red - 0.418688 * green - 0.081312 * blue,
        ),
        dim=1,
    )


def gradient_error(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx = (recon[..., 1:] - recon[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (recon[..., 1:, :] - recon[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def lpips_distance(metric: torch.nn.Module, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    # LPIPS reference nets operate in float32 and expect [-1, 1].
    return metric(first.float() * 2 - 1, second.float() * 2 - 1).mean()


def loss_terms(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    events: dict[str, torch.Tensor],
    reset: torch.Tensor,
    perceptual: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    batch, _, frames, height, width = reconstruction.shape
    count = frames // 6
    boundary_indices = torch.arange(6, frames, 6, device=reconstruction.device)
    delta = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
    source_delta = target[:, :, 1:] - target[:, :, :-1]
    delta_error = delta - source_delta
    seam = boundary_indices - 1
    boundary_delta_error = delta_error.index_select(2, seam)

    lowpass = F.avg_pool2d(
        rgb_to_ycbcr_delta(boundary_delta_error)
        .permute(0, 2, 1, 3, 4)
        .reshape(-1, 3, height, width),
        9,
        stride=1,
        padding=4,
    )
    lowpass_y = lowpass[:, :1].abs().mean()
    lowpass_chroma = lowpass[:, 1:].abs().mean()

    acceleration_error = delta_error[:, :, 1:] - delta_error[:, :, :-1]
    acceleration_positions = torch.cat((boundary_indices - 2, boundary_indices - 1))
    acceleration = acceleration_error.index_select(2, acceleration_positions).abs().mean()

    within_mask = torch.ones(frames - 1, dtype=torch.bool, device=reconstruction.device)
    within_mask[seam] = False
    within_delta = delta_error[:, :, within_mask].abs().mean()

    recon_boundary_delta = delta.index_select(2, seam).permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
    source_boundary_delta = source_delta.index_select(2, seam).permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
    boundary_lpips = lpips_distance(
        perceptual,
        (0.5 + 0.5 * recon_boundary_delta).clamp(0, 1),
        (0.5 + 0.5 * source_boundary_delta).clamp(0, 1),
    )

    sample_frames = torch.linspace(0, frames - 1, min(4, frames), device=reconstruction.device).round().long()
    spatial_recon = reconstruction.index_select(2, sample_frames).permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
    spatial_target = target.index_select(2, sample_frames).permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
    spatial_lpips = lpips_distance(perceptual, spatial_recon, spatial_target)

    low_recon = F.avg_pool3d(reconstruction, (1, 5, 5), stride=1, padding=(0, 2, 2))
    low_target = F.avg_pool3d(target, (1, 5, 5), stride=1, padding=(0, 2, 2))
    spatial_highpass = (reconstruction - low_recon - (target - low_target)).abs().mean()

    # Source-referenced error energy at one cycle/GOP (1 Hz at 6 fps).  The
    # spatial grid is retained before the temporal FFT so local flicker cannot
    # disappear through global averaging.
    luminance_error = rgb_to_ycbcr_delta(reconstruction - target)[:, :1]
    luminance_error = F.avg_pool3d(luminance_error, (1, 8, 8), stride=(1, 8, 8))
    spectrum = torch.fft.rfft(luminance_error.float(), dim=2)
    signature_bin = max(1, min(spectrum.shape[2] - 1, round(frames / 6)))
    gop_signature = spectrum[:, :, signature_bin].abs().mean() / math.sqrt(frames)

    recovery = events["fade_end"] | reset.to(events["fade_end"].device)
    # The first good GOP after a fade/reset must already be back at baseline.
    recovery_frames = recovery.repeat_interleave(6, dim=1)
    recovery_l1 = (
        (reconstruction - target).abs()
        * recovery_frames[:, None, :, None, None]
    ).sum() / (recovery_frames.sum().clamp_min(1) * 3 * height * width)

    missing_frames = events["missing"].repeat_interleave(6, dim=1)
    observed_frames = ~missing_frames
    spatial_l1 = (
        (reconstruction - target).abs() * observed_frames[:, None, :, None, None]
    ).sum() / (observed_frames.sum().clamp_min(1) * 3 * height * width)
    concealment_l1 = (
        (reconstruction - target).abs() * missing_frames[:, None, :, None, None]
    ).sum() / (missing_frames.sum().clamp_min(1) * 3 * height * width)

    return {
        "spatial_l1": spatial_l1,
        "spatial_mse": F.mse_loss(reconstruction, target),
        "spatial_lpips": spatial_lpips,
        "detail": gradient_error(reconstruction, target),
        "spatial_highpass": spatial_highpass,
        "within_delta": within_delta,
        "boundary_delta": boundary_delta_error.abs().mean(),
        "boundary_lowpass_y": lowpass_y,
        "boundary_lowpass_chroma": lowpass_chroma,
        "boundary_acceleration": acceleration,
        "boundary_delta_lpips": boundary_lpips,
        "gop_signature_1hz": gop_signature,
        "recovery_l1": recovery_l1,
        "concealment_l1": concealment_l1,
    }


QUALITY_WEIGHTS = {
    "spatial_l1": 2.0,
    "spatial_mse": 0.5,
    "spatial_lpips": 1.2,
    "detail": 6.0,
    "spatial_highpass": 4.0,
    "within_delta": 2.5,
    "boundary_delta": 3.5,
    "boundary_lowpass_y": 2.5,
    "boundary_lowpass_chroma": 1.0,
    "boundary_acceleration": 1.5,
    "boundary_delta_lpips": 0.4,
    "gop_signature_1hz": 0.5,
    "recovery_l1": 1.0,
    "concealment_l1": 0.0,
}


CHANNEL_WEIGHTS = {
    "spatial_l1": 1.5,
    "spatial_mse": 0.35,
    "spatial_lpips": 0.8,
    "detail": 4.0,
    "spatial_highpass": 3.0,
    "within_delta": 2.5,
    "boundary_delta": 6.0,
    "boundary_lowpass_y": 4.0,
    "boundary_lowpass_chroma": 1.5,
    "boundary_acceleration": 2.0,
    "boundary_delta_lpips": 0.5,
    "gop_signature_1hz": 0.8,
    "recovery_l1": 2.0,
    "concealment_l1": 0.15,
}


def released_anchor_terms(
    reconstruction: torch.Tensor,
    released: torch.Tensor,
    perceptual: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    frames = reconstruction.shape[2]
    selected = torch.linspace(0, frames - 1, min(4, frames), device=reconstruction.device).round().long()
    candidate_frames = reconstruction.index_select(2, selected).permute(0, 2, 1, 3, 4).flatten(0, 1)
    released_frames = released.index_select(2, selected).permute(0, 2, 1, 3, 4).flatten(0, 1)
    low_recon = F.avg_pool3d(reconstruction, (1, 5, 5), stride=1, padding=(0, 2, 2))
    low_released = F.avg_pool3d(released, (1, 5, 5), stride=1, padding=(0, 2, 2))
    return {
        "released_l1": F.l1_loss(reconstruction, released),
        "released_lpips": lpips_distance(perceptual, candidate_frames, released_frames),
        "released_detail": gradient_error(reconstruction, released),
        "released_highpass": (reconstruction - low_recon - (released - low_released)).abs().mean(),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    trainable_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": model.checkpoint_kind,
            "base_checkpoint": str(Path(args.checkpoint).resolve()),
            "step": step,
            "args": vars(args),
            "trainable_counts": trainable_counts,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "wire_contract": {"values_per_gop": 2816, "frames_per_gop": 6},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_16gop_joint_train"))
    parser.add_argument("--checkpoint", default="models/v8-hf3k-face-gan.pt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--min-gops", type=int, default=5)
    parser.add_argument("--max-gops", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=1.5e-6)
    parser.add_argument("--encoder-context-lr", type=float, default=1.5e-4)
    parser.add_argument("--tail-lr", type=float, default=4e-7)
    parser.add_argument("--state-lr", type=float, default=4e-5)
    parser.add_argument("--anchor-values", type=int, choices=(0, 32, 64), default=0)
    parser.add_argument("--anchor-lr", type=float, default=2e-4)
    parser.add_argument("--cut-probability", type=float, default=0.25)
    parser.add_argument("--reset-probability", type=float, default=0.12)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--feature-tcm",
        action="store_true",
        help="Freeze the released encoder and train RAFT-aligned full-res feature TCM",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = Path(
            "runs/v8-recurrent-feature-tcm"
            if args.feature_tcm
            else "runs/v8-recurrent-joint-16gop-txcontext"
        )
    if not (5 <= args.min_gops <= args.max_gops <= 16):
        parser.error("unroll contract requires 5 <= min-gops <= max-gops <= 16")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.anchor_values:
        model = V8TransitionAnchorCodec.from_released(
            args.checkpoint, anchor_values=args.anchor_values
        ).to(device)
    elif args.feature_tcm:
        model = V8RecurrentFeatureTCM.from_released(args.checkpoint).to(device)
        model.aligner.eval()
    else:
        model = V8RecurrentJointCodec.from_released(args.checkpoint).to(device)
    counts = model.set_trainable_contract()
    channel = MultiGOPChannelCurriculum().to(device)
    clean_channel = IdentityMultiGOPChannel().to(device)
    released = V8RecurrentJointCodec.from_released(args.checkpoint).to(device).eval()
    for parameter in released.parameters():
        parameter.requires_grad_(False)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    state_module = getattr(model, "fuser", None) or model.context_adapter
    state_parameters = [p for p in state_module.parameters() if p.requires_grad]
    encoder_context_parameters = []
    encoder_context = getattr(model, "encoder_context", None)
    if encoder_context is not None:
        encoder_context_parameters = [
            p for p in encoder_context.parameters() if p.requires_grad
        ]
    anchor_parameters = []
    if isinstance(model, V8TransitionAnchorCodec):
        anchor_parameters = [
            p for module in (model.anchor_encoder, model.anchor_decoder)
            for p in module.parameters() if p.requires_grad
        ]
    used = {
        id(p)
        for p in encoder_parameters
        + state_parameters
        + encoder_context_parameters
        + anchor_parameters
    }
    tail_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
    optimizer = torch.optim.AdamW(
        [
            group
            for group in (
                {"params": encoder_parameters, "lr": args.encoder_lr},
                {"params": encoder_context_parameters, "lr": args.encoder_context_lr},
                {"params": state_parameters, "lr": args.state_lr},
                {"params": tail_parameters, "lr": args.tail_lr},
                {"params": anchor_parameters, "lr": args.anchor_lr},
            )
            if group["params"]
        ],
        weight_decay=1e-5,
    )
    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["step"])

    dataset = EventSequenceDataset(args.data, args.max_gops)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = iter(loader)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "contract.json").write_text(json.dumps({
        "trainable_counts": counts,
        "quality_loss_weights": QUALITY_WEIGHTS,
        "channel_loss_weights": CHANNEL_WEIGHTS,
        "released_anchor_weights": {"l1": 2.5, "lpips": 0.8, "detail": 4.0, "highpass": 3.0},
        "alternating_clean_channel": True,
        "wire_values_per_gop": 2816,
        "unroll_gops": [args.min_gops, args.max_gops],
        "feature_tcm": bool(args.feature_tcm),
        "frozen_encoder": bool(args.feature_tcm),
        "raft_aligned_fullres_features": bool(args.feature_tcm),
        "source": str(args.data.resolve()),
    }, indent=2) + "\n")
    print(json.dumps({"event": "start", "trainable": counts, "clips": len(dataset)}), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    interval_start = time.perf_counter()
    interval_gops = 0
    for step in range(start_step + 1, args.steps + 1):
        try:
            primary, donor = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            primary, donor = next(iterator)
        primary = primary.to(device, non_blocking=True)
        donor = donor.to(device, non_blocking=True)
        video, reset, augmentation = select_unroll(
            primary,
            donor,
            step=step,
            steps=args.steps,
            min_gops=args.min_gops,
            max_gops=args.max_gops,
            cut_probability=args.cut_probability,
            reset_probability=args.reset_probability,
        )
        reset = reset.to(device)
        quality_step = step % 2 == 1
        active_channel = clean_channel if quality_step else channel
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reconstruction, events = model(
                video,
                active_channel,
                reset=reset,
                progress=step / args.steps,
                use_checkpoint=True,
            )
        terms = loss_terms(reconstruction.float(), video.float(), events, reset, perceptual)
        weights_for_step = QUALITY_WEIGHTS if quality_step else CHANNEL_WEIGHTS
        loss = sum(weights_for_step[name] * value for name, value in terms.items())
        anchor = {}
        if quality_step:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                released_video, _ = released(
                    video, clean_channel, reset=reset, progress=0.0, use_checkpoint=False
                )
            anchor = released_anchor_terms(
                reconstruction.float(), released_video.float(), perceptual
            )
            loss = (
                loss
                + 2.5 * anchor["released_l1"]
                + 0.8 * anchor["released_lpips"]
                + 4.0 * anchor["released_detail"]
                + 3.0 * anchor["released_highpass"]
            )
        loss = loss / args.accum
        loss.backward()
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        interval_gops += int(augmentation["gops"]) * args.batch
        if step == 1 or step % args.log_interval == 0:
            elapsed = time.perf_counter() - interval_start
            record = {
                "step": step,
                "loss": float(loss.detach() * args.accum),
                "gops": int(augmentation["gops"]),
                "gops_per_second": interval_gops / max(elapsed, 1e-9),
                "missing_fraction": float(events["missing"].float().mean()),
                "fade_fraction": float(events["fade"].float().mean()),
                "cut_count": int(augmentation["cut"].sum()),
                "branch": "clean_anchor" if quality_step else "channel",
                **{name: float(value.detach()) for name, value in terms.items()},
                **{name: float(value.detach()) for name, value in anchor.items()},
            }
            print(json.dumps(record), flush=True)
            with (args.out / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            interval_start = time.perf_counter()
            interval_gops = 0
        if step % args.checkpoint_interval == 0 or step == args.steps:
            save_checkpoint(args.out / f"checkpoint_step_{step:06d}.pt", model, optimizer, step, args, counts)
            save_checkpoint(args.out / "candidate.pt", model, optimizer, step, args, counts)


if __name__ == "__main__":
    main()
