#!/usr/bin/env python3
"""Short gated training run for the V8-preserving overlap adapter."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES  # noqa: E402
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from aetv.overlap_adapter import V8OverlapAdapter  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    SequenceCache,
    boundary_losses,
    cache_name,
    sequence_metrics,
)
from eval import write_labeled_grid_mp4  # noqa: E402
from render_overlap_checkpoint import extract_video  # noqa: E402
from train import (  # noqa: E402
    dwt3d_loss,
    spatial_gradient_loss,
    temporal_acceleration_loss,
    temporal_cosine_loss,
    temporal_delta_loss,
    temporal_energy_loss,
)
from train_gop_overlap import streaming_loader  # noqa: E402


LOSS_WEIGHTS = {
    "mse": 0.25,
    "l1": 0.8,
    "dwt": 3.0,
    "gradient": 1.5,
    "temporal": 1.5,
    "acceleration": 0.3,
    "temporal_energy": 2.0,
    "temporal_cosine": 0.2,
    "perceptual": 0.18,
    "boundary": 1.0,
    "boundary_lowpass": 0.5,
    "boundary_acceleration": 0.25,
    "teacher_anchor": 1.0,
}


def losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    perceptual: MultiLayerVGGPerceptualLoss,
    frames_per_gop: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    boundary = boundary_losses(recon, target, frames_per_gop)
    values = {
        "mse": F.mse_loss(recon, target),
        "l1": F.l1_loss(recon, target),
        "dwt": dwt3d_loss(recon, target, levels=3),
        "gradient": spatial_gradient_loss(recon, target),
        "temporal": temporal_delta_loss(recon, target),
        "acceleration": temporal_acceleration_loss(recon, target),
        "temporal_energy": temporal_energy_loss(recon, target),
        "temporal_cosine": temporal_cosine_loss(recon, target),
        "perceptual": perceptual(recon, target),
        "boundary": boundary["boundary_delta"],
        "boundary_lowpass": boundary["boundary_lowpass_step"],
        "boundary_acceleration": boundary["boundary_acceleration"],
        "teacher_anchor": F.l1_loss(recon, teacher),
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in values.items())
    return total, values


def between_call_delta(
    recon: torch.Tensor, target: torch.Tensor, frames_per_gop: int, emit_gops: int
) -> float:
    index = emit_gops * frames_per_gop
    recon_delta = recon[:, :, index] - recon[:, :, index - 1]
    target_delta = target[:, :, index] - target[:, :, index - 1]
    return float(F.l1_loss(recon_delta, target_delta))


def evaluate_cache(
    model: V8OverlapAdapter,
    cache: Path,
    device: torch.device,
    limit: int,
    *,
    include_lpips: bool,
) -> dict:
    dataset = SequenceCache(cache)
    rows = {"base": [], "adapter": []}
    model.eval()
    with torch.inference_mode():
        for index in range(min(limit, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device).float()
            if source.max() > 1:
                source.div_(255)
            latents = model.encode_sequence(source)
            target = model.target_for_sequence(source)
            for name, enabled in (("base", False), ("adapter", True)):
                recon = model.decode_sequence(latents, use_adapter=enabled)
                row = sequence_metrics(
                    recon,
                    target,
                    model.mode.gop_frames,
                    device,
                    include_lpips=include_lpips,
                )
                row["between_call_delta"] = between_call_delta(
                    recon, target, model.mode.gop_frames, model.emit_gops
                )
                rows[name].append(row)
    return {
        name: {
            key: sum(row[key] for row in values) / len(values)
            for key in values[0]
        }
        for name, values in rows.items()
    }


def simpsons_gate(
    model: V8OverlapAdapter,
    source_path: Path,
    out: Path,
    step: int,
    device: torch.device,
) -> dict:
    mode = model.mode
    gops = 14
    raw = extract_video(
        source_path,
        gops * mode.gop_frames,
        int(mode.fps),
        mode.width,
        mode.height,
    )
    source = torch.from_numpy(raw).permute(0, 3, 1, 2).unsqueeze(0)
    source = source.permute(0, 2, 1, 3, 4).to(device).float().div_(255)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        latents = model.encode_sequence(source)
        target = model.target_for_sequence(source)
        base = model.decode_sequence(latents, use_adapter=False)
        adapted = model.decode_sequence(latents, use_adapter=True)
    path = out / f"simpsons_step_{step:06d}_clean.mp4"
    write_labeled_grid_mp4(
        [("SOURCE", target), ("RELEASED V8", base), (f"ADAPTER {step}", adapted)],
        path,
        mode.fps,
        columns=3,
    )
    report = {"step": step, "video": str(path)}
    for name, recon in (("base", base), ("adapter", adapted)):
        mse = float(F.mse_loss(recon.float(), target.float()))
        report[name] = {
            "psnr": -10 * math.log10(max(mse, 1e-12)),
            "mse": mse,
            "between_call_delta": between_call_delta(
                recon.float(), target.float(), mode.gop_frames, model.emit_gops
            ),
        }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def save_checkpoint(
    path: Path,
    model: V8OverlapAdapter,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "kind": model.checkpoint_kind,
            "step": step,
            "base_checkpoint": str(args.base_checkpoint),
            "model_config": model.config(),
            "adapter_state_dict": model.adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def run_gate(
    model: V8OverlapAdapter,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
) -> dict:
    fixed = evaluate_cache(
        model,
        args.eval_cache,
        device,
        args.eval_sequences,
        include_lpips=step == args.steps,
    )
    simpsons = simpsons_gate(model, args.simpsons, args.out / "renders", step, device)
    report = {"step": step, "fixed": fixed, "simpsons": simpsons}
    (args.out / f"gate_step_{step:06d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    parser.add_argument("--out", type=Path, default=Path("/pool0/AETV-runs/v8-overlap-adapter-gate"))
    parser.add_argument("--eval-cache", type=Path, default=Path("/pool0/AETV-runs/gop-boundary-data/v8_192x108_8gop_eval"))
    parser.add_argument("--simpsons", type=Path, default=Path("/home/plaing/SSTVAE/The Simpsons Season 31 Episode 20 - The Simpsons Full NoCuts-iex52uxH460.mp4"))
    parser.add_argument("--hf-dataset", default="lance-format/Openvid-1M")
    parser.add_argument("--train-gops", type=int, default=8)
    parser.add_argument("--window-gops", type=int, default=5)
    parser.add_argument("--emit-gops", type=int, default=3)
    parser.add_argument("--adapter-width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-sequences", type=int, default=8)
    parser.add_argument("--gate-steps", default="0,100,500")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "renders").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = V8OverlapAdapter.from_v8_checkpoint(
        str(args.base_checkpoint),
        window_gops=args.window_gops,
        emit_gops=args.emit_gops,
        adapter_width=args.adapter_width,
        freeze_base=True,
    ).to(device)
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    perceptual = MultiLayerVGGPerceptualLoss().to(device).eval()
    writer = SummaryWriter(args.out / "tensorboard")
    gate_steps = {int(value) for value in args.gate_steps.split(",")}
    print(
        json.dumps(
            {
                "model": model.config(),
                "parameters_total": sum(value.numel() for value in model.parameters()),
                "parameters_trainable": sum(value.numel() for value in trainable),
                "loss_weights": LOSS_WEIGHTS,
            },
            indent=2,
        ),
        flush=True,
    )
    if 0 in gate_steps:
        run_gate(model, args, 0, device)

    batches = streaming_loader(args, AETV_MODES["V8"], device)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        source = next(batches)
        if source.max() > 1:
            source = source.float().div_(255)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            latents = model.encode_sequence(source)
            teacher = model.decode_sequence(latents, use_adapter=False)
            target = model.target_for_sequence(source)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            recon = model.decode_sequence(latents, use_adapter=True)
            total, values = losses(
                recon, target, teacher, perceptual, model.mode.gop_frames
            )
        (total / args.accum).backward()
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        for name, value in values.items():
            writer.add_scalar(f"train/{name}", float(value.detach()), step)
        writer.add_scalar("train/total", float(total.detach()), step)
        if step == 1:
            contribution = {
                name: float(value.detach()) * LOSS_WEIGHTS[name]
                for name, value in values.items()
            }
            print(json.dumps({"step_1_contributions": contribution}, indent=2), flush=True)
        if step == 1 or step % args.log_interval == 0:
            print(
                f"step {step:>4}/{args.steps} total={float(total.detach()):.5f} "
                f"l1={float(values['l1'].detach()):.5f} "
                f"boundary={float(values['boundary'].detach()):.5f} "
                f"anchor={float(values['teacher_anchor'].detach()):.5f} "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True,
            )
        if step in gate_steps:
            save_checkpoint(args.out / f"checkpoint_step_{step:06d}.pt", model, optimizer, step, args)
            run_gate(model, args, step, device)
    writer.close()


if __name__ == "__main__":
    main()
