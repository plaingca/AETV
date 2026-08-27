#!/usr/bin/env python3
"""Gated clean training for the internal V8 decoder context adapter."""

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
from aetv.decoder_context_adapter import V8DecoderContextAdapter  # noqa: E402
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    SequenceCache,
    boundary_losses,
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
    "cut_gate": 0.02,
}


def make_context_sources(
    batch: int,
    gops: int,
    device: torch.device,
    mismatch_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return previous-feature indices plus synthetic scene-cut labels."""
    rows = torch.arange(batch, device=device)[:, None]
    previous_gops = torch.arange(gops - 1, device=device)[None, :]
    indices = rows * gops + previous_gops
    mismatch = torch.rand(batch, gops - 1, device=device) < mismatch_probability
    if mismatch_probability <= 0:
        return indices, mismatch
    if batch > 1:
        shift = torch.randint(1, batch, (batch, gops - 1), device=device)
        other_rows = (rows + shift) % batch
        other_gops = torch.randint(0, gops, (batch, gops - 1), device=device)
        replacements = other_rows * gops + other_gops
    else:
        # Deterministic non-adjacent fallback for a one-item diagnostic batch.
        replacements = rows * gops + (previous_gops + max(2, gops // 2)) % gops
    return torch.where(mismatch, replacements, indices), mismatch


def balanced_gate_loss(gates: torch.Tensor, mismatch: torch.Tensor) -> torch.Tensor:
    gates = gates.float().clamp(1e-5, 1 - 1e-5)
    pieces = []
    if (~mismatch).any():
        pieces.append(-torch.log(gates[~mismatch]).mean())
    if mismatch.any():
        pieces.append(-torch.log1p(-gates[mismatch]).mean())
    return torch.stack(pieces).mean()


def losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    gates: torch.Tensor,
    mismatch: torch.Tensor,
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
        "cut_gate": balanced_gate_loss(gates, mismatch),
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in values.items())
    return total, values


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def evaluate_cache(
    model: V8DecoderContextAdapter,
    cache: Path,
    device: torch.device,
    limit: int,
    *,
    include_lpips: bool,
) -> dict:
    dataset = SequenceCache(cache)
    rows: dict[str, list[dict[str, float]]] = {
        "base": [],
        "context": [],
        "mismatched_context": [],
    }
    gate_rows = {"continuous": [], "mismatched": []}
    usage_rows = {
        "context_delta_from_base": [],
        "mismatch_delta_from_context": [],
        "mismatch_delta_from_base": [],
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, min(limit, len(dataset)), 2):
            indexes = [start, (start + 1) % len(dataset)]
            source = torch.stack([dataset[index] for index in indexes]).to(device).float()
            latents = model.encode_sequence(source)
            batch, gops = latents.shape[:2]
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                context, base, natural_gates = model.decode_sequence(
                    latents, return_base=True, return_gates=True
                )
                cut_sources, _ = make_context_sources(batch, gops, device, 1.0)
                mismatched, cut_gates = model.decode_sequence(
                    latents,
                    context_source_indices=cut_sources,
                    return_gates=True,
                )
            for name, recon in (
                ("base", base),
                ("context", context),
                ("mismatched_context", mismatched),
            ):
                rows[name].append(
                    sequence_metrics(
                        recon,
                        source,
                        model.mode.gop_frames,
                        device,
                        include_lpips=include_lpips,
                    )
                )
            gate_rows["continuous"].append(float(natural_gates.float().mean()))
            gate_rows["mismatched"].append(float(cut_gates.float().mean()))
            usage_rows["context_delta_from_base"].append(
                float(F.l1_loss(context.float(), base.float()))
            )
            usage_rows["mismatch_delta_from_context"].append(
                float(F.l1_loss(mismatched.float(), context.float()))
            )
            usage_rows["mismatch_delta_from_base"].append(
                float(F.l1_loss(mismatched.float(), base.float()))
            )
    report = {name: aggregate(values) for name, values in rows.items()}
    report["scene_gate"] = {
        name: sum(values) / len(values) for name, values in gate_rows.items()
    }
    report["context_usage"] = {
        name: sum(values) / len(values) for name, values in usage_rows.items()
    }
    report["context_usage"]["sensitivity_ratio"] = (
        report["context_usage"]["mismatch_delta_from_context"]
        / max(report["context_usage"]["context_delta_from_base"], 1e-12)
    )
    return report


def simpsons_gate(
    model: V8DecoderContextAdapter,
    source_path: Path,
    out: Path,
    step: int,
    device: torch.device,
    gops: int,
) -> dict:
    mode = model.mode
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
        context, base, gates = model.decode_sequence(
            latents, return_base=True, return_gates=True
        )
    seconds = gops * mode.gop_frames / mode.fps
    suffix = "60s" if 59.5 <= seconds <= 60.5 else f"{seconds:g}s"
    path = out / f"simpsons_step_{step:06d}_{suffix}_clean.mp4"
    write_labeled_grid_mp4(
        [
            ("SOURCE", source),
            ("RELEASED V8", base),
            (f"DECODER CONTEXT {step}", context),
        ],
        path,
        mode.fps,
        columns=3,
    )
    report = {
        "step": step,
        "gops": gops,
        "seconds": seconds,
        "video": str(path),
        "scene_gate_mean": float(gates.float().mean()),
    }
    for name, recon in (("base", base), ("context", context)):
        report[name] = sequence_metrics(
            recon.float(),
            source.float(),
            mode.gop_frames,
            device,
            include_lpips=False,
        )
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def save_checkpoint(
    path: Path,
    model: V8DecoderContextAdapter,
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
            "adapter_state_dict": model.context_adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def run_gate(
    model: V8DecoderContextAdapter,
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
    render_gops = (
        args.final_simpsons_gops if step == args.steps else args.simpsons_gops
    )
    simpsons = simpsons_gate(
        model,
        args.simpsons,
        args.out / "renders",
        step,
        device,
        render_gops,
    )
    report = {"step": step, "fixed": fixed, "simpsons": simpsons}
    (args.out / f"gate_step_{step:06d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/pool0/AETV-runs/v8-decoder-context-adapter-gate-v2"),
    )
    parser.add_argument(
        "--eval-cache",
        type=Path,
        default=Path(
            "/pool0/AETV-runs/gop-boundary-data/v8_192x108_8gop_eval"
        ),
    )
    parser.add_argument(
        "--simpsons",
        type=Path,
        default=Path(
            "/home/plaing/SSTVAE/The Simpsons Season 31 Episode 20 - "
            "The Simpsons Full NoCuts-iex52uxH460.mp4"
        ),
    )
    parser.add_argument("--hf-dataset", default="lance-format/Openvid-1M")
    parser.add_argument("--train-gops", type=int, default=8)
    parser.add_argument("--adapter-width", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--adapter-blocks", type=int, default=3)
    parser.add_argument("--mismatch-probability", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-sequences", type=int, default=8)
    parser.add_argument("--gate-steps", default="0,100,500")
    parser.add_argument("--simpsons-gops", type=int, default=14)
    parser.add_argument("--final-simpsons-gops", type=int, default=60)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "renders").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = V8DecoderContextAdapter.from_v8_checkpoint(
        str(args.base_checkpoint),
        adapter_width=args.adapter_width,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        adapter_blocks=args.adapter_blocks,
        freeze_base=True,
    ).to(device)
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )
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
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            latents = model.encode_sequence(source)
        batch, gops = latents.shape[:2]
        context_sources, mismatch = make_context_sources(
            batch, gops, device, args.mismatch_probability
        )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            recon, teacher, gates = model.decode_sequence(
                latents,
                context_source_indices=context_sources,
                return_base=True,
                return_gates=True,
            )
            total, values = losses(
                recon,
                source,
                teacher,
                gates,
                mismatch,
                perceptual,
                model.mode.gop_frames,
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
        writer.add_scalar(
            "train/gate_continuous",
            float(gates[~mismatch].detach().mean()) if (~mismatch).any() else math.nan,
            step,
        )
        writer.add_scalar(
            "train/gate_mismatch",
            float(gates[mismatch].detach().mean()) if mismatch.any() else math.nan,
            step,
        )
        if step == 1:
            contribution = {
                name: float(value.detach()) * LOSS_WEIGHTS[name]
                for name, value in values.items()
            }
            print(json.dumps({"step_1_contributions": contribution}, indent=2), flush=True)
        if step == 1 or step % args.log_interval == 0:
            continuous_gate = (
                float(gates[~mismatch].detach().mean())
                if (~mismatch).any()
                else math.nan
            )
            mismatch_gate = (
                float(gates[mismatch].detach().mean())
                if mismatch.any()
                else math.nan
            )
            print(
                f"step {step:>4}/{args.steps} total={float(total.detach()):.5f} "
                f"l1={float(values['l1'].detach()):.5f} "
                f"boundary={float(values['boundary'].detach()):.5f} "
                f"anchor={float(values['teacher_anchor'].detach()):.5f} "
                f"gate={continuous_gate:.3f}/{mismatch_gate:.3f} "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True,
            )
        if step in gate_steps:
            save_checkpoint(
                args.out / f"checkpoint_step_{step:06d}.pt",
                model,
                optimizer,
                step,
                args,
            )
            run_gate(model, args, step, device)
    writer.close()


if __name__ == "__main__":
    main()
