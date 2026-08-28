#!/usr/bin/env python3
"""Train/evaluate a bottleneck-state V8 GOP continuity adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES
from aetv.decoder_context_adapter import V8DecoderContextAdapter
from experiment_gop_boundaries import (
    DEFAULT_CELLS,
    ChannelCell,
    SequenceCache,
    boundary_losses,
    cache_name,
    join_gops,
    load_model,
    sequence_metrics,
    simulate_transmission,
    split_gops,
)


def decode_received(model, source, mode, cell):
    separated = split_gops(source, mode.gop_frames)
    batch = source.shape[0]
    count = separated.shape[0] // batch
    z = model.encoder(separated)
    received, weights = [], []
    for item in z:
        if cell.snr_db is None and cell.fading in (None, "none"):
            received.append(item)
            weights.append(torch.ones_like(item))
        else:
            latent, weight, _ = simulate_transmission(
                item.float().cpu().numpy(),
                mode_name=mode.name,
                snr_db=cell.snr_db,
                fading_preset=None if cell.fading == "none" else cell.fading,
            )
            received.append(torch.from_numpy(latent).to(source.device))
            weights.append(torch.from_numpy(weight).to(source.device))
    latents = torch.stack(received).reshape(batch, count, -1)
    confidence = torch.stack(weights).reshape(batch, count, -1).mean(dim=-1).clamp(0, 1)
    weights = torch.stack(weights).reshape(batch, count, -1)
    return latents, weights, confidence


def train(args):
    mode = AETV_MODES[args.mode]
    device = torch.device(args.device)
    model = V8DecoderContextAdapter.from_v8_checkpoint(
        str(args.checkpoint), adapter_width=args.width, attention_dim=args.attention_dim,
        attention_heads=args.heads, adapter_blocks=args.blocks,
        freeze_base=not (args.finetune_decoder or args.finetune_encoder),
    ).to(device)
    if not args.finetune_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    dataset = SequenceCache(args.data_dir / cache_name(mode, args.gops, "train"))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
    iterator = iter(loader)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            source = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            source = next(iterator)
        source = source.to(device).float()
        with torch.no_grad():
            latents = model.encode_sequence(source)
        optimizer.zero_grad(set_to_none=True)
        recon, base, gates = model.decode_sequence(
            latents, return_base=True, return_gates=True
        )
        cross = boundary_losses(recon, source, mode.gop_frames)
        total = (
            args.source_weight * F.l1_loss(recon, source)
            + args.anchor_weight * F.l1_loss(recon, base)
            + args.boundary_weight * cross["boundary_rgb_delta"]
            + args.lowpass_weight * cross["boundary_lowpass_step"]
            + args.acceleration_weight * cross["boundary_acceleration"]
            + args.within_weight * cross["within_gop_temporal_error"]
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(total), "boundary": float(cross["boundary_rgb_delta"]), "gate": float(gates.mean())}), flush=True)
    torch.save({
        "kind": "aetv-v8-bottleneck-context-experiment",
        "base_checkpoint": str(args.checkpoint.resolve()),
        "model_config": model.config(),
        "adapter_state_dict": model.context_adapter.state_dict(),
        "model_state_dict": ({name: value.detach().cpu() for name, value in model.state_dict().items()} if args.finetune_decoder else None),
        "training": vars(args),
        "wire_contract_changed": False,
    }, args.out / "adapter.pt")


def evaluate(args):
    mode = AETV_MODES[args.mode]
    device = torch.device(args.device)
    payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    config = payload["model_config"]["adapter"]
    model = V8DecoderContextAdapter.from_v8_checkpoint(
        str(args.checkpoint), adapter_width=config["width"], attention_dim=config["attention_dim"],
        attention_heads=config["heads"], adapter_blocks=config["blocks"], freeze_base=True,
    ).to(device).eval()
    if payload.get("model_state_dict") is not None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    else:
        model.context_adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    dataset = SequenceCache(args.data_dir / cache_name(mode, args.gops, "eval"))
    rows = {cell.label: {"base": [], "context": []} for cell in DEFAULT_CELLS}
    with torch.inference_mode():
        for index in range(min(args.eval_sequences, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device).float()
            for cell in DEFAULT_CELLS:
                latents, weights, confidence = decode_received(model, source, mode, cell)
                context = model.decode_sequence(latents, weights)
                base = model.decode_sequence(latents, weights, use_adapter=False)
                for name, value in (("base", base), ("context", context)):
                    rows[cell.label][name].append(sequence_metrics(value, source, mode.gop_frames, device, include_lpips=True))
            print(f"evaluated {index + 1}/{min(args.eval_sequences, len(dataset))}", flush=True)
    report = {"cells": {}}
    for label, values in rows.items():
        report["cells"][label] = {}
        for metric in values["base"][0]:
            b = sum(row[metric] for row in values["base"]) / len(values["base"])
            c = sum(row[metric] for row in values["context"]) / len(values["context"])
            report["cells"][label][metric] = {"baseline_mean": b, "candidate_mean": c, "reduction_percent": 100 * (b - c) / max(abs(b), 1e-12)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("train", "eval"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    p.add_argument("--adapter", type=Path, default=Path("runs/v8-latent-context/adapter.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/v8-latent-context"))
    p.add_argument("--data-dir", type=Path, default=Path("runs/gop-boundary-data"))
    p.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    p.add_argument("--gops", type=int, default=3)
    p.add_argument("--eval-sequences", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--attention-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--source-weight", type=float, default=0.5)
    p.add_argument("--anchor-weight", type=float, default=2.0)
    p.add_argument("--boundary-weight", type=float, default=8.0)
    p.add_argument("--lowpass-weight", type=float, default=4.0)
    p.add_argument("--acceleration-weight", type=float, default=1.0)
    p.add_argument("--within-weight", type=float, default=1.0)
    p.add_argument("--finetune-decoder", action="store_true")
    p.add_argument("--finetune-encoder", action="store_true")
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.command == "train": train(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
