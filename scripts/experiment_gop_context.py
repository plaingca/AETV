#!/usr/bin/env python3
"""Train and evaluate a resettable stateful GOP correction adapter.

Unlike ``experiment_gop_boundaries.py`` this experiment does not modify the
released AETV encoder or decoder.  It decodes every GOP exactly as before and
then applies a small, confidence-gated residual using the previous decoded GOP
as context.  The first GOP after reset is therefore bit-exact with the released
model, and a receiver can bypass the adapter after loss or late entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES, AETVModeSpec  # noqa: E402
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
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
from eval import write_labeled_grid_mp4  # noqa: E402


class ContextResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        return value + self.conv2(F.silu(self.norm2(hidden)))


class StatefulGOPCorrector(nn.Module):
    """Low-resolution learned correction with an exact reset/bypass path."""

    def __init__(
        self,
        width: int = 24,
        blocks: int = 3,
        spatial_scale: int = 4,
        max_residual: float = 0.12,
        context_mode: str = "last",
        taper_floor: float = 0.0,
    ):
        super().__init__()
        if context_mode not in {"last", "full"}:
            raise ValueError(f"unknown context mode {context_mode!r}")
        if not 0.0 <= taper_floor <= 1.0:
            raise ValueError("taper_floor must be between zero and one")
        self.width = width
        self.blocks = blocks
        self.spatial_scale = spatial_scale
        self.max_residual = max_residual
        self.context_mode = context_mode
        self.taper_floor = taper_floor
        # The full-context mode adds the previous GOP at matching temporal
        # positions, exposing its motion trajectory as well as the adjacent
        # final frame. The compact legacy mode keeps only the latter.
        input_channels = 9 if context_mode == "last" else 12
        self.input = nn.Conv3d(input_channels, width, 3, padding=1)
        self.body = nn.Sequential(*(ContextResidualBlock(width) for _ in range(blocks)))
        self.output = nn.Conv3d(width, 3, 3, padding=1)
        # A fresh adapter must be an exact no-op, which makes checkpoint and
        # cold-start behavior directly testable.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def config(self) -> dict:
        return {
            "width": self.width,
            "blocks": self.blocks,
            "spatial_scale": self.spatial_scale,
            "max_residual": self.max_residual,
            "context_mode": self.context_mode,
            "taper_floor": self.taper_floor,
        }

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor | None,
        confidence: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Correct one BCTHW GOP, or return it exactly when state is absent."""
        if previous is None:
            return current
        if current.ndim != 5 or previous.ndim != 5:
            raise ValueError("current and previous GOPs must be BCTHW tensors")
        if current.shape[0] != previous.shape[0] or current.shape[1] != previous.shape[1]:
            raise ValueError("current and previous GOP batch/channel shapes must match")
        batch, _, frames, height, width = current.shape
        previous_frame = previous[:, :, -1:].expand(-1, -1, frames, -1, -1)
        features = [current, previous_frame, current - previous_frame]
        if self.context_mode == "full":
            if previous.shape[2] != frames:
                previous_full = F.interpolate(
                    previous,
                    size=(frames, height, width),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                previous_full = previous
            features.append(previous_full)
        features = torch.cat(features, dim=1)
        low_height = max(1, math.ceil(height / self.spatial_scale))
        low_width = max(1, math.ceil(width / self.spatial_scale))
        features = F.interpolate(
            features,
            size=(frames, low_height, low_width),
            mode="trilinear",
            align_corners=False,
        )
        residual = self.output(self.body(F.silu(self.input(features))))
        residual = F.interpolate(
            residual,
            size=(frames, height, width),
            mode="trilinear",
            align_corners=False,
        )
        # Reach zero at the final frame. This confines correction to the GOP
        # entrance and prevents a hidden appearance offset from accumulating.
        taper = torch.linspace(
            1.0,
            self.taper_floor,
            frames,
            device=current.device,
            dtype=current.dtype,
        )
        taper = taper.view(1, 1, frames, 1, 1)
        if confidence is None:
            gate = current.new_ones((batch, 1, 1, 1, 1))
        else:
            gate = torch.as_tensor(confidence, device=current.device, dtype=current.dtype)
            if gate.ndim == 0:
                gate = gate.expand(batch)
            gate = gate.reshape(batch, 1, 1, 1, 1).clamp(0, 1)
        correction = self.max_residual * torch.tanh(residual) * taper * gate
        return (current + correction).clamp(0, 1)


def adapter_checkpoint(adapter: StatefulGOPCorrector, source: Path, args: argparse.Namespace) -> dict:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "kind": "aetv-stateful-gop-corrector",
        "mode": args.mode,
        "source_checkpoint": str(source.resolve()),
        "source_sha256": digest,
        "adapter_config": adapter.config(),
        "adapter_state_dict": adapter.state_dict(),
        "training": {
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
            "reset_probability": args.reset_probability,
            "loss_weights": {
                "source": args.source_weight,
                "anchor": args.anchor_weight,
                "boundary": args.boundary_weight,
                "lowpass": args.lowpass_weight,
                "acceleration": args.acceleration_weight,
                "within": args.within_weight,
                "vgg_source": args.vgg_source_weight,
                "vgg_anchor": args.vgg_anchor_weight,
            },
        },
        "wire_contract_changed": False,
        "base_codec_modified": False,
        "reset_is_exact_bypass": True,
    }


def load_adapter(path: Path, device: torch.device) -> tuple[StatefulGOPCorrector, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "aetv-stateful-gop-corrector":
        raise ValueError(f"{path} is not a GOP corrector checkpoint")
    adapter = StatefulGOPCorrector(**payload["adapter_config"]).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return adapter, payload


def apply_adapter_sequence(
    adapter: StatefulGOPCorrector,
    base_gops: torch.Tensor,
    *,
    reset_probability: float = 0.0,
    generator: random.Random | None = None,
    confidences: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply state sequentially to (B,G,C,T,H,W) decoded GOPs."""
    if base_gops.ndim != 6:
        raise ValueError(f"expected BGCTHW base GOPs, got {tuple(base_gops.shape)}")
    batch, count = base_gops.shape[:2]
    outputs = [base_gops[:, 0]]
    for index in range(1, count):
        # Context is only as trustworthy as the weaker of the previous state
        # and current payload. A clean current GOP must not inherit a badly
        # faded predecessor merely because its own confidence is high.
        confidence = (
            None
            if confidences is None
            else torch.minimum(confidences[:, index - 1], confidences[:, index])
        )
        corrected = adapter(base_gops[:, index], outputs[-1], confidence)
        if reset_probability > 0:
            rng = generator or random
            reset = torch.tensor(
                [rng.random() < reset_probability for _ in range(batch)],
                device=base_gops.device,
                dtype=torch.bool,
            ).view(batch, 1, 1, 1, 1)
            corrected = torch.where(reset, base_gops[:, index], corrected)
        outputs.append(corrected)
    return torch.stack(outputs, dim=1)


def decode_base_gops(
    model: nn.Module,
    sequence: torch.Tensor,
    mode: AETVModeSpec,
    cell: ChannelCell,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = sequence.shape[0]
    if batch != 1 and cell.label != "clean":
        raise ValueError("waveform evaluation expects batch size one")
    separated = split_gops(sequence, mode.gop_frames)
    count = separated.shape[0] // batch
    z = model.encoder(separated)
    if cell.snr_db is None and cell.fading is None:
        received = z
        weights = torch.ones_like(z)
    else:
        received_rows = []
        weight_rows = []
        for item in z:
            latent, weight, _ = simulate_transmission(
                item.float().cpu().numpy(),
                mode_name=mode.name,
                snr_db=cell.snr_db,
                fading_preset=cell.fading,
            )
            received_rows.append(torch.from_numpy(latent))
            weight_rows.append(torch.from_numpy(weight))
        received = torch.stack(received_rows).to(sequence.device)
        weights = torch.stack(weight_rows).to(sequence.device)
    decoded = model.decoder(received, weights)
    decoded = decoded.reshape(batch, count, *decoded.shape[1:])
    confidence = weights.float().mean(dim=1).reshape(batch, count).clamp(0, 1)
    return decoded, confidence


def train_adapter(args: argparse.Namespace, mode: AETVModeSpec, device: torch.device) -> list[dict]:
    model = load_model(args.checkpoint, mode, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = StatefulGOPCorrector(
        width=args.adapter_width,
        blocks=args.adapter_blocks,
        spatial_scale=args.spatial_scale,
        max_residual=args.max_residual,
        context_mode=args.context_mode,
        taper_floor=args.taper_floor,
    ).to(device)
    perceptual = MultiLayerVGGPerceptualLoss().to(device).eval()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    cache = args.data_dir / cache_name(mode, args.gops, "train")
    dataset = SequenceCache(cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    iterator = iter(loader)
    reset_rng = random.Random(args.seed + 17)
    history = []
    for step in range(1, args.steps + 1):
        try:
            source = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            source = next(iterator)
        source = source.to(device, non_blocking=True)
        with torch.no_grad():
            separated = split_gops(source, mode.gop_frames)
            base_flat = model.decoder(model.encoder(separated), torch.ones(
                separated.shape[0], mode.latents_per_gop, device=device
            ))
            count = base_flat.shape[0] // source.shape[0]
            base_gops = base_flat.reshape(source.shape[0], count, *base_flat.shape[1:])
            base_sequence = join_gops(base_flat, source.shape[0], count)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            corrected_gops = apply_adapter_sequence(
                adapter,
                base_gops,
                reset_probability=args.reset_probability,
                generator=reset_rng,
            )
            corrected = join_gops(
                corrected_gops.flatten(0, 1), source.shape[0], count
            )
            cross = boundary_losses(corrected, source, mode.gop_frames)
            source_l1 = F.l1_loss(corrected, source)
            anchor_l1 = F.l1_loss(corrected, base_sequence)
            vgg_source = perceptual(corrected, source)
            vgg_anchor = perceptual(corrected, base_sequence)
            total = (
                args.source_weight * source_l1
                + args.anchor_weight * anchor_l1
                + args.boundary_weight * cross["boundary_delta"]
                + args.lowpass_weight * cross["boundary_lowpass_step"]
                + args.acceleration_weight * cross["boundary_acceleration"]
                + args.within_weight * cross["within_delta"]
                + args.vgg_source_weight * vgg_source
                + args.vgg_anchor_weight * vgg_anchor
            )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        row = {
            "step": step,
            "total": float(total.detach()),
            "source_l1": float(source_l1.detach()),
            "anchor_l1": float(anchor_l1.detach()),
            "vgg_source": float(vgg_source.detach()),
            "vgg_anchor": float(vgg_anchor.detach()),
            **{name: float(value.detach()) for name, value in cross.items()},
        }
        history.append(row)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"step {step:>4}/{args.steps} total={row['total']:.5f} "
                f"boundary={row['boundary_delta']:.5f} source={row['source_l1']:.5f} "
                f"anchor={row['anchor_l1']:.5f} vgg={row['vgg_source']:.5f}",
                flush=True,
            )
    args.adapter.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_checkpoint(adapter, args.checkpoint, args), args.adapter)
    del perceptual, adapter, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def evaluate(
    args: argparse.Namespace,
    mode: AETVModeSpec,
    device: torch.device,
    adapter: StatefulGOPCorrector | None,
) -> dict:
    model = load_model(args.checkpoint, mode, device).eval()
    if adapter is not None:
        adapter.eval()
    cache = args.data_dir / cache_name(mode, args.gops, "eval")
    dataset = SequenceCache(cache)
    rows = {cell.label: [] for cell in DEFAULT_CELLS}
    with torch.inference_mode():
        for index in range(min(args.eval_sequences, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                base_gops, confidence = decode_base_gops(model, source, mode, cell)
                if adapter is None:
                    output_gops = base_gops
                else:
                    output_gops = apply_adapter_sequence(
                        adapter,
                        base_gops,
                        confidences=confidence,
                    )
                output = join_gops(output_gops.flatten(0, 1), 1, output_gops.shape[1])
                rows[cell.label].append(
                    sequence_metrics(
                        output,
                        source,
                        mode.gop_frames,
                        device,
                        include_lpips=not args.skip_lpips,
                    )
                )
            print(
                f"  evaluated {'adapter' if adapter is not None else 'baseline'} "
                f"{index + 1:>2}/{min(args.eval_sequences, len(dataset))}",
                flush=True,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"cells": [asdict(cell) for cell in DEFAULT_CELLS], "sequences": rows}


def render_examples(
    args: argparse.Namespace,
    mode: AETVModeSpec,
    device: torch.device,
    adapter: StatefulGOPCorrector,
) -> None:
    """Write source/baseline/context videos with GUI blending disabled."""
    model = load_model(args.checkpoint, mode, device).eval()
    adapter.eval()
    cache = args.data_dir / cache_name(mode, args.gops, "eval")
    dataset = SequenceCache(cache)
    render_dir = args.out / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(min(args.render_count, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            clean_base, clean_confidence = decode_base_gops(
                model, source, mode, ChannelCell("clean")
            )
            clean_context = apply_adapter_sequence(
                adapter, clean_base, confidences=clean_confidence
            )
            mpp_base, mpp_confidence = decode_base_gops(
                model, source, mode, ChannelCell("mpp_12db", 12.0, "mpp")
            )
            mpp_context = apply_adapter_sequence(
                adapter, mpp_base, confidences=mpp_confidence
            )
            count = clean_base.shape[1]
            panels = [
                ("Source", source),
                ("Baseline clean", join_gops(clean_base.flatten(0, 1), 1, count)),
                ("Context clean", join_gops(clean_context.flatten(0, 1), 1, count)),
                ("Baseline MPP 12", join_gops(mpp_base.flatten(0, 1), 1, count)),
                ("Context MPP 12", join_gops(mpp_context.flatten(0, 1), 1, count)),
            ]
            path = render_dir / f"sequence_{index:02d}.mp4"
            write_labeled_grid_mp4(panels, path, fps=mode.fps, columns=3)
            print(f"wrote {path}", flush=True)


def paired(values: list[float], baseline: list[float]) -> dict[str, float]:
    difference = [new - old for new, old in zip(values, baseline)]
    mean = st.mean(difference)
    se = st.stdev(difference) / math.sqrt(len(difference)) if len(difference) > 1 else 0.0
    return {"mean": mean, "se": se, "two_se": 2 * se}


def compare(baseline: dict, candidate: dict) -> dict:
    output = {"cells": {}}
    for cell, old_rows in baseline["sequences"].items():
        new_rows = candidate["sequences"][cell]
        output["cells"][cell] = {}
        for metric in old_rows[0]:
            output["cells"][cell][metric] = {
                "baseline_mean": st.mean(row[metric] for row in old_rows),
                "candidate_mean": st.mean(row[metric] for row in new_rows),
                "paired_delta": paired(
                    [row[metric] for row in new_rows],
                    [row[metric] for row in old_rows],
                ),
            }
    return output


def print_report(report: dict) -> None:
    lower = {
        "lpips", "boundary_delta", "boundary_lowpass_step", "boundary_acceleration",
        "within_delta", "boundary_ratio", "boundary_delta_lpips",
    }
    for cell, metrics in report["cells"].items():
        print(f"\n=== {cell} ===")
        for metric, values in metrics.items():
            delta = values["paired_delta"]
            resolved = delta["two_se"] > 0 and abs(delta["mean"]) > delta["two_se"]
            preferred = delta["mean"] < 0 if metric in lower else delta["mean"] > 0
            mark = "PASS" if resolved and preferred else ("REGRESS" if resolved else "flat")
            print(
                f"{metric:>26}: {values['baseline_mean']:.6f} -> "
                f"{values['candidate_mean']:.6f} delta={delta['mean']:+.6f} "
                f"+/-{delta['se']:.6f} {mark}"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("baseline", "train", "compare", "render", "all"))
    value.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    value.add_argument("--adapter", type=Path, default=Path("runs/gop-context-v8/adapter.pt"))
    value.add_argument("--out", type=Path, default=Path("runs/gop-context-v8"))
    value.add_argument("--data-dir", type=Path, default=Path("runs/gop-boundary-data"))
    value.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    value.add_argument("--gops", type=int, default=3)
    value.add_argument("--eval-sequences", type=int, default=32)
    value.add_argument("--render-count", type=int, default=3)
    value.add_argument("--steps", type=int, default=500)
    value.add_argument("--batch", type=int, default=2)
    value.add_argument("--lr", type=float, default=2e-4)
    value.add_argument("--adapter-width", type=int, default=24)
    value.add_argument("--adapter-blocks", type=int, default=3)
    value.add_argument("--spatial-scale", type=int, default=4)
    value.add_argument("--max-residual", type=float, default=0.12)
    value.add_argument("--context-mode", choices=("last", "full"), default="last")
    value.add_argument("--taper-floor", type=float, default=0.0)
    value.add_argument("--reset-probability", type=float, default=0.2)
    value.add_argument("--source-weight", type=float, default=0.5)
    value.add_argument("--anchor-weight", type=float, default=2.0)
    value.add_argument("--boundary-weight", type=float, default=4.0)
    value.add_argument("--lowpass-weight", type=float, default=2.0)
    value.add_argument("--acceleration-weight", type=float, default=0.5)
    value.add_argument("--within-weight", type=float, default=0.2)
    value.add_argument("--vgg-source-weight", type=float, default=0.25)
    value.add_argument("--vgg-anchor-weight", type=float, default=0.5)
    value.add_argument("--seed", type=int, default=20260825)
    value.add_argument("--log-interval", type=int, default=25)
    value.add_argument("--skip-lpips", action="store_true")
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    mode = AETV_MODES[args.mode]
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_path = args.out / "baseline.json"
    candidate_path = args.out / "candidate.json"
    comparison_path = args.out / "comparison.json"

    if args.command in {"baseline", "all"}:
        print("Evaluating exact-reset baseline", flush=True)
        baseline = evaluate(args, mode, device, adapter=None)
        baseline_path.write_text(json.dumps(baseline, indent=2, allow_nan=True) + "\n")
        if args.command == "baseline":
            return

    if args.command in {"train", "all"}:
        print("Training stateful residual adapter with frozen V8 codec", flush=True)
        started = time.time()
        history = train_adapter(args, mode, device)
        (args.out / "training.json").write_text(json.dumps({
            "elapsed_s": time.time() - started,
            "steps": history,
        }, indent=2) + "\n")
        if args.command == "train":
            return

    if args.command in {"compare", "all"}:
        if not baseline_path.exists():
            raise SystemExit(f"missing baseline report: {baseline_path}")
        adapter, payload = load_adapter(args.adapter, device)
        source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if payload["source_sha256"] != source_hash:
            raise SystemExit("adapter was trained against a different base checkpoint")
        print("Evaluating stateful residual adapter", flush=True)
        candidate = evaluate(args, mode, device, adapter)
        candidate_path.write_text(json.dumps(candidate, indent=2, allow_nan=True) + "\n")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report = compare(baseline, candidate)
        comparison_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
        print_report(report)
        print(f"\nwrote {comparison_path}")

    if args.command in {"render", "all"}:
        adapter, payload = load_adapter(args.adapter, device)
        source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if payload["source_sha256"] != source_hash:
            raise SystemExit("adapter was trained against a different base checkpoint")
        render_examples(args, mode, device, adapter)


if __name__ == "__main__":
    main()
