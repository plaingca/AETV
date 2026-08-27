#!/usr/bin/env python3
"""Evaluate motion-aligned context across independently decoded AETV GOPs.

The V7/V8 wire format and codec remain unchanged.  At each GOP boundary a
pretrained RAFT-Small model aligns the last decoded frame of the previous GOP
to the first frames of the current GOP.  A photometric/scene gate then fuses
only trustworthy aligned pixels, with a temporal taper that reaches zero
before the end of the GOP.  The state can therefore be reset or bypassed at
any boundary without changing latent compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES, AETVModeSpec  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    DEFAULT_CELLS,
    ChannelCell,
    SequenceCache,
    cache_name,
    join_gops,
    load_model,
    sequence_metrics,
)
from experiment_gop_context import (  # noqa: E402
    compare,
    decode_base_gops,
    print_report,
)
from eval import write_labeled_grid_mp4  # noqa: E402


@dataclass(frozen=True)
class FlowContextConfig:
    strength: float = 0.75
    photometric_threshold: float = 0.10
    photometric_softness: float = 0.02
    scene_threshold_multiplier: float = 1.5
    taper: tuple[float, ...] = (1.0, 0.7, 0.35, 0.1, 0.0, 0.0)

    def validate(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between zero and one")
        if self.photometric_threshold <= 0 or self.photometric_softness <= 0:
            raise ValueError("photometric threshold and softness must be positive")
        if self.scene_threshold_multiplier <= 0:
            raise ValueError("scene threshold multiplier must be positive")
        if not self.taper or any(not 0.0 <= value <= 1.0 for value in self.taper):
            raise ValueError("taper values must be between zero and one")


def _raft_padding(height: int, width: int) -> tuple[int, int, int, int]:
    """Symmetric padding to RAFT's >=128 and multiple-of-eight contract."""
    padded_height = max(128, math.ceil(height / 8) * 8)
    padded_width = max(128, math.ceil(width / 8) * 8)
    vertical = padded_height - height
    horizontal = padded_width - width
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    return left, right, top, bottom


def sample_padded_reference(
    reference: torch.Tensor,
    flow: torch.Tensor,
    *,
    left: int,
    top: int,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    """Backward-warp a padded NCHW reference with target-to-reference flow."""
    batch, _, padded_height, padded_width = reference.shape
    if flow.shape != (batch, 2, output_height, output_width):
        raise ValueError(
            f"flow shape {tuple(flow.shape)} does not match "
            f"{(batch, 2, output_height, output_width)}"
        )
    y, x = torch.meshgrid(
        torch.arange(output_height, device=reference.device, dtype=flow.dtype),
        torch.arange(output_width, device=reference.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = x.unsqueeze(0) + left + flow[:, 0]
    sample_y = y.unsqueeze(0) + top + flow[:, 1]
    grid_x = 2.0 * sample_x / max(1, padded_width - 1) - 1.0
    grid_y = 2.0 * sample_y / max(1, padded_height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        reference,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class RAFTAligner(nn.Module):
    """Frozen RAFT-Small target-to-reference aligner."""

    def __init__(self, device: torch.device):
        super().__init__()
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transforms = weights.transforms()

    def estimate_flow(self, reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if reference.shape != target.shape or reference.ndim != 4:
            raise ValueError("reference and target must have the same NCHW shape")
        _, _, height, width = target.shape
        left, right, top, bottom = _raft_padding(height, width)
        padding = (left, right, top, bottom)
        padded_reference = F.pad(reference, padding, mode="replicate")
        padded_target = F.pad(target, padding, mode="replicate")
        normalized_target, normalized_reference = self.transforms(
            padded_target, padded_reference
        )
        # RAFT(image1, image2) predicts image1-to-image2 flow.  Running target
        # first produces the backward sampling field required to warp the
        # previous reference into the current target coordinates.
        flow = self.model(normalized_target, normalized_reference)[-1]
        return flow[:, :, top : top + height, left : left + width]

    def warp_with_flow(self, reference: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        height, width = flow.shape[-2:]
        return sample_padded_reference(
            reference,
            flow,
            left=0,
            top=0,
            output_height=height,
            output_width=width,
        )

    def forward(self, reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.warp_with_flow(reference, self.estimate_flow(reference, target))


def temporal_taper(
    values: tuple[float, ...], frames: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    taper = torch.tensor(values, device=device, dtype=dtype)
    if taper.numel() != frames:
        taper = F.interpolate(
            taper.view(1, 1, -1), size=frames, mode="linear", align_corners=True
        ).reshape(-1)
    return taper


def apply_flow_sequence(
    base_gops: torch.Tensor,
    aligner: nn.Module,
    config: FlowContextConfig,
    *,
    confidences: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply resettable motion-aligned context to BGCTHW decoded GOPs."""
    config.validate()
    if base_gops.ndim != 6:
        raise ValueError(f"expected BGCTHW base GOPs, got {tuple(base_gops.shape)}")
    batch, count, channels, frames, height, width = base_gops.shape
    outputs = [base_gops[:, 0]]
    taper = temporal_taper(
        config.taper,
        frames,
        device=base_gops.device,
        dtype=base_gops.dtype,
    ).view(1, frames, 1, 1, 1)
    for index in range(1, count):
        current = base_gops[:, index]
        current_frames = current.permute(0, 2, 1, 3, 4).flatten(0, 1)
        previous_frame = outputs[-1][:, :, -1]
        references = previous_frame[:, None].expand(
            -1, frames, -1, -1, -1
        ).flatten(0, 1)
        warped = aligner(references, current_frames)
        photometric_error = (warped - current_frames).abs().mean(dim=1, keepdim=True)
        pixel_gate = torch.sigmoid(
            (config.photometric_threshold - photometric_error)
            / config.photometric_softness
        )
        frame_error = photometric_error.mean(dim=(1, 2, 3), keepdim=True)
        scene_gate = torch.sigmoid(
            (
                config.photometric_threshold * config.scene_threshold_multiplier
                - frame_error
            )
            / config.photometric_softness
        )
        gate = pixel_gate * scene_gate
        if confidences is not None:
            boundary_confidence = torch.minimum(
                confidences[:, index - 1], confidences[:, index]
            ).clamp(0, 1)
            gate = gate.reshape(batch, frames, 1, height, width)
            gate = gate * boundary_confidence.view(batch, 1, 1, 1, 1)
            gate = gate.flatten(0, 1)
        alpha = taper.expand(batch, -1, -1, -1, -1).flatten(0, 1)
        corrected = current_frames + config.strength * alpha * gate * (
            warped - current_frames
        )
        corrected = corrected.clamp(0, 1).reshape(
            batch, frames, channels, height, width
        ).permute(0, 2, 1, 3, 4)
        outputs.append(corrected)
    return torch.stack(outputs, dim=1)


def evaluate(
    args: argparse.Namespace,
    mode: AETVModeSpec,
    device: torch.device,
    aligner: RAFTAligner | None,
    config: FlowContextConfig,
) -> dict:
    model = load_model(args.checkpoint, mode, device).eval()
    cache = args.data_dir / cache_name(mode, args.gops, "eval")
    dataset = SequenceCache(cache)
    rows = {cell.label: [] for cell in DEFAULT_CELLS}
    timings: list[float] = []
    with torch.inference_mode():
        for index in range(min(args.eval_sequences, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                base_gops, confidence = decode_base_gops(model, source, mode, cell)
                if aligner is None:
                    output_gops = base_gops
                else:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    output_gops = apply_flow_sequence(
                        base_gops, aligner, config, confidences=confidence
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timings.append(time.perf_counter() - started)
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
            label = "flow" if aligner is not None else "baseline"
            print(
                f"  evaluated {label} {index + 1:>2}/"
                f"{min(args.eval_sequences, len(dataset))}",
                flush=True,
            )
    result = {
        "cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "sequences": rows,
    }
    if timings:
        result["flow_runtime_ms_per_sequence"] = 1000.0 * st.mean(timings)
        result["flow_runtime_ms_per_boundary"] = (
            1000.0 * st.mean(timings) / max(1, args.gops - 1)
        )
    return result


def render_examples(
    args: argparse.Namespace,
    mode: AETVModeSpec,
    device: torch.device,
    aligner: RAFTAligner,
    config: FlowContextConfig,
) -> None:
    model = load_model(args.checkpoint, mode, device).eval()
    cache = args.data_dir / cache_name(mode, args.gops, "eval")
    dataset = SequenceCache(cache)
    render_dir = args.out / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(min(args.render_count, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            panels: list[tuple[str, torch.Tensor]] = [("Source", source)]
            for cell in (ChannelCell("clean"), ChannelCell("mpp_12db", 12.0, "mpp")):
                base, confidence = decode_base_gops(model, source, mode, cell)
                corrected = apply_flow_sequence(
                    base, aligner, config, confidences=confidence
                )
                panels.extend(
                    [
                        (f"Baseline {cell.label}", join_gops(base.flatten(0, 1), 1, base.shape[1])),
                        (
                            f"Flow context {cell.label}",
                            join_gops(corrected.flatten(0, 1), 1, corrected.shape[1]),
                        ),
                    ]
                )
            path = render_dir / f"sequence_{index:02d}.mp4"
            write_labeled_grid_mp4(panels, path, fps=mode.fps, columns=3)
            print(f"wrote {path}", flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("baseline", "compare", "render", "all"))
    value.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    value.add_argument("--out", type=Path, default=Path("runs/gop-flow-v8"))
    value.add_argument("--data-dir", type=Path, default=Path("runs/gop-boundary-data"))
    value.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    value.add_argument("--gops", type=int, default=3)
    value.add_argument("--eval-sequences", type=int, default=32)
    value.add_argument("--render-count", type=int, default=3)
    value.add_argument("--strength", type=float, default=0.75)
    value.add_argument("--photometric-threshold", type=float, default=0.10)
    value.add_argument("--photometric-softness", type=float, default=0.02)
    value.add_argument("--scene-threshold-multiplier", type=float, default=1.5)
    value.add_argument("--taper", type=float, nargs="+", default=[1, 0.7, 0.35, 0.1, 0, 0])
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
    config = FlowContextConfig(
        strength=args.strength,
        photometric_threshold=args.photometric_threshold,
        photometric_softness=args.photometric_softness,
        scene_threshold_multiplier=args.scene_threshold_multiplier,
        taper=tuple(args.taper),
    )
    config.validate()
    (args.out / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    baseline_path = args.out / "baseline.json"
    candidate_path = args.out / "candidate.json"

    if args.command in {"baseline", "all"}:
        baseline = evaluate(args, mode, device, None, config)
        baseline_path.write_text(
            json.dumps(baseline, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
        if args.command == "baseline":
            return

    aligner = RAFTAligner(device)
    if args.command in {"compare", "all"}:
        if not baseline_path.is_file():
            raise SystemExit(f"missing baseline report: {baseline_path}")
        candidate = evaluate(args, mode, device, aligner, config)
        candidate_path.write_text(
            json.dumps(candidate, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
        report = compare(
            json.loads(baseline_path.read_text(encoding="utf-8")), candidate
        )
        comparison_path = args.out / "comparison.json"
        comparison_path.write_text(
            json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
        print_report(report)
        print(
            f"\nflow runtime: {candidate['flow_runtime_ms_per_boundary']:.2f} "
            f"ms/boundary\nwrote {comparison_path}"
        )

    if args.command in {"render", "all"}:
        render_examples(args, mode, device, aligner, config)


if __name__ == "__main__":
    main()
