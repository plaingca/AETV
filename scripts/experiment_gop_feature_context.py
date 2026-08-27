#!/usr/bin/env python3
"""Train motion-aligned decoder-feature context across AETV GOP boundaries.

This is the feature-domain counterpart to ``experiment_gop_flow.py``.  It
keeps the encoder, transmitted latent budget, and released decoder frozen, but
captures the decoder's final full-resolution feature map.  RAFT aligns the
previous GOP's last feature map to the new GOP, and a residual adapter learns
which aligned features improve the reconstruction before the stock RGB output
layer.  Missing state remains an exact bypass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from aetv.models import AETVDecoder, MultiLayerVGGPerceptualLoss  # noqa: E402
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
from experiment_gop_context import ContextResidualBlock, compare, print_report  # noqa: E402
from experiment_gop_flow import RAFTAligner, temporal_taper  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402


class FeatureContextRefiner(nn.Module):
    """Residual adapter over current and motion-aligned decoder features."""

    def __init__(
        self,
        feature_channels: int = 32,
        width: int = 64,
        blocks: int = 6,
        spatial_scale: int = 2,
        max_residual: float = 1.0,
        taper: tuple[float, ...] = (1.0, 0.7, 0.35, 0.1, 0.0, 0.0),
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.width = width
        self.blocks = blocks
        self.spatial_scale = spatial_scale
        self.max_residual = max_residual
        self.taper = tuple(taper)
        # Current, aligned previous, their difference, and an RGB-derived
        # reliability mask.  The adapter acts on decoder features, not pixels.
        input_channels = 3 * feature_channels + 1
        self.input = nn.Conv3d(input_channels, width, 3, padding=1)
        self.body = nn.Sequential(*(ContextResidualBlock(width) for _ in range(blocks)))
        self.output = nn.Conv3d(width, feature_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def config(self) -> dict:
        return {
            "feature_channels": self.feature_channels,
            "width": self.width,
            "blocks": self.blocks,
            "spatial_scale": self.spatial_scale,
            "max_residual": self.max_residual,
            "taper": self.taper,
        }

    def forward(
        self,
        current: torch.Tensor,
        aligned_previous: torch.Tensor | None,
        reliability: torch.Tensor | None = None,
        confidence: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if aligned_previous is None:
            return current
        if current.shape != aligned_previous.shape or current.ndim != 5:
            raise ValueError("current and aligned features must have equal BCTHW shapes")
        batch, channels, frames, height, width = current.shape
        if channels != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels")
        if reliability is None:
            reliability = current.new_ones((batch, 1, frames, height, width))
        elif reliability.shape != (batch, 1, frames, height, width):
            reliability = F.interpolate(
                reliability,
                size=(frames, height, width),
                mode="trilinear",
                align_corners=False,
            )
        features = torch.cat(
            (current, aligned_previous, current - aligned_previous, reliability), dim=1
        )
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
        taper = temporal_taper(
            self.taper, frames, device=current.device, dtype=current.dtype
        ).view(1, 1, frames, 1, 1)
        if confidence is None:
            gate = current.new_ones((batch, 1, 1, 1, 1))
        else:
            gate = torch.as_tensor(confidence, device=current.device, dtype=current.dtype)
            if gate.ndim == 0:
                gate = gate.expand(batch)
            gate = gate.reshape(batch, 1, 1, 1, 1).clamp(0, 1)
        return current + self.max_residual * torch.tanh(residual) * taper * gate


def _latent_grids(
    decoder: AETVDecoder,
    latents: torch.Tensor,
    weights: torch.Tensor | None,
    output_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    t_lat, h_lat, w_lat = decoder._get_grid_shape(output_shape)
    total = decoder.latent_channels * t_lat * h_lat * w_lat
    batch = latents.shape[0]
    z_flat = latents.new_zeros((batch, total))
    w_flat = latents.new_zeros((batch, total))
    copy_len = min(latents.shape[1], total)
    z_flat[:, :copy_len] = latents[:, :copy_len]
    if weights is None:
        w_flat[:, :copy_len] = 1.0
    else:
        w_flat[:, :copy_len] = weights[:, :copy_len]
    shape = (batch, decoder.latent_channels, t_lat, h_lat, w_lat)
    return z_flat.reshape(shape), w_flat.reshape(shape)


def decode_to_features(
    decoder: AETVDecoder,
    latents: torch.Tensor,
    weights: torch.Tensor | None = None,
    output_shape: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the stock decoder and expose its final feature and skip tensors."""
    if output_shape is None:
        output_shape = (
            decoder.mode.gop_frames,
            decoder.mode.height,
            decoder.mode.width,
        )
    z, w = _latent_grids(decoder, latents, weights, output_shape)
    inner = decoder.decoder
    frames, height, width = output_shape
    if inner.smooth_temporal_skip:
        temporal_skip = F.interpolate(
            inner.temporal_skip(z * w),
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )
    else:
        temporal_skip = F.interpolate(
            inner.temporal_skip(z * w), size=output_shape, mode="nearest"
        )
    x = inner.attn(inner.r0b(inner.r0(inner.input(torch.cat([z * w, w], dim=1)))))
    if inner.deeper:
        x = inner.m0b(inner.m0a(x))
    if inner.deepest:
        x = inner.m0c(x)
    if inner.deep4:
        x = inner.m0d(x)
    if inner.compact:
        x = inner.r0c(inner.up0(x))
    x = inner.r1b(inner.r1(inner.up1(x)))
    if inner.deeper:
        x = inner.m1b(inner.m1a(x))
    if inner.deepest:
        x = inner.attn1(inner.m1c(x))
    x = F.interpolate(x, size=(frames, height // 4, width // 4), mode="nearest")
    x = inner.r2b(inner.r2(inner.up2(x)))
    if inner.deep_tail:
        x = inner.r2c(x)
    if inner.deeper:
        x = inner.r2d(x)
    if inner.deepest:
        x = inner.r2e(x)
    if inner.deep4:
        x = inner.attn2(x)
    x = F.silu(inner.up3(x))
    if inner.deep_tail:
        x = inner.r3b(inner.r3(x))
    if inner.deeper:
        x = inner.r3d(inner.r3c(x))
    if inner.deepest:
        x = inner.r3f(inner.r3e(x))
    if inner.deep4:
        x = inner.r3h(inner.r3g(x))
    output = torch.sigmoid(inner.output(x) + temporal_skip)
    return output, x, temporal_skip


def render_features(
    decoder: AETVDecoder, features: torch.Tensor, temporal_skip: torch.Tensor
) -> torch.Tensor:
    return torch.sigmoid(decoder.decoder.output(features) + temporal_skip)


def receive_and_decode_features(
    model: nn.Module,
    sequence: torch.Tensor,
    mode: AETVModeSpec,
    cell: ChannelCell,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = sequence.shape[0]
    separated = split_gops(sequence, mode.gop_frames)
    count = separated.shape[0] // batch
    z = model.encoder(separated)
    if cell.snr_db is None and cell.fading is None:
        received = z
        weights = torch.ones_like(z)
    else:
        received_rows, weight_rows = [], []
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
    output, features, skip = decode_to_features(model.decoder, received, weights)
    confidence = weights.float().mean(dim=1).reshape(batch, count).clamp(0, 1)
    reshape = lambda value: value.reshape(batch, count, *value.shape[1:])
    return reshape(output), reshape(features), reshape(skip), confidence


def apply_feature_sequence(
    refiner: FeatureContextRefiner,
    decoder: AETVDecoder,
    aligner: RAFTAligner,
    base_gops: torch.Tensor,
    features: torch.Tensor,
    skips: torch.Tensor,
    *,
    confidences: torch.Tensor | None = None,
    photometric_threshold: float = 0.10,
    photometric_softness: float = 0.02,
    scene_threshold_multiplier: float = 1.5,
    output_flow_strength: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= output_flow_strength <= 1.0:
        raise ValueError("output flow strength must be between zero and one")
    batch, count, _, frames, height, width = base_gops.shape
    outputs = [base_gops[:, 0]]
    feature_states = [features[:, 0]]
    for index in range(1, count):
        current_frames = base_gops[:, index].permute(0, 2, 1, 3, 4).flatten(0, 1)
        previous_rgb = outputs[-1][:, :, -1]
        rgb_references = previous_rgb[:, None].expand(
            -1, frames, -1, -1, -1
        ).flatten(0, 1)
        flow = aligner.estimate_flow(rgb_references, current_frames)
        warped_rgb = aligner.warp_with_flow(rgb_references, flow)
        current_features = features[:, index]
        feature_references = feature_states[-1][:, :, -1]
        feature_references = feature_references[:, None].expand(
            -1, frames, -1, -1, -1
        ).flatten(0, 1)
        warped_features = aligner.warp_with_flow(feature_references, flow)
        warped_features = warped_features.reshape(
            batch, frames, -1, height, width
        ).permute(0, 2, 1, 3, 4)
        error = (warped_rgb - current_frames).abs().mean(dim=1, keepdim=True)
        pixel_gate = torch.sigmoid(
            (photometric_threshold - error) / photometric_softness
        )
        frame_error = error.mean(dim=(1, 2, 3), keepdim=True)
        scene_gate = torch.sigmoid(
            (photometric_threshold * scene_threshold_multiplier - frame_error)
            / photometric_softness
        )
        reliability = (pixel_gate * scene_gate).reshape(
            batch, frames, 1, height, width
        ).permute(0, 2, 1, 3, 4)
        confidence = (
            None
            if confidences is None
            else torch.minimum(confidences[:, index - 1], confidences[:, index])
        )
        corrected_features = refiner(
            current_features, warped_features, reliability, confidence
        )
        output = render_features(decoder, corrected_features, skips[:, index])
        if output_flow_strength > 0:
            output_frames = output.permute(0, 2, 1, 3, 4).flatten(0, 1)
            output_taper = temporal_taper(
                refiner.taper,
                frames,
                device=output.device,
                dtype=output.dtype,
            ).view(1, frames, 1, 1, 1)
            output_gate = reliability
            if confidence is not None:
                output_gate = output_gate * confidence.view(batch, 1, 1, 1, 1)
            output_gate = output_gate.permute(0, 2, 1, 3, 4).flatten(0, 1)
            alpha = output_taper.expand(batch, -1, -1, -1, -1).flatten(0, 1)
            output_frames = output_frames + output_flow_strength * alpha * output_gate * (
                warped_rgb - output_frames
            )
            output = output_frames.clamp(0, 1).reshape(
                batch, frames, 3, height, width
            ).permute(0, 2, 1, 3, 4)
        feature_states.append(corrected_features)
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def save_checkpoint(
    refiner: FeatureContextRefiner, args: argparse.Namespace, elapsed_s: float
) -> None:
    payload = {
        "kind": "aetv-motion-aligned-feature-context",
        "mode": args.mode,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "refiner_config": refiner.config(),
        "refiner_state_dict": refiner.state_dict(),
        "wire_contract_changed": False,
        "base_codec_modified": False,
        "reset_is_exact_bypass": True,
        "training": {
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "elapsed_s": elapsed_s,
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
    }
    args.refiner.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.refiner)


def load_refiner(path: Path, device: torch.device) -> tuple[FeatureContextRefiner, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "aetv-motion-aligned-feature-context":
        raise ValueError(f"{path} is not a feature-context checkpoint")
    refiner = FeatureContextRefiner(**payload["refiner_config"]).to(device)
    refiner.load_state_dict(payload["refiner_state_dict"], strict=True)
    return refiner, payload


def train(args: argparse.Namespace, mode: AETVModeSpec, device: torch.device) -> list[dict]:
    model = load_model(args.checkpoint, mode, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    aligner = RAFTAligner(device).eval()
    feature_channels = model.decoder.decoder.output.conv.in_channels
    refiner = FeatureContextRefiner(
        feature_channels=feature_channels,
        width=args.refiner_width,
        blocks=args.refiner_blocks,
        spatial_scale=args.spatial_scale,
        max_residual=args.max_feature_residual,
        taper=tuple(args.taper),
    ).to(device)
    perceptual = MultiLayerVGGPerceptualLoss().to(device).eval()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)
    dataset = SequenceCache(args.data_dir / cache_name(mode, args.gops, "train"))
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    iterator = iter(loader)
    history: list[dict] = []
    started = time.time()
    for step in range(1, args.steps + 1):
        try:
            source = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            source = next(iterator)
        source = source.to(device)
        with torch.no_grad():
            base, features, skips, confidence = receive_and_decode_features(
                model, source, mode, ChannelCell("clean")
            )
            base_sequence = join_gops(
                base.flatten(0, 1), source.shape[0], base.shape[1]
            )
        optimizer.zero_grad(set_to_none=True)
        corrected_gops = apply_feature_sequence(
            refiner,
            model.decoder,
            aligner,
            base,
            features,
            skips,
            confidences=confidence,
            photometric_threshold=args.photometric_threshold,
            photometric_softness=args.photometric_softness,
            scene_threshold_multiplier=args.scene_threshold_multiplier,
        )
        corrected = join_gops(
            corrected_gops.flatten(0, 1), source.shape[0], corrected_gops.shape[1]
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
        grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
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
    save_checkpoint(refiner, args, time.time() - started)
    return history


def evaluate(
    args: argparse.Namespace,
    mode: AETVModeSpec,
    device: torch.device,
    refiner: FeatureContextRefiner | None,
) -> dict:
    model = load_model(args.checkpoint, mode, device).eval()
    aligner = RAFTAligner(device).eval() if refiner is not None else None
    if refiner is not None:
        refiner.eval()
    dataset = SequenceCache(args.data_dir / cache_name(mode, args.gops, "eval"))
    rows = {cell.label: [] for cell in DEFAULT_CELLS}
    timings = []
    with torch.inference_mode():
        for index in range(min(args.eval_sequences, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                base, features, skips, confidence = receive_and_decode_features(
                    model, source, mode, cell
                )
                if refiner is None:
                    output_gops = base
                else:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    output_gops = apply_feature_sequence(
                        refiner,
                        model.decoder,
                        aligner,
                        base,
                        features,
                        skips,
                        confidences=confidence,
                        photometric_threshold=args.photometric_threshold,
                        photometric_softness=args.photometric_softness,
                        scene_threshold_multiplier=args.scene_threshold_multiplier,
                        output_flow_strength=args.output_flow_strength,
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
            print(
                f"  evaluated {'feature context' if refiner else 'baseline'} "
                f"{index + 1:>2}/{min(args.eval_sequences, len(dataset))}",
                flush=True,
            )
    result = {"cells": [asdict(cell) for cell in DEFAULT_CELLS], "sequences": rows}
    if timings:
        result["runtime_ms_per_boundary"] = (
            1000 * st.mean(timings) / max(1, args.gops - 1)
        )
    return result


def render_examples(
    args: argparse.Namespace,
    mode: AETVModeSpec,
    device: torch.device,
    refiner: FeatureContextRefiner,
) -> None:
    model = load_model(args.checkpoint, mode, device).eval()
    aligner = RAFTAligner(device).eval()
    dataset = SequenceCache(args.data_dir / cache_name(mode, args.gops, "eval"))
    render_dir = args.out / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(min(args.render_count, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            panels = [("Source", source)]
            for cell in (ChannelCell("clean"), ChannelCell("mpp_12db", 12.0, "mpp")):
                base, features, skips, confidence = receive_and_decode_features(
                    model, source, mode, cell
                )
                corrected = apply_feature_sequence(
                    refiner,
                    model.decoder,
                    aligner,
                    base,
                    features,
                    skips,
                    confidences=confidence,
                    photometric_threshold=args.photometric_threshold,
                    photometric_softness=args.photometric_softness,
                    scene_threshold_multiplier=args.scene_threshold_multiplier,
                    output_flow_strength=args.output_flow_strength,
                )
                panels.extend(
                    [
                        (f"Baseline {cell.label}", join_gops(base.flatten(0, 1), 1, base.shape[1])),
                        (
                            f"Feature context {cell.label}",
                            join_gops(corrected.flatten(0, 1), 1, corrected.shape[1]),
                        ),
                    ]
                )
            path = render_dir / f"sequence_{index:02d}.mp4"
            write_labeled_grid_mp4(panels, path, fps=mode.fps, columns=3)
            print(f"wrote {path}", flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("baseline", "train", "compare", "render", "all"))
    value.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    value.add_argument("--refiner", type=Path, default=Path("runs/gop-feature-context-v8/refiner.pt"))
    value.add_argument("--out", type=Path, default=Path("runs/gop-feature-context-v8"))
    value.add_argument("--data-dir", type=Path, default=Path("runs/gop-boundary-data"))
    value.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    value.add_argument("--gops", type=int, default=3)
    value.add_argument("--eval-sequences", type=int, default=32)
    value.add_argument("--render-count", type=int, default=3)
    value.add_argument("--steps", type=int, default=500)
    value.add_argument("--batch", type=int, default=1)
    value.add_argument("--lr", type=float, default=5e-5)
    value.add_argument("--refiner-width", type=int, default=64)
    value.add_argument("--refiner-blocks", type=int, default=6)
    value.add_argument("--spatial-scale", type=int, default=2)
    value.add_argument("--max-feature-residual", type=float, default=1.0)
    value.add_argument("--taper", type=float, nargs="+", default=[1, 0.7, 0.35, 0.1, 0, 0])
    value.add_argument("--photometric-threshold", type=float, default=0.10)
    value.add_argument("--photometric-softness", type=float, default=0.02)
    value.add_argument("--scene-threshold-multiplier", type=float, default=1.5)
    value.add_argument("--output-flow-strength", type=float, default=0.0)
    value.add_argument("--source-weight", type=float, default=1.0)
    value.add_argument("--anchor-weight", type=float, default=0.5)
    value.add_argument("--boundary-weight", type=float, default=8.0)
    value.add_argument("--lowpass-weight", type=float, default=4.0)
    value.add_argument("--acceleration-weight", type=float, default=1.0)
    value.add_argument("--within-weight", type=float, default=1.0)
    value.add_argument("--vgg-source-weight", type=float, default=0.5)
    value.add_argument("--vgg-anchor-weight", type=float, default=0.25)
    value.add_argument("--seed", type=int, default=20260825)
    value.add_argument("--log-interval", type=int, default=25)
    value.add_argument("--skip-lpips", action="store_true")
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return value


def main() -> None:
    args = parser().parse_args()
    mode = AETV_MODES[args.mode]
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    run_config = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "refiner": str(args.refiner.resolve()),
        "eval_sequences": args.eval_sequences,
        "gops": args.gops,
        "photometric_threshold": args.photometric_threshold,
        "photometric_softness": args.photometric_softness,
        "scene_threshold_multiplier": args.scene_threshold_multiplier,
        "output_flow_strength": args.output_flow_strength,
        "taper": args.taper,
    }
    (args.out / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    baseline_path = args.out / "baseline.json"

    if args.command in {"baseline", "all"}:
        baseline = evaluate(args, mode, device, None)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        if args.command == "baseline":
            return
    if args.command in {"train", "all"}:
        history = train(args, mode, device)
        (args.out / "training.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if args.command == "train":
            return
    if args.command in {"compare", "all"}:
        if not baseline_path.is_file():
            raise SystemExit(f"missing baseline report: {baseline_path}")
        refiner, payload = load_refiner(args.refiner, device)
        if payload["source_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise SystemExit("refiner was trained against a different checkpoint")
        candidate = evaluate(args, mode, device, refiner)
        (args.out / "candidate.json").write_text(
            json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
        )
        report = compare(json.loads(baseline_path.read_text()), candidate)
        (args.out / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print_report(report)
        print(f"\nruntime: {candidate['runtime_ms_per_boundary']:.2f} ms/boundary")
    if args.command in {"render", "all"}:
        refiner, _ = load_refiner(args.refiner, device)
        render_examples(args, mode, device, refiner.eval())


if __name__ == "__main__":
    main()
