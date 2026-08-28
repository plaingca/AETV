#!/usr/bin/env python3
"""Paired 32-clip exact-runtime evaluation for recurrent joint V8."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from aetv.config import AETV_MODES
from aetv.hfchannel import StreamingChannelEmulator
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.recurrent_feature_tcm import V8RecurrentFeatureTCM
from aetv.recurrent_joint_codec import V8RecurrentJointCodec
from aetv.transition_anchor_codec import V8TransitionAnchorCodec
from scripts.experiment_gop_boundaries import (
    DEFAULT_CELLS,
    TX_LEVEL,
    SequenceCache,
    load_model,
    profile_for_cell,
    runtime_channel_seed,
    runtime_retry_seed,
    simple_ssim,
)


def runtime_transmit(values: np.ndarray, cell, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact continuous station waveform/channel/demodulator path."""
    mode = AETV_MODES["V8"]
    channel = StreamingChannelEmulator(profile_for_cell(cell), seed=seed, fs=mode.geometry.fs)
    demodulator = StreamingDemodulator(mode.band, continuous=True, mode_name=mode.name)
    received, weights = [], []
    block = max(1, mode.geometry.fs // 10)
    for clean in modulate_continuous_chunks(values, mode_name=mode.name, callsign="EVAL"):
        clean = np.asarray(clean, dtype=np.float32).copy()
        peak = float(np.max(np.abs(clean))) if clean.size else 0.0
        if peak:
            clean *= TX_LEVEL / peak
        impaired = channel.process(clean)
        peak = float(np.max(np.abs(impaired))) if impaired.size else 0.0
        if peak:
            impaired *= TX_LEVEL / peak
        for start in range(0, len(impaired), block):
            for result in demodulator.feed(impaired[start : start + block]):
                for latent, confidence in zip(result.gops_latents, result.gops_weights):
                    received.append(np.asarray(latent, dtype=np.float32))
                    weights.append(np.asarray(confidence, dtype=np.float32))
    if len(received) != values.shape[0]:
        raise RuntimeError(f"recovered {len(received)}/{values.shape[0]} GOPs")
    return np.stack(received), np.stack(weights)


def paired_runtime(
    baseline: np.ndarray,
    candidate: np.ndarray,
    cell,
    initial_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    for attempt in range(32):
        seed = runtime_retry_seed(initial_seed, attempt)
        try:
            base_rx, base_w = runtime_transmit(baseline, cell, seed)
            cand_rx, cand_w = runtime_transmit(candidate, cell, seed)
            return base_rx, base_w, cand_rx, cand_w, seed
        except RuntimeError:
            if attempt == 31:
                raise
    raise AssertionError("unreachable")


def decode_baseline(model, received: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    outputs = [model.decoder(received[:, index], weights[:, index]) for index in range(received.shape[1])]
    return torch.cat(outputs, dim=2)


def lpips_batch(metric, first: torch.Tensor, second: torch.Tensor, chunk: int = 24) -> float:
    values = []
    with torch.inference_mode():
        for start in range(0, len(first), chunk):
            values.append(metric(first[start : start + chunk] * 2 - 1, second[start : start + chunk] * 2 - 1))
    return float(torch.cat(values).mean())


def sequence_metrics(recon: torch.Tensor, target: torch.Tensor, perceptual) -> dict[str, float]:
    recon = recon.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    frames = recon.shape[2]
    boundaries = torch.arange(6, frames, 6, device=recon.device)
    seams = boundaries - 1
    delta = recon[:, :, 1:] - recon[:, :, :-1]
    source_delta = target[:, :, 1:] - target[:, :, :-1]
    error = delta - source_delta
    boundary_error = error.index_select(2, seams)
    mask = torch.ones(frames - 1, dtype=torch.bool, device=recon.device)
    mask[seams] = False
    within = error[:, :, mask]

    boundary_recon = delta.index_select(2, seams).permute(0, 2, 1, 3, 4).flatten(0, 1)
    boundary_target = source_delta.index_select(2, seams).permute(0, 2, 1, 3, 4).flatten(0, 1)
    all_recon = recon.permute(0, 2, 1, 3, 4).flatten(0, 1)
    all_target = target.permute(0, 2, 1, 3, 4).flatten(0, 1)

    luminance = (
        0.299 * (recon[:, 0:1] - target[:, 0:1])
        + 0.587 * (recon[:, 1:2] - target[:, 1:2])
        + 0.114 * (recon[:, 2:3] - target[:, 2:3])
    )
    luminance = F.avg_pool3d(luminance, (1, 8, 8), stride=(1, 8, 8))
    spectrum = torch.fft.rfft(luminance, dim=2)
    signature_bin = max(1, min(spectrum.shape[2] - 1, round(frames / 6)))

    def detail(value: torch.Tensor) -> torch.Tensor:
        dx = value[..., 1:] - value[..., :-1]
        dy = value[..., 1:, :] - value[..., :-1, :]
        return 0.5 * (dx.abs().mean() + dy.abs().mean())

    mse = F.mse_loss(recon, target)
    return {
        "psnr": float(10 * torch.log10(1 / mse.clamp_min(1e-12))),
        "ssim": simple_ssim(recon, target),
        "lpips": lpips_batch(perceptual, all_recon, all_target),
        "boundary_error": float(boundary_error.abs().mean()),
        "boundary_delta_lpips": lpips_batch(
            perceptual,
            (0.5 + 0.5 * boundary_recon).clamp(0, 1),
            (0.5 + 0.5 * boundary_target).clamp(0, 1),
        ),
        "gop_signature_1hz": float(spectrum[:, :, signature_bin].abs().mean() / math.sqrt(frames)),
        "within_gop_delta_error": float(within.abs().mean()),
        "spatial_detail_ratio": float(detail(recon) / detail(target).clamp_min(1e-12)),
    }


class GridWriter:
    def __init__(self, path: Path, fps: float = 6.0):
        self.path = path
        self.width, self.height = 192, 108
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width * 3}x{self.height}", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libopenh264", "-b:v", "3M", "-pix_fmt", "yuv420p", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, source: torch.Tensor, baseline: torch.Tensor, candidate: torch.Tensor) -> None:
        panels = (("Source", source), ("Released V8", baseline), ("Recurrent joint", candidate))
        frames = [value[0].mul(255).byte().permute(1, 2, 3, 0).cpu().numpy() for _, value in panels]
        for frame_set in zip(*frames):
            image = Image.fromarray(np.concatenate(frame_set, axis=1))
            draw = ImageDraw.Draw(image)
            for index, (label, _) in enumerate(panels):
                x = index * self.width
                draw.rectangle((x, 0, x + self.width, 16), fill=(0, 0, 0))
                draw.text((x + 4, 2), label, fill=(255, 255, 255))
            assert self.process.stdin is not None
            self.process.stdin.write(image.tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr[-1000:]}")


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    keys = rows[0]["baseline"]
    result = {}
    for key in keys:
        base = float(np.mean([row["baseline"][key] for row in rows]))
        candidate = float(np.mean([row["candidate"][key] for row in rows]))
        result[key] = {
            "baseline": base,
            "candidate": candidate,
            "delta": candidate - base,
            "relative_percent": 100 * (candidate - base) / max(abs(base), 1e-12),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_16gop_joint_eval"))
    parser.add_argument("--out", type=Path, default=Path("runs/v8-recurrent-joint-16gop/eval-paired-32"))
    parser.add_argument("--clips", type=int, default=32)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    device = torch.device(args.device)
    mode = AETV_MODES["V8"]
    data = SequenceCache(args.data, limit=args.clips, max_frames=96)
    if len(data) != args.clips:
        raise ValueError(f"requested {args.clips} clips but cache contains {len(data)}")

    baseline = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    payload = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if payload.get("kind") == V8TransitionAnchorCodec.checkpoint_kind:
        anchor_values = int(payload.get("args", {}).get("anchor_values", 0))
        candidate = V8TransitionAnchorCodec.from_released(
            anchor_values=anchor_values
        ).to(device).eval()
    elif payload.get("kind") == V8RecurrentFeatureTCM.checkpoint_kind:
        candidate = V8RecurrentFeatureTCM.from_released().to(device).eval()
        candidate.aligner.eval()
    elif payload.get("kind") == V8RecurrentJointCodec.checkpoint_kind:
        candidate = V8RecurrentJointCodec.from_released().to(device).eval()
    else:
        raise ValueError("candidate is not a recurrent joint/anchor/feature-TCM V8 checkpoint")
    candidate.load_state_dict(payload["model_state_dict"], strict=True)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    writers = (
        {cell.label: GridWriter(args.out / "renders" / f"full-paired-{args.clips}-{cell.label}-no-gui-blend.mp4") for cell in DEFAULT_CELLS}
        if not args.no_render else {}
    )
    rows_by_cell = {cell.label: [] for cell in DEFAULT_CELLS}
    recovery_rows = []

    try:
        for sequence_index in range(len(data)):
            source = data[sequence_index].unsqueeze(0).to(device)
            with torch.inference_mode():
                baseline_tx = torch.stack([
                    baseline.encoder(source[:, :, index * 6 : (index + 1) * 6])
                    for index in range(16)
                ], dim=1)
                candidate_tx = candidate.encode_gops(source)
            if baseline_tx.shape != candidate_tx.shape or candidate_tx.shape[-1] != 2816:
                raise RuntimeError("candidate changed the V8 wire geometry")
            runtime = {}
            for cell_index, cell in enumerate(DEFAULT_CELLS):
                initial_seed = runtime_channel_seed(args.seed, sequence_index, cell_index)
                base_rx, base_w, cand_rx, cand_w, used_seed = paired_runtime(
                    baseline_tx[0].float().cpu().numpy(),
                    candidate_tx[0].float().cpu().numpy(),
                    cell,
                    initial_seed,
                )
                base_rx_t = torch.from_numpy(base_rx).to(device).unsqueeze(0)
                base_w_t = torch.from_numpy(base_w).to(device).unsqueeze(0)
                cand_rx_t = torch.from_numpy(cand_rx).to(device).unsqueeze(0)
                cand_w_t = torch.from_numpy(cand_w).to(device).unsqueeze(0)
                with torch.inference_mode():
                    base_video = decode_baseline(baseline, base_rx_t, base_w_t)
                    cand_video = candidate.decode_gops(cand_rx_t, cand_w_t)
                base_metrics = sequence_metrics(base_video, source, perceptual)
                cand_metrics = sequence_metrics(cand_video, source, perceptual)
                row = {
                    "sequence": sequence_index,
                    "seed": used_seed,
                    "baseline": base_metrics,
                    "candidate": cand_metrics,
                }
                rows_by_cell[cell.label].append(row)
                runtime[cell.label] = (base_rx_t, base_w_t, cand_rx_t, cand_w_t, base_video, cand_video)
                if writers:
                    writers[cell.label].write(source.cpu(), base_video.cpu(), cand_video.cpu())

            # Exact steady-path RX tensors are composed into receiver event
            # tests.  The fade cells themselves each came from continuous
            # runtime streams with identical paired seeds.
            schedule = ["clean", "clean", "measured_hf", "mpp_12db"] + ["clean"] * 12
            cand_rx = torch.cat([runtime[label][2][:, index : index + 1] for index, label in enumerate(schedule)], dim=1)
            cand_w = torch.cat([runtime[label][3][:, index : index + 1] for index, label in enumerate(schedule)], dim=1)
            base_rx = torch.cat([runtime[label][0][:, index : index + 1] for index, label in enumerate(schedule)], dim=1)
            base_w = torch.cat([runtime[label][1][:, index : index + 1] for index, label in enumerate(schedule)], dim=1)
            with torch.inference_mode():
                fade_candidate = candidate.decode_gops(cand_rx, cand_w)
                fade_baseline = decode_baseline(baseline, base_rx, base_w)
                reset = torch.zeros(1, 16, dtype=torch.bool, device=device)
                reset[:, 8] = True
                reset_candidate = candidate.decode_gops(runtime["measured_hf"][2], runtime["measured_hf"][3], reset=reset)
            def gop_l1(value, index):
                return float(F.l1_loss(value[:, :, index * 6 : (index + 1) * 6], source[:, :, index * 6 : (index + 1) * 6]))
            recovery_rows.append({
                "sequence": sequence_index,
                "fade_recovery_gop": 4,
                "fade_baseline_l1": gop_l1(fade_baseline, 4),
                "fade_candidate_l1": gop_l1(fade_candidate, 4),
                "reset_recovery_gop": 8,
                "reset_baseline_l1": gop_l1(runtime["measured_hf"][4], 8),
                "reset_candidate_l1": gop_l1(reset_candidate, 8),
            })
            print(json.dumps({"sequence": sequence_index + 1, "of": len(data)}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    cells = {label: aggregate(rows) for label, rows in rows_by_cell.items()}
    fade_ratio = float(np.mean([r["fade_candidate_l1"] / max(r["fade_baseline_l1"], 1e-12) for r in recovery_rows]))
    reset_ratio = float(np.mean([r["reset_candidate_l1"] / max(r["reset_baseline_l1"], 1e-12) for r in recovery_rows]))
    reasons = []
    if args.clips != 32:
        reasons.append(f"screening run has {args.clips} clips; promotion requires 32")
    for label, cell in cells.items():
        boundary_reduction = -cell["boundary_error"]["relative_percent"]
        if boundary_reduction < 60:
            reasons.append(f"{label} boundary reduction {boundary_reduction:.2f}% < 60%")
        if cell["boundary_delta_lpips"]["relative_percent"] >= -1:
            reasons.append(f"{label} boundary-delta LPIPS lacks clear reduction")
        if cell["gop_signature_1hz"]["relative_percent"] >= -1:
            reasons.append(f"{label} 1 Hz signature lacks clear reduction")
        if cell["lpips"]["relative_percent"] > 1:
            reasons.append(f"{label} LPIPS regression exceeds 1%")
        if cell["psnr"]["delta"] < -0.1:
            reasons.append(f"{label} PSNR loss exceeds 0.1 dB")
        if cell["within_gop_delta_error"]["relative_percent"] > 1:
            reasons.append(f"{label} within-GOP motion regression exceeds 1%")
        if cell["spatial_detail_ratio"]["candidate"] + 0.01 < cell["spatial_detail_ratio"]["baseline"]:
            reasons.append(f"{label} material spatial-detail regression")
    if fade_ratio > 1.0:
        reasons.append(f"fade recovery ratio {fade_ratio:.4f} exceeds baseline")
    if reset_ratio > 1.0:
        reasons.append(f"reset recovery ratio {reset_ratio:.4f} exceeds baseline")
    result = {
        "schema": 1,
        "candidate": str(args.candidate.resolve()),
        "baseline": str(Path("models/v8-hf3k-face-gan.pt").resolve()),
        "wire_contract": {"values_per_six_frame_gop": 2816, "changed": False},
        "runtime_path": ["modulate_continuous_chunks", "StreamingChannelEmulator", "StreamingDemodulator(continuous=True)"],
        "gui_blending": False,
        "clips": args.clips,
        "frames_per_clip": 96,
        "cells": cells,
        "recovery": {"fade_error_ratio": fade_ratio, "reset_error_ratio": reset_ratio},
        "promoted": not reasons,
        "reasons": reasons,
        "per_sequence": rows_by_cell,
        "recovery_per_sequence": recovery_rows,
    }
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"promoted": not reasons, "reasons": reasons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
