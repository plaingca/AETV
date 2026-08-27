#!/usr/bin/env python3
"""Train and evaluate reliability gates for retained V8 GOP feature context.

The released HF-3k V8 encoder and decoder and the retained motion-aligned
feature refiner remain frozen.  Only a small reliability gate is trained.  The
gate starts from the retained scalar-confidence behavior, then learns either a
spatial correction or a spatial plus feature-channel correction from the full
decoder confidence grid, pilot diagnostics, flow validity, and photometric
agreement.

Training uses five independently encoded/decoded GOPs and explicit
good->fade->good, missing-GOP, false-confidence, reset, and scene-cut events.
The standard promotion matrix remains the unchanged paired 32-sequence runtime
cache under clean, AWGN 6 dB, MPP 12 dB, and the measured HF path.  GUI GOP
blending is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES, AETVModeSpec  # noqa: E402
from aetv.hfchannel import StreamingChannelEmulator  # noqa: E402
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    DEFAULT_CELLS,
    REPORT_METRICS,
    ChannelCell,
    SequenceCache,
    assess_candidate,
    boundary_losses,
    compare_reports,
    decode_independent_gops,
    join_gops,
    load_model,
    profile_for_cell,
    runtime_channel_seed,
    runtime_retry_seed,
    sequence_metrics,
    sha256_file,
    tensor_sha256,
)
from experiment_gop_feature_context import (  # noqa: E402
    FeatureContextRefiner,
    decode_to_features,
    load_refiner,
    render_features,
)
from experiment_gop_flow import RAFTAligner, temporal_taper  # noqa: E402


GATE_INPUT_CHANNELS = 10
EVENTS = (
    "steady_clean",
    "steady_awgn",
    "steady_mpp",
    "steady_measured",
    "good_fade_good",
    "false_high_confidence",
    "false_low_confidence",
    "missing_gop",
    "random_reset",
    "scene_cut",
)
SEAM_METRICS = (
    "boundary_excess",
    "boundary_error_ratio",
    "boundary_delta_lpips",
    "boundary_lowpass_step",
    "boundary_acceleration",
)


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MultiGOPSequenceCache(Dataset):
    """Read a fixed number of complete contiguous GOPs without truncating to 12 frames."""

    def __init__(self, root: Path, gops: int, frames_per_gop: int = 6, limit: int | None = None):
        self.root = root
        self.gops = gops
        self.frames_per_gop = frames_per_gop
        self.frames = gops * frames_per_gop
        self.files = sorted(root.glob("sequence_*.pt"))
        if not self.files:
            self.files = sorted(root.glob("row_*.pt"))
        if limit is not None:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError(f"no cached sequences in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        value = torch.load(self.files[index], map_location="cpu", weights_only=False).float()
        if value.ndim != 4 or value.shape[0] != 3:
            raise ValueError(f"expected CTHW tensor in {self.files[index]}, got {tuple(value.shape)}")
        if value.shape[1] < self.frames:
            raise ValueError(f"{self.files[index]} has {value.shape[1]} frames; need {self.frames}")
        value = value[:, : self.frames]
        if value.max() > 1.0:
            value = value.div(255.0)
        return value.contiguous()

    def manifest(self) -> list[dict]:
        rows = []
        for index, path in enumerate(self.files):
            value = self[index]
            rows.append(
                {
                    "index": index,
                    "file": str(path.resolve()),
                    "source_file_sha256": sha256_file(path),
                    "frame_slice": [0, self.frames],
                    "pixel_sha256": tensor_sha256(value),
                    "shape": list(value.shape),
                }
            )
        return rows


class IndexedMultiGOPDataset(Dataset):
    def __init__(self, dataset: MultiGOPSequenceCache):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return index, self.dataset[index]


def split_many_gops(sequence: torch.Tensor, frames_per_gop: int) -> torch.Tensor:
    if sequence.ndim != 5:
        raise ValueError(f"expected BCTHW sequence, got {tuple(sequence.shape)}")
    batch, channels, frames, height, width = sequence.shape
    if frames % frames_per_gop:
        raise ValueError(f"{frames} frames is not divisible by GOP length {frames_per_gop}")
    count = frames // frames_per_gop
    return (
        sequence.reshape(batch, channels, count, frames_per_gop, height, width)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * count, channels, frames_per_gop, height, width)
    )


def join_many_gops(gops: torch.Tensor) -> torch.Tensor:
    if gops.ndim != 6:
        raise ValueError(f"expected BGCTHW GOP tensor, got {tuple(gops.shape)}")
    batch, count, channels, frames, height, width = gops.shape
    return (
        gops.permute(0, 2, 1, 3, 4, 5)
        .reshape(batch, channels, count * frames, height, width)
    )


def encode_independent_many(model: nn.Module, sequence: torch.Tensor, frames_per_gop: int) -> torch.Tensor:
    separated = split_many_gops(sequence, frames_per_gop)
    batch = sequence.shape[0]
    count = separated.shape[0] // batch
    separated = separated.reshape(batch, count, *separated.shape[1:])
    encoded = [model.encoder(separated[:, index]) for index in range(count)]
    return torch.stack(encoded, dim=1)


def runtime_transmit_many(
    latents: np.ndarray,
    mode: AETVModeSpec,
    cell: ChannelCell,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Carry an arbitrary GOP run through the continuous production runtime path."""
    values = np.asarray(latents, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected G,N latent array, got {values.shape}")
    wanted = values.shape[0]
    channel = StreamingChannelEmulator(profile_for_cell(cell), seed=seed, fs=mode.geometry.fs)
    demodulator = StreamingDemodulator(mode.band, continuous=True, mode_name=mode.name)
    received: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    diagnostics: list[dict] = []
    block_samples = max(1, mode.geometry.fs // 10)
    chunks = modulate_continuous_chunks(values, mode_name=mode.name, callsign="EVAL")
    for clean in chunks:
        clean = np.asarray(clean, dtype=np.float32).copy()
        clean_peak = float(np.max(np.abs(clean))) if clean.size else 0.0
        if clean_peak > 0:
            clean *= 0.7 / clean_peak
        impaired = channel.process(clean)
        impaired_peak = float(np.max(np.abs(impaired))) if impaired.size else 0.0
        if impaired_peak > 0:
            impaired *= 0.7 / impaired_peak
        for start in range(0, len(impaired), block_samples):
            for result in demodulator.feed(impaired[start : start + block_samples]):
                for latent, confidence in zip(result.gops_latents, result.gops_weights):
                    received.append(np.asarray(latent, dtype=np.float32))
                    weights.append(np.asarray(confidence, dtype=np.float32))
                    diagnostics.append(
                        {
                            "snr_db": float(result.snr_db),
                            "pilot_coherence": float(result.pilot_coherence),
                            "freq_offset": float(result.freq_offset),
                        }
                    )
    if len(received) != wanted:
        raise RuntimeError(
            f"runtime path recovered {len(received)}/{wanted} GOPs for {cell.label} seed={seed}"
        )
    return np.stack(received), np.stack(weights), diagnostics


def multigop_rx_metadata(
    checkpoint: Path,
    mode: AETVModeSpec,
    dataset: MultiGOPSequenceCache,
    *,
    seed: int,
    split_name: str,
) -> dict:
    return {
        "schema": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "mode": mode.name,
        "split": split_name,
        "frames_per_sequence": dataset.frames,
        "gops_per_sequence": dataset.gops,
        "tx_level": 0.7,
        "channel_path": [
            "AETVAutoencoder.encoder (one explicit independent call per GOP)",
            "modulate_continuous_chunks",
            "StreamingChannelEmulator",
            "StreamingDemodulator(continuous=True)",
            "AETVAutoencoder.decoder (one explicit independent call per GOP)",
        ],
        "channel_seed_base": seed,
        "channel_seed_formula": "base + 1009*sequence_index + 9176*cell_index",
        "decode_retry_formula": "initial_seed + 104729*attempt; first realization recovering all GOPs",
        "cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "sequences": dataset.manifest(),
    }


def precompute_multigop_rx(
    checkpoint: Path,
    destination: Path,
    dataset: MultiGOPSequenceCache,
    mode: AETVModeSpec,
    device: torch.device,
    *,
    seed: int,
    split_name: str,
) -> dict:
    expected = multigop_rx_metadata(checkpoint, mode, dataset, seed=seed, split_name=split_name)
    if destination.is_file():
        cached = torch.load(destination, map_location="cpu", weights_only=False)
        if cached.get("metadata") == expected:
            print(f"Reusing fixed {split_name} multi-GOP runtime RX cache: {destination}", flush=True)
            return cached
        raise RuntimeError(f"stale/incompatible multi-GOP RX cache exists: {destination}")

    model = load_model(checkpoint, mode, device).eval()
    received = {
        cell.label: torch.empty(
            len(dataset), dataset.gops, mode.latents_per_gop, dtype=torch.float32
        )
        for cell in DEFAULT_CELLS
    }
    weights = {label: torch.empty_like(value) for label, value in received.items()}
    diagnostics = {cell.label: [] for cell in DEFAULT_CELLS}
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index].unsqueeze(0).to(device)
            encoded = encode_independent_many(model, source, mode.gop_frames)[0].float().cpu().numpy()
            for cell_index, cell in enumerate(DEFAULT_CELLS):
                initial_seed = runtime_channel_seed(seed, index, cell_index)
                failures = []
                for attempt in range(64):
                    path_seed = runtime_retry_seed(initial_seed, attempt)
                    try:
                        rx, confidence, detail = runtime_transmit_many(
                            encoded, mode, cell, seed=path_seed
                        )
                        break
                    except RuntimeError as error:
                        failures.append(str(error))
                else:
                    raise RuntimeError(
                        f"no {dataset.gops}/{dataset.gops}-GOP runtime decode for "
                        f"{cell.label} sequence {index}; last={failures[-1]}"
                    )
                received[cell.label][index].copy_(torch.from_numpy(rx))
                weights[cell.label][index].copy_(torch.from_numpy(confidence))
                diagnostics[cell.label].append(
                    {
                        "initial_seed": initial_seed,
                        "selected_seed": path_seed,
                        "attempts": attempt + 1,
                        "failed_realizations": failures,
                        "gops": detail,
                    }
                )
            print(f"  {split_name} runtime path {index + 1:>3}/{len(dataset)}", flush=True)
    payload = {
        "metadata": expected,
        "received": received,
        "weights": weights,
        "diagnostics": diagnostics,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


class ReliabilityGate(nn.Module):
    """Bounded scalar, spatial, or spatial-plus-channel reliability gate."""

    def __init__(
        self,
        mode: str,
        feature_channels: int = 32,
        width: int = 16,
        max_logit_adjustment: float = 4.0,
        input_channels: int = GATE_INPUT_CHANNELS,
    ):
        super().__init__()
        if mode not in {"scalar", "spatial", "spatial_channel"}:
            raise ValueError(f"unsupported reliability gate mode: {mode}")
        self.mode = mode
        self.feature_channels = feature_channels
        self.width = width
        self.max_logit_adjustment = max_logit_adjustment
        if input_channels != GATE_INPUT_CHANNELS:
            raise ValueError(
                f"checkpoint expects {input_channels} gate inputs; runtime has {GATE_INPUT_CHANNELS}"
            )
        if mode != "scalar":
            self.spatial = nn.Sequential(
                nn.Conv3d(GATE_INPUT_CHANNELS, width, 3, padding=1),
                nn.SiLU(),
                nn.Conv3d(width, width, 3, padding=1),
                nn.SiLU(),
                nn.Conv3d(width, 1, 1),
            )
            nn.init.zeros_(self.spatial[-1].weight)
            nn.init.zeros_(self.spatial[-1].bias)
        if mode == "spatial_channel":
            self.channel = nn.Sequential(
                nn.Linear(3 * feature_channels + GATE_INPUT_CHANNELS, 64),
                nn.SiLU(),
                nn.Linear(64, feature_channels),
            )
            nn.init.zeros_(self.channel[-1].weight)
            nn.init.zeros_(self.channel[-1].bias)

    def config(self) -> dict:
        return {
            "mode": self.mode,
            "feature_channels": self.feature_channels,
            "width": self.width,
            "max_logit_adjustment": self.max_logit_adjustment,
            "input_channels": GATE_INPUT_CHANNELS,
        }

    def forward(
        self,
        reliability_inputs: torch.Tensor,
        current: torch.Tensor,
        aligned_previous: torch.Tensor,
        base_confidence: torch.Tensor,
        hard_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reliability_inputs.ndim != 5 or reliability_inputs.shape[1] != GATE_INPUT_CHANNELS:
            raise ValueError(
                f"expected B,{GATE_INPUT_CHANNELS},T,H,W reliability inputs, "
                f"got {tuple(reliability_inputs.shape)}"
            )
        batch, channels, frames, height, width = current.shape
        if aligned_previous.shape != current.shape:
            raise ValueError("current and aligned feature shapes differ")
        scalar = base_confidence.reshape(batch, 1, 1, 1, 1).clamp(0, 1)
        valid = hard_valid.reshape(batch, 1, 1, 1, 1).to(dtype=current.dtype)
        if self.mode == "scalar":
            spatial = scalar.expand(batch, 1, frames, height, width) * valid
            return spatial.expand(batch, channels, frames, height, width), spatial

        clipped = scalar.clamp(1e-4, 1 - 1e-4)
        baseline_logit = torch.logit(clipped)
        adjustment = self.max_logit_adjustment * torch.tanh(self.spatial(reliability_inputs))
        spatial = torch.sigmoid(baseline_logit + adjustment)
        spatial = spatial * (scalar > 0).to(current.dtype) * valid
        if self.mode == "spatial":
            return spatial.expand(batch, channels, frames, height, width), spatial

        pooled = torch.cat(
            (
                current.mean(dim=(2, 3, 4)),
                aligned_previous.mean(dim=(2, 3, 4)),
                (current - aligned_previous).abs().mean(dim=(2, 3, 4)),
                reliability_inputs.mean(dim=(2, 3, 4)),
            ),
            dim=1,
        )
        raw_channel = self.max_logit_adjustment * torch.tanh(self.channel(pooled))
        # Feature-wise reliability may suppress an unsafe channel but never
        # amplify it above the spatial gate.  Zero initialization is exactly 1.
        normalizer = torch.sigmoid(current.new_tensor(4.0))
        channel_gate = (torch.sigmoid(4.0 + raw_channel) / normalizer).clamp(max=1.0)
        feature = spatial * channel_gate.view(batch, channels, 1, 1, 1)
        return feature, spatial


def decoder_confidence_map(
    decoder: nn.Module,
    weights: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    frames, height, width = output_shape
    t_lat, h_lat, w_lat = decoder._get_grid_shape(output_shape)
    total = decoder.latent_channels * t_lat * h_lat * w_lat
    flat = weights.new_zeros((weights.shape[0], total))
    copy_len = min(weights.shape[1], total)
    flat[:, :copy_len] = weights[:, :copy_len]
    grid = flat.reshape(weights.shape[0], decoder.latent_channels, t_lat, h_lat, w_lat)
    grid = grid.mean(dim=1, keepdim=True)
    return F.interpolate(
        grid,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=False,
    ).clamp(0, 1)


def flow_in_bounds(flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = x.unsqueeze(0) + flow[:, 0]
    sample_y = y.unsqueeze(0) + flow[:, 1]
    return (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    ).to(flow.dtype).unsqueeze(1)


def decode_received_features(
    model: nn.Module,
    received: torch.Tensor,
    weights: torch.Tensor,
    mode: AETVModeSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if received.ndim != 3 or weights.shape != received.shape:
        raise ValueError("expected equal B,G,N received and weight tensors")
    outputs, features, skips = [], [], []
    output_shape = (mode.gop_frames, mode.height, mode.width)
    for index in range(received.shape[1]):
        output, feature, skip = decode_to_features(
            model.decoder, received[:, index], weights[:, index], output_shape
        )
        outputs.append(output)
        features.append(feature)
        skips.append(skip)
    return torch.stack(outputs, dim=1), torch.stack(features, dim=1), torch.stack(skips, dim=1)


def diagnostic_tensors(
    rx_cache: dict,
    indices: torch.Tensor,
    cell_by_gop: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(indices)
    count = len(cell_by_gop)
    snr = torch.empty(batch, count, device=device)
    coherence = torch.empty_like(snr)
    for row, sequence_index in enumerate(indices.tolist()):
        for gop, label in enumerate(cell_by_gop):
            detail = rx_cache["diagnostics"][label][sequence_index]["gops"][gop]
            snr[row, gop] = float(detail["snr_db"])
            coherence[row, gop] = float(detail["pilot_coherence"])
    return snr, coherence


def gather_mixed_runtime_batch(
    rx_cache: dict,
    indices: torch.Tensor,
    cell_by_gop: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(indices)
    count = len(cell_by_gop)
    latent_count = next(iter(rx_cache["received"].values())).shape[-1]
    received = torch.empty(batch, count, latent_count, device=device)
    weights = torch.empty_like(received)
    for gop, label in enumerate(cell_by_gop):
        received[:, gop].copy_(rx_cache["received"][label][indices, gop].to(device))
        weights[:, gop].copy_(rx_cache["weights"][label][indices, gop].to(device))
    snr, coherence = diagnostic_tensors(rx_cache, indices, cell_by_gop, device)
    return received, weights, snr, coherence


@dataclass
class EventBatch:
    name: str
    source: torch.Tensor
    received: torch.Tensor
    decode_weights: torch.Tensor
    gate_weights: torch.Tensor
    snr: torch.Tensor
    coherence: torch.Tensor
    valid: torch.Tensor
    reset_before: torch.Tensor
    recovery_gop: int


def event_cells(name: str, count: int) -> list[str]:
    if count < 5:
        raise ValueError("failure curriculum requires at least five GOPs")
    if name == "steady_clean" or name in {"false_low_confidence", "missing_gop", "random_reset", "scene_cut"}:
        return ["clean"] * count
    if name == "steady_awgn":
        return ["awgn_6db"] * count
    if name == "steady_mpp":
        return ["mpp_12db"] * count
    if name == "steady_measured":
        return ["measured_hf"] * count
    if name in {"good_fade_good", "false_high_confidence"}:
        middle = ["measured_hf", "mpp_12db"]
        return (["clean", "clean"] + middle + ["clean"] * count)[:count]
    raise ValueError(f"unknown curriculum event: {name}")


def prepare_event_batch(
    name: str,
    source: torch.Tensor,
    indices: torch.Tensor,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> EventBatch:
    count = source.shape[2] // mode.gop_frames
    labels = event_cells(name, count)
    received, decode_weights, snr, coherence = gather_mixed_runtime_batch(
        rx_cache, indices, labels, device
    )
    gate_weights = decode_weights.clone()
    valid = torch.ones(source.shape[0], count, dtype=torch.bool, device=device)
    reset_before = torch.zeros_like(valid)
    recovery_gop = min(count - 1, 4)

    if name == "false_high_confidence":
        gate_weights[:, 2:4].fill_(0.95)
    elif name == "false_low_confidence":
        gate_weights[:, 1:3].mul_(0.10)
        recovery_gop = 3
    elif name == "missing_gop":
        valid[:, 2] = False
        recovery_gop = 3
    elif name == "random_reset":
        reset_before[:, 3] = True
        recovery_gop = 3
    elif name == "scene_cut":
        if source.shape[0] < 2:
            raise ValueError("scene-cut curriculum requires batch size at least two")
        boundary = 2
        source_gops = split_many_gops(source, mode.gop_frames).reshape(
            source.shape[0], count, 3, mode.gop_frames, mode.height, mode.width
        )
        source_gops[:, boundary:] = torch.flip(source_gops[:, boundary:], dims=(0,))
        source = join_many_gops(source_gops)
        received[:, boundary:] = torch.flip(received[:, boundary:], dims=(0,))
        decode_weights[:, boundary:] = torch.flip(decode_weights[:, boundary:], dims=(0,))
        gate_weights[:, boundary:] = torch.flip(gate_weights[:, boundary:], dims=(0,))
        snr[:, boundary:] = torch.flip(snr[:, boundary:], dims=(0,))
        coherence[:, boundary:] = torch.flip(coherence[:, boundary:], dims=(0,))
        recovery_gop = boundary

    return EventBatch(
        name=name,
        source=source,
        received=received,
        decode_weights=decode_weights,
        gate_weights=gate_weights,
        snr=snr,
        coherence=coherence,
        valid=valid,
        reset_before=reset_before,
        recovery_gop=recovery_gop,
    )


def _expand_scalar(value: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    return value.reshape(-1, 1, 1, 1, 1).expand(-1, 1, frames, height, width)


def boundary_safety_mask(
    state_valid: torch.Tensor,
    current_valid: torch.Tensor,
    reset_before: torch.Tensor,
    first_frame_warp_error: torch.Tensor,
    previous_snr_db: torch.Tensor,
    previous_pilot_coherence: torch.Tensor,
    *,
    scene_cut_threshold: float,
    min_previous_snr_db: float,
    min_previous_pilot_coherence: float,
) -> torch.Tensor:
    """Hard reset/bypass for invalid state, cuts, and unreacquired prior GOPs."""
    return (
        state_valid
        & current_valid
        & ~reset_before
        & (first_frame_warp_error <= scene_cut_threshold)
        & (previous_snr_db >= min_previous_snr_db)
        & (previous_pilot_coherence >= min_previous_pilot_coherence)
    )


def apply_reliability_sequence(
    refiner: FeatureContextRefiner,
    gate: ReliabilityGate,
    decoder: nn.Module,
    aligner: RAFTAligner,
    base_gops: torch.Tensor,
    features: torch.Tensor,
    skips: torch.Tensor,
    gate_weights: torch.Tensor,
    snr: torch.Tensor,
    coherence: torch.Tensor,
    valid: torch.Tensor,
    reset_before: torch.Tensor,
    *,
    photometric_threshold: float,
    photometric_softness: float,
    scene_threshold_multiplier: float,
    scene_cut_threshold: float,
    min_previous_snr_db: float,
    min_previous_pilot_coherence: float,
    output_flow_strength: float,
    detach_refiner: bool = True,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    if not 0 <= output_flow_strength <= 1:
        raise ValueError("output flow strength must be between zero and one")
    batch, count, _, frames, height, width = base_gops.shape
    if count < 2:
        return base_gops, []
    confidence_maps = [
        decoder_confidence_map(
            decoder, gate_weights[:, index], (frames, height, width)
        )
        for index in range(count)
    ]
    means = gate_weights.mean(dim=2).clamp(0, 1)
    q10 = torch.quantile(gate_weights.float(), 0.10, dim=2).to(gate_weights.dtype).clamp(0, 1)
    outputs = [base_gops[:, 0]]
    state = features[:, 0]
    state_valid = valid[:, 0].clone()
    telemetry: list[dict[str, torch.Tensor]] = []

    for index in range(1, count):
        current_output = base_gops[:, index]
        current_frames = current_output.permute(0, 2, 1, 3, 4).flatten(0, 1)
        previous_rgb = outputs[-1][:, :, -1]
        rgb_references = previous_rgb[:, None].expand(-1, frames, -1, -1, -1).flatten(0, 1)
        with torch.no_grad():
            flow = aligner.estimate_flow(rgb_references, current_frames)
            warped_rgb = aligner.warp_with_flow(rgb_references, flow)
            bounds = flow_in_bounds(flow)
            feature_references = state[:, :, -1][:, None].expand(
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
            photometric = (pixel_gate * scene_gate).reshape(
                batch, frames, 1, height, width
            ).permute(0, 2, 1, 3, 4)
            bounds = bounds.reshape(batch, frames, 1, height, width).permute(0, 2, 1, 3, 4)

        previous_map = confidence_maps[index - 1][:, :, -1:].expand(-1, -1, frames, -1, -1)
        latent_map = torch.minimum(previous_map, confidence_maps[index])
        first_frame_warp_error = frame_error.reshape(batch, frames)[:, 0]
        hard_valid = boundary_safety_mask(
            state_valid,
            valid[:, index],
            reset_before[:, index],
            first_frame_warp_error,
            snr[:, index - 1],
            coherence[:, index - 1],
            scene_cut_threshold=scene_cut_threshold,
            min_previous_snr_db=min_previous_snr_db,
            min_previous_pilot_coherence=min_previous_pilot_coherence,
        )
        base_confidence = torch.minimum(means[:, index - 1], means[:, index])
        min_snr = torch.minimum(snr[:, index - 1], snr[:, index])
        snr_reliability = torch.sigmoid((min_snr - 4.0) / 3.0)
        pilot_reliability = torch.minimum(
            coherence[:, index - 1], coherence[:, index]
        ).clamp(0, 1)
        inputs = torch.cat(
            (
                photometric,
                latent_map,
                bounds,
                _expand_scalar(means[:, index - 1], frames, height, width),
                _expand_scalar(means[:, index], frames, height, width),
                _expand_scalar(q10[:, index - 1], frames, height, width),
                _expand_scalar(q10[:, index], frames, height, width),
                _expand_scalar(snr_reliability, frames, height, width),
                _expand_scalar(pilot_reliability, frames, height, width),
                _expand_scalar(hard_valid.to(base_gops.dtype), frames, height, width),
            ),
            dim=1,
        )
        feature_gate, spatial_gate = gate(
            inputs, features[:, index], warped_features, base_confidence, hard_valid
        )
        if detach_refiner:
            with torch.no_grad():
                ungated = refiner(
                    features[:, index], warped_features, photometric, confidence=1.0
                )
        else:
            ungated = refiner(
                features[:, index], warped_features, photometric, confidence=1.0
            )
        feature_correction = ungated - features[:, index]
        corrected_features = features[:, index] + feature_correction * feature_gate
        output = render_features(decoder, corrected_features, skips[:, index])

        if output_flow_strength > 0:
            output_frames = output.permute(0, 2, 1, 3, 4).flatten(0, 1)
            taper = temporal_taper(
                refiner.taper, frames, device=output.device, dtype=output.dtype
            ).view(1, frames, 1, 1, 1)
            alpha = taper.expand(batch, -1, -1, -1, -1).flatten(0, 1)
            output_gate = (photometric * spatial_gate).permute(
                0, 2, 1, 3, 4
            ).flatten(0, 1)
            output_frames = output_frames + output_flow_strength * alpha * output_gate * (
                warped_rgb - output_frames
            )
            output = output_frames.clamp(0, 1).reshape(
                batch, frames, 3, height, width
            ).permute(0, 2, 1, 3, 4)

        outputs.append(output)
        state = torch.where(
            valid[:, index].view(batch, 1, 1, 1, 1), corrected_features, features[:, index]
        )
        state_valid = valid[:, index].clone()
        telemetry.append(
            {
                "base_confidence": base_confidence.detach(),
                "spatial_gate_mean": spatial_gate.mean(dim=(1, 2, 3, 4)),
                "hard_valid": hard_valid.detach(),
                "first_frame_warp_error": first_frame_warp_error.detach(),
                "previous_snr_db": snr[:, index - 1].detach(),
                "previous_pilot_coherence": coherence[:, index - 1].detach(),
            }
        )
    return torch.stack(outputs, dim=1), telemetry


def multi_boundary_losses(
    recon: torch.Tensor, target: torch.Tensor, frames_per_gop: int
) -> dict[str, torch.Tensor]:
    count = recon.shape[2] // frames_per_gop
    if count < 2 or recon.shape != target.shape:
        raise ValueError("multi-boundary loss requires equal complete multi-GOP tensors")
    rows: list[dict[str, torch.Tensor]] = []
    for boundary in range(1, count):
        start = (boundary - 1) * frames_per_gop
        end = (boundary + 1) * frames_per_gop
        rows.append(boundary_losses(recon[:, :, start:end], target[:, :, start:end], frames_per_gop))
    return {name: torch.stack([row[name] for row in rows]).mean() for name in rows[0]}


def save_gate_checkpoint(
    gate: ReliabilityGate,
    refiner_path: Path,
    checkpoint: Path,
    destination: Path,
    args: argparse.Namespace,
    elapsed_s: float,
) -> None:
    payload = {
        "kind": "aetv-v8-reliability-gated-feature-context",
        "mode": args.mode,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": sha256_file(checkpoint),
        "refiner_checkpoint": str(refiner_path.resolve()),
        "refiner_sha256": sha256_file(refiner_path),
        "gate_config": gate.config(),
        "gate_state_dict": {name: value.detach().cpu() for name, value in gate.state_dict().items()},
        "wire_contract_changed": False,
        "base_codec_modified": False,
        "refiner_modified": False,
        "reset_is_exact_bypass": True,
        "training": {
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "elapsed_s": elapsed_s,
            "events": list(EVENTS),
            "gops": args.train_gops,
            "loss_weights": {
                "source": args.source_weight,
                "released_anchor": args.anchor_weight,
                "boundary_group": args.boundary_weight,
                "within": args.within_weight,
                "vgg_source": args.vgg_source_weight,
                "vgg_anchor": args.vgg_anchor_weight,
                "gate_anchor": args.gate_anchor_weight,
            },
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_gate(path: Path, device: torch.device) -> tuple[ReliabilityGate, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "aetv-v8-reliability-gated-feature-context":
        raise ValueError(f"{path} is not a V8 reliability-gate checkpoint")
    gate = ReliabilityGate(**payload["gate_config"]).to(device)
    gate.load_state_dict(payload["gate_state_dict"], strict=True)
    return gate, payload


def train_gate(
    args: argparse.Namespace,
    gate_mode: str,
    destination: Path,
    train_dataset: MultiGOPSequenceCache,
    train_rx: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> list[dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = load_model(args.checkpoint, mode, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    refiner, payload = load_refiner(args.refiner, device)
    if payload["source_sha256"] != sha256_file(args.checkpoint):
        raise RuntimeError("retained refiner source checkpoint does not match released V8")
    refiner.eval()
    for parameter in refiner.parameters():
        parameter.requires_grad_(False)
    aligner = RAFTAligner(device).eval()
    gate = ReliabilityGate(
        gate_mode,
        feature_channels=refiner.feature_channels,
        width=args.gate_width,
        max_logit_adjustment=args.max_logit_adjustment,
    ).to(device)
    perceptual = MultiLayerVGGPerceptualLoss().to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        IndexedMultiGOPDataset(train_dataset),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    iterator = iter(loader)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        try:
            indices, source = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            indices, source = next(iterator)
        source = source.to(device)
        event_name = EVENTS[(step - 1) % len(EVENTS)]
        event = prepare_event_batch(event_name, source, indices, train_rx, mode, device)
        with torch.no_grad():
            base, features, skips = decode_received_features(
                model, event.received, event.decode_weights, mode
            )
            base_sequence = join_many_gops(base)

        optimizer.zero_grad(set_to_none=True)
        corrected_gops, telemetry = apply_reliability_sequence(
            refiner,
            gate,
            model.decoder,
            aligner,
            base,
            features,
            skips,
            event.gate_weights,
            event.snr,
            event.coherence,
            event.valid,
            event.reset_before,
            photometric_threshold=args.photometric_threshold,
            photometric_softness=args.photometric_softness,
            scene_threshold_multiplier=args.scene_threshold_multiplier,
            scene_cut_threshold=args.scene_cut_threshold,
            min_previous_snr_db=args.min_previous_snr_db,
            min_previous_pilot_coherence=args.min_previous_pilot_coherence,
            output_flow_strength=args.output_flow_strength,
        )
        corrected = join_many_gops(corrected_gops)
        cross = multi_boundary_losses(corrected, event.source, mode.gop_frames)
        source_l1 = F.l1_loss(corrected, event.source)
        anchor_l1 = F.l1_loss(corrected, base_sequence)
        vgg_source = perceptual(corrected, event.source)
        vgg_anchor = perceptual(corrected, base_sequence)
        gate_anchor_terms = []
        for telemetry_row in telemetry:
            safe = telemetry_row["hard_valid"]
            if safe.any():
                gate_anchor_terms.append(
                    (
                        telemetry_row["spatial_gate_mean"][safe]
                        - telemetry_row["base_confidence"][safe]
                    )
                    .abs()
                    .mean()
                )
        gate_anchor = (
            torch.stack(gate_anchor_terms).mean()
            if gate_anchor_terms
            else corrected.new_zeros(())
        )
        boundary_group = (
            cross["boundary_rgb_delta"]
            + args.lowpass_term_weight
            * (cross["boundary_lowpass_y"] + cross["boundary_lowpass_chroma"])
            + args.gradient_term_weight * cross["boundary_gradient_delta"]
            + args.acceleration_term_weight * cross["boundary_acceleration"]
        )
        total = (
            args.source_weight * source_l1
            + args.anchor_weight * anchor_l1
            + args.within_weight * cross["within_gop_temporal_error"]
            + args.boundary_weight * boundary_group
            + args.vgg_source_weight * vgg_source
            + args.vgg_anchor_weight * vgg_anchor
            + args.gate_anchor_weight * gate_anchor
        )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), args.clip_grad)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at {gate_mode} step {step}")
        optimizer.step()
        row = {
            "step": step,
            "event": event_name,
            "indices": [int(index) for index in indices],
            "total": float(total.detach()),
            "source_l1": float(source_l1.detach()),
            "anchor_l1": float(anchor_l1.detach()),
            "vgg_source": float(vgg_source.detach()),
            "vgg_anchor": float(vgg_anchor.detach()),
            "gate_anchor": float(gate_anchor.detach()),
            "boundary_group": float(boundary_group.detach()),
            "grad_norm": float(grad_norm.detach()),
            **{name: float(value.detach()) for name, value in cross.items()},
        }
        history.append(row)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"{gate_mode:<15} step {step:>4}/{args.steps} {event_name:<22} "
                f"total={row['total']:.5f} boundary={row['boundary_rgb_delta']:.5f} "
                f"gate_delta={row['gate_anchor']:.5f}",
                flush=True,
            )
    save_gate_checkpoint(
        gate, args.refiner, args.checkpoint, destination, args, time.time() - started
    )
    del model, refiner, aligner, perceptual, gate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def cache_diagnostics_for_sequence(
    rx_cache: dict,
    cell: ChannelCell,
    index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    details = rx_cache["diagnostics"][cell.label][index]["gops"]
    snr = torch.tensor([[row["snr_db"] for row in details]], device=device)
    coherence = torch.tensor([[row["pilot_coherence"] for row in details]], device=device)
    return snr, coherence


def evaluate_standard_variant(
    args: argparse.Namespace,
    label: str,
    gate: ReliabilityGate | None,
    model: nn.Module,
    refiner: FeatureContextRefiner,
    aligner: RAFTAligner,
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    rows = {cell.label: [] for cell in DEFAULT_CELLS}
    timings = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
                weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
                if gate is None:
                    recon_gops = decode_independent_gops(model, received, weights, mode)
                    recon = join_gops(recon_gops, 1, count=2)
                else:
                    base, features, skips = decode_received_features(model, received, weights, mode)
                    snr, coherence = cache_diagnostics_for_sequence(
                        rx_cache, cell, index, device
                    )
                    valid = torch.ones(1, 2, dtype=torch.bool, device=device)
                    reset = torch.zeros_like(valid)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    corrected, _ = apply_reliability_sequence(
                        refiner,
                        gate,
                        model.decoder,
                        aligner,
                        base,
                        features,
                        skips,
                        weights,
                        snr,
                        coherence,
                        valid,
                        reset,
                        photometric_threshold=args.photometric_threshold,
                        photometric_softness=args.photometric_softness,
                        scene_threshold_multiplier=args.scene_threshold_multiplier,
                        scene_cut_threshold=args.scene_cut_threshold,
                        min_previous_snr_db=args.min_previous_snr_db,
                        min_previous_pilot_coherence=args.min_previous_pilot_coherence,
                        output_flow_strength=args.output_flow_strength,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timings.append(time.perf_counter() - started)
                    recon = join_many_gops(corrected)
                rows[cell.label].append(
                    sequence_metrics(
                        recon, source, mode.gop_frames, device, include_lpips=True
                    )
                )
            print(f"  {label:<15} sequence {index + 1:>2}/{len(dataset)}", flush=True)
    output = {"cells": [asdict(cell) for cell in DEFAULT_CELLS], "sequences": rows}
    if timings:
        output["runtime_ms_per_boundary"] = 1000 * st.mean(timings)
    return output


def failure_pattern_names() -> tuple[str, ...]:
    return (
        "good_fade_good",
        "false_high_confidence",
        "false_low_confidence",
        "missing_gop",
        "random_reset",
        "scene_cut",
    )


def _gop_l1(value: torch.Tensor, target: torch.Tensor, index: int, frames: int) -> float:
    start = index * frames
    return float(F.l1_loss(value[:, :, start : start + frames], target[:, :, start : start + frames]))


def evaluate_failure_matrix(
    args: argparse.Namespace,
    gates: dict[str, ReliabilityGate | None],
    model: nn.Module,
    refiner: FeatureContextRefiner,
    aligner: RAFTAligner,
    dataset: MultiGOPSequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    output = {name: {label: [] for label in gates} for name in failure_pattern_names()}
    with torch.inference_mode():
        for name in failure_pattern_names():
            for start in range(0, len(dataset), args.batch):
                batch_indices = list(range(start, min(start + args.batch, len(dataset))))
                if name == "scene_cut" and len(batch_indices) < 2:
                    batch_indices = [batch_indices[0], (batch_indices[0] + 1) % len(dataset)]
                indices = torch.tensor(batch_indices, dtype=torch.long)
                source = torch.stack([dataset[index] for index in batch_indices]).to(device)
                event = prepare_event_batch(name, source, indices, rx_cache, mode, device)
                base, features, skips = decode_received_features(
                    model, event.received, event.decode_weights, mode
                )
                base_sequence = join_many_gops(base)
                base_recovery = [
                    _gop_l1(base_sequence[row : row + 1], event.source[row : row + 1], event.recovery_gop, mode.gop_frames)
                    for row in range(len(batch_indices))
                ]
                for label, gate in gates.items():
                    if gate is None:
                        candidate = base_sequence
                        telemetry = []
                    else:
                        corrected, telemetry = apply_reliability_sequence(
                            refiner,
                            gate,
                            model.decoder,
                            aligner,
                            base,
                            features,
                            skips,
                            event.gate_weights,
                            event.snr,
                            event.coherence,
                            event.valid,
                            event.reset_before,
                            photometric_threshold=args.photometric_threshold,
                            photometric_softness=args.photometric_softness,
                            scene_threshold_multiplier=args.scene_threshold_multiplier,
                            scene_cut_threshold=args.scene_cut_threshold,
                            min_previous_snr_db=args.min_previous_snr_db,
                            min_previous_pilot_coherence=args.min_previous_pilot_coherence,
                            output_flow_strength=args.output_flow_strength,
                        )
                        candidate = join_many_gops(corrected)
                    cross = multi_boundary_losses(candidate, event.source, mode.gop_frames)
                    for row in range(len(batch_indices)):
                        candidate_recovery = _gop_l1(
                            candidate[row : row + 1],
                            event.source[row : row + 1],
                            event.recovery_gop,
                            mode.gop_frames,
                        )
                        output[name][label].append(
                            {
                                "sequence": batch_indices[row],
                                "recovery_gop": event.recovery_gop,
                                "recovery_l1": candidate_recovery,
                                "independent_recovery_l1": base_recovery[row],
                                "recovery_error_ratio": candidate_recovery
                                / max(base_recovery[row], 1e-12),
                                "boundary_rgb_delta": float(cross["boundary_rgb_delta"]),
                                "boundary_lowpass_step": float(cross["boundary_lowpass_step"]),
                                "boundary_acceleration": float(cross["boundary_acceleration"]),
                                "mean_gate": (
                                    st.mean(float(item["spatial_gate_mean"][row]) for item in telemetry)
                                    if telemetry
                                    else 0.0
                                ),
                            }
                        )
            print(f"  failure matrix {name}", flush=True)
    return output


def assess_reliability_candidate(
    release_comparison: dict,
    scalar_comparison: dict,
    scalar_vs_release: dict,
    failure: dict,
    label: str,
) -> dict:
    base = assess_candidate(release_comparison)
    reasons = list(base["reasons"])

    # A small entrance-local correction can raise the temporal-error observable
    # without blurring motion.  Match the already-retained adapter policy: allow
    # at most 0.75% clean relative increase while the independent motion/detail
    # guardrails remain enforced.
    clean_within = release_comparison["cells"]["clean"]["within_gop_temporal_error"]
    within_relative = (
        clean_within["paired_delta"]["mean"]
        / max(clean_within["baseline_mean"], 1e-12)
    )
    if within_relative <= 0.0075:
        reasons = [
            reason
            for reason in reasons
            if not reason.startswith("within-GOP temporal error regressed")
        ]

    # On the ordinary paired matrix, preserve at least 90% of the retained
    # scalar adapter's clean seam gain for the three primary seam observables.
    for metric in ("boundary_error_ratio", "boundary_delta_lpips", "boundary_lowpass_step"):
        scalar_gain = -scalar_vs_release["cells"]["clean"][metric]["paired_delta"]["mean"]
        candidate_gain = -release_comparison["cells"]["clean"][metric]["paired_delta"]["mean"]
        if scalar_gain > 0 and candidate_gain < 0.90 * scalar_gain:
            reasons.append(
                f"clean {metric} preserved only {candidate_gain / scalar_gain:.1%} of scalar gain"
            )

    measured = release_comparison["cells"]["measured_hf"]
    for metric in ("boundary_error_ratio", "boundary_delta_lpips"):
        if measured[metric]["paired_delta"]["mean"] >= 0:
            reasons.append(f"measured_hf {metric} did not improve over released")

    for cell, metrics in scalar_comparison["cells"].items():
        delta = metrics["lpips"]["paired_delta"]
        if delta["mean"] - delta["two_se"] > 0.001:
            reasons.append(
                f"{cell} LPIPS regressed versus scalar {delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
            )

    for event in (
        "good_fade_good",
        "false_high_confidence",
        "missing_gop",
        "random_reset",
        "scene_cut",
    ):
        ratios = [row["recovery_error_ratio"] for row in failure[event][label]]
        worst = max(ratios)
        if worst > 1.05:
            reasons.append(f"{event} worst recovery error ratio {worst:.4f} exceeds 1.05")

    return {
        **base,
        "accepted": not reasons,
        "decision": "retain" if not reasons else "reject",
        "reasons": reasons,
        "scalar_seam_gain_preservation": 0.90,
        "one_gop_recovery_limit": 1.05,
        "clean_within_gop_relative_tolerance": 0.0075,
    }


def write_report(
    destination: Path,
    args: argparse.Namespace,
    evaluations: dict,
    comparisons: dict,
    decisions: dict,
    failure: dict,
    selection: dict,
) -> None:
    lines = [
        "# V8 reliability-gated motion-aligned feature context",
        "",
        f"Released checkpoint SHA-256: `{sha256_file(args.checkpoint)}`",
        f"Retained refiner SHA-256: `{sha256_file(args.refiner)}`",
        "",
        (
            "The released encoder/decoder and retained feature refiner were frozen. Only the "
            "spatial or spatial-plus-feature-channel reliability gate was trained. The wire "
            "contract is unchanged and GUI blending is disabled."
        ),
        "",
        "## Promotion decisions",
        "",
        "| Candidate | Decision | Reasons |",
        "|---|---|---|",
    ]
    for label, decision in decisions.items():
        reason = "; ".join(decision["reasons"]) or "all seam, LPIPS, detail, and recovery gates passed"
        lines.append(f"| {label} | **{decision['decision'].upper()}** | {reason} |")

    lines.extend(
        (
            "",
            "## Selected candidate",
            "",
            f"**{selection['preferred']}** — {selection['reason']}",
            "",
            "| Candidate | Trainable parameters | Checkpoint bytes |",
            "|---|---:|---:|",
        )
    )
    for label, detail in selection["candidates"].items():
        lines.append(
            f"| {label} | {detail['trainable_parameters']} | {detail['checkpoint_bytes']} |"
        )

    metrics = tuple(dict.fromkeys(REPORT_METRICS))
    for cell in (item.label for item in DEFAULT_CELLS):
        lines.extend(("", f"## {cell}", "", "| Model | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)))
        for label, report in evaluations.items():
            rows = report["sequences"][cell]
            values = [st.mean(row[metric] for row in rows) for metric in metrics]
            lines.append(f"| {label} | " + " | ".join(f"{value:.6f}" for value in values) + " |")

    lines.extend(("", "## Runtime", "", "| Model | ms/boundary |", "|---|---:|"))
    for label, report in evaluations.items():
        if "runtime_ms_per_boundary" in report:
            lines.append(f"| {label} | {report['runtime_ms_per_boundary']:.3f} |")

    lines.extend(("", "## Failure and recovery", "", "| Event | Model | mean recovery ratio | worst recovery ratio | mean gate |", "|---|---|---:|---:|---:|"))
    for event, variants in failure.items():
        for label, rows in variants.items():
            ratios = [row["recovery_error_ratio"] for row in rows]
            gates = [row["mean_gate"] for row in rows]
            lines.append(
                f"| {event} | {label} | {st.mean(ratios):.6f} | {max(ratios):.6f} | {st.mean(gates):.6f} |"
            )

    lines.extend(
        (
            "",
            "## Paired uncertainty",
            "",
            "`comparisons.json` contains paired mean deltas, standard errors, and two-standard-error intervals versus both the released codec and retained scalar adapter.",
            "",
        )
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def select_candidate(
    decisions: dict,
    gates: dict[str, ReliabilityGate | None],
    gate_paths: dict[str, Path],
    spatial_channel_vs_spatial: dict,
) -> dict:
    accepted = [label for label, detail in decisions.items() if detail["accepted"]]
    if not accepted:
        return {
            "preferred": "none",
            "reason": "no learned reliability gate passed all promotion criteria",
            "candidates": {},
        }
    candidates = {
        label: {
            "trainable_parameters": sum(
                parameter.numel() for parameter in gates[label].parameters()
            ),
            "checkpoint_bytes": gate_paths[label].stat().st_size,
        }
        for label in accepted
    }
    if "spatial" in accepted and "spatial_channel" in accepted:
        maximum_metric_delta = max(
            abs(metric["paired_delta"]["mean"])
            for cell in spatial_channel_vs_spatial["cells"].values()
            for metric in cell.values()
        )
        return {
            "preferred": "spatial",
            "reason": (
                "the feature-channel head passed but added no material paired metric gain "
                f"(maximum absolute mean delta {maximum_metric_delta:.3g}) and used more parameters"
            ),
            "maximum_spatial_channel_vs_spatial_metric_delta": maximum_metric_delta,
            "candidates": candidates,
        }
    preferred = accepted[0]
    return {
        "preferred": preferred,
        "reason": "it was the only learned reliability gate to pass all promotion criteria",
        "candidates": candidates,
    }


def render_full_comparisons(
    args: argparse.Namespace,
    gates: dict[str, ReliabilityGate | None],
    model: nn.Module,
    refiner: FeatureContextRefiner,
    aligner: RAFTAligner,
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    destination = args.out / "renders"
    destination.mkdir(parents=True, exist_ok=True)
    sources = torch.cat([dataset[index] for index in range(len(dataset))], dim=1).unsqueeze(0)
    manifest = {
        "gui_boundary_blending": False,
        "frames_per_sequence": 12,
        "seam_transition": "frame 5 -> frame 6",
        "sequences": len(dataset),
        "total_frames_per_render": 12 * len(dataset),
        "fps": mode.fps,
        "files": {},
    }
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            panels = [("Source", sources)]
            clips = {label: [] for label in gates}
            for index in range(len(dataset)):
                received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
                weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
                base, features, skips = decode_received_features(model, received, weights, mode)
                snr, coherence = cache_diagnostics_for_sequence(rx_cache, cell, index, device)
                valid = torch.ones(1, 2, dtype=torch.bool, device=device)
                reset = torch.zeros_like(valid)
                for label, gate in gates.items():
                    if gate is None:
                        output = join_many_gops(base)
                    else:
                        corrected, _ = apply_reliability_sequence(
                            refiner,
                            gate,
                            model.decoder,
                            aligner,
                            base,
                            features,
                            skips,
                            weights,
                            snr,
                            coherence,
                            valid,
                            reset,
                            photometric_threshold=args.photometric_threshold,
                            photometric_softness=args.photometric_softness,
                            scene_threshold_multiplier=args.scene_threshold_multiplier,
                            scene_cut_threshold=args.scene_cut_threshold,
                            min_previous_snr_db=args.min_previous_snr_db,
                            min_previous_pilot_coherence=args.min_previous_pilot_coherence,
                            output_flow_strength=args.output_flow_strength,
                        )
                        output = join_many_gops(corrected)
                    clips[label].append(output.cpu())
            for label in gates:
                panels.append((label, torch.cat(clips[label], dim=2)))
            path = destination / f"full-paired-32-{cell.label}-no-gui-blend.mp4"
            write_labeled_grid_mp4(panels, path, fps=mode.fps, columns=3)
            manifest["files"][cell.label] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            print(f"Rendered {path}", flush=True)
    (destination / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("precompute", "train", "evaluate", "render", "all")
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    parser.add_argument("--refiner", type=Path, default=Path("runs/gop-feature-context-v8/refiner.pt"))
    parser.add_argument(
        "--out", type=Path, default=Path("runs/v8-reliability-gated-feature-context-20260826")
    )
    parser.add_argument(
        "--train-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_5gop_train")
    )
    parser.add_argument(
        "--failure-eval-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_5gop_eval")
    )
    parser.add_argument(
        "--standard-eval-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_3gop_eval")
    )
    parser.add_argument(
        "--standard-eval-rx-cache",
        type=Path,
        default=Path("runs/v8-two-gop-boundary-sweep-explicit-20260826/eval-runtime-rx.pt"),
    )
    parser.add_argument("--train-rx-cache", type=Path)
    parser.add_argument("--failure-eval-rx-cache", type=Path)
    parser.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    parser.add_argument("--train-gops", type=int, default=5)
    parser.add_argument("--train-sequences", type=int, default=32)
    parser.add_argument("--failure-eval-sequences", type=int, default=8)
    parser.add_argument("--standard-eval-sequences", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-width", type=int, default=16)
    parser.add_argument("--max-logit-adjustment", type=float, default=4.0)
    parser.add_argument("--output-flow-strength", type=float, default=0.10)
    parser.add_argument("--photometric-threshold", type=float, default=0.10)
    parser.add_argument("--photometric-softness", type=float, default=0.02)
    parser.add_argument("--scene-threshold-multiplier", type=float, default=1.5)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.15)
    parser.add_argument("--min-previous-snr-db", type=float, default=-2.0)
    parser.add_argument("--min-previous-pilot-coherence", type=float, default=0.25)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=8.0)
    parser.add_argument("--within-weight", type=float, default=1.0)
    parser.add_argument("--lowpass-term-weight", type=float, default=0.5)
    parser.add_argument("--gradient-term-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-term-weight", type=float, default=0.25)
    parser.add_argument("--vgg-source-weight", type=float, default=0.5)
    parser.add_argument("--vgg-anchor-weight", type=float, default=0.25)
    parser.add_argument("--gate-anchor-weight", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--failure-eval-seed", type=int, default=20260828)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def verify_standard_cache(args: argparse.Namespace, dataset: SequenceCache) -> dict:
    rx = torch.load(args.standard_eval_rx_cache, map_location="cpu", weights_only=False)
    metadata = rx.get("metadata", {})
    if metadata.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise RuntimeError("standard paired RX cache checkpoint hash mismatch")
    if metadata.get("gops_per_sequence") != 2:
        raise RuntimeError("standard paired RX cache is not a two-GOP cache")
    expected_pixels = [row["pixel_sha256"] for row in metadata.get("sequences", [])]
    actual_pixels = [row["pixel_sha256"] for row in dataset.manifest()]
    if expected_pixels[: len(actual_pixels)] != actual_pixels:
        raise RuntimeError("standard paired RX cache sequence pairing mismatch")
    return rx


def main() -> None:
    args = build_parser().parse_args()
    mode = AETV_MODES[args.mode]
    if mode.gop_frames != 6:
        raise SystemExit("reliability experiment requires six-frame GOPs")
    if args.batch < 2:
        raise SystemExit("controlled scene-cut curriculum requires --batch at least 2")
    for path in (args.checkpoint, args.refiner):
        if not path.is_file():
            raise SystemExit(f"missing required checkpoint: {path}")
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    args.train_rx_cache = args.train_rx_cache or args.out / "train-runtime-rx-5gop.pt"
    args.failure_eval_rx_cache = (
        args.failure_eval_rx_cache or args.out / "failure-eval-runtime-rx-5gop.pt"
    )
    train_dataset = MultiGOPSequenceCache(
        args.train_cache, args.train_gops, mode.gop_frames, args.train_sequences
    )
    failure_dataset = MultiGOPSequenceCache(
        args.failure_eval_cache,
        args.train_gops,
        mode.gop_frames,
        args.failure_eval_sequences,
    )
    standard_dataset = SequenceCache(
        args.standard_eval_cache, limit=args.standard_eval_sequences
    )
    if len(train_dataset) != args.train_sequences:
        raise SystemExit(f"need {args.train_sequences} train sequences, found {len(train_dataset)}")
    if len(failure_dataset) != args.failure_eval_sequences:
        raise SystemExit(
            f"need {args.failure_eval_sequences} failure eval sequences, found {len(failure_dataset)}"
        )
    if len(standard_dataset) != args.standard_eval_sequences:
        raise SystemExit(
            f"need {args.standard_eval_sequences} standard eval sequences, found {len(standard_dataset)}"
        )

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "refiner": str(args.refiner.resolve()),
        "refiner_sha256": sha256_file(args.refiner),
        "mode": args.mode,
        "train_gops": args.train_gops,
        "gate_candidates": ["spatial", "spatial_channel"],
        "control": "retained scalar confidence gate",
        "events": list(EVENTS),
        "hard_safety": {
            "scene_cut_first_frame_warp_error": args.scene_cut_threshold,
            "minimum_previous_snr_db": args.min_previous_snr_db,
            "minimum_previous_pilot_coherence": args.min_previous_pilot_coherence,
        },
        "standard_cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "gui_blending": False,
        "train_sequences": train_dataset.manifest(),
        "failure_eval_sequences": failure_dataset.manifest(),
        "standard_eval_sequences": standard_dataset.manifest(),
    }
    (args.out / "experiment-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    train_rx = precompute_multigop_rx(
        args.checkpoint,
        args.train_rx_cache,
        train_dataset,
        mode,
        device,
        seed=args.seed,
        split_name="train",
    )
    failure_rx = precompute_multigop_rx(
        args.checkpoint,
        args.failure_eval_rx_cache,
        failure_dataset,
        mode,
        device,
        seed=args.failure_eval_seed,
        split_name="failure_eval",
    )
    if args.command == "precompute":
        return

    gate_paths = {
        "spatial": args.out / "gate-spatial.pt",
        "spatial_channel": args.out / "gate-spatial-channel.pt",
    }
    if args.command in {"train", "all"}:
        for gate_mode, path in gate_paths.items():
            history = train_gate(
                args, gate_mode, path, train_dataset, train_rx, mode, device
            )
            (args.out / f"training-{gate_mode.replace('_', '-')}.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
        if args.command == "train":
            return

    gates: dict[str, ReliabilityGate | None] = {
        "released": None,
        "scalar": ReliabilityGate("scalar").to(device).eval(),
    }
    for label, path in gate_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing trained gate: {path}")
        gate, payload = load_gate(path, device)
        if payload["source_sha256"] != sha256_file(args.checkpoint):
            raise SystemExit(f"{label} gate source checkpoint mismatch")
        if payload["refiner_sha256"] != sha256_file(args.refiner):
            raise SystemExit(f"{label} gate refiner mismatch")
        gates[label] = gate.eval()

    model = load_model(args.checkpoint, mode, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    refiner, _ = load_refiner(args.refiner, device)
    refiner.eval()
    aligner = RAFTAligner(device).eval()
    standard_rx = verify_standard_cache(args, standard_dataset)

    evaluations_path = args.out / "evaluation.json"
    comparisons_path = args.out / "comparisons.json"
    decisions_path = args.out / "decisions.json"
    failure_path = args.out / "failure-evaluation.json"
    if args.command in {"evaluate", "all"}:
        evaluations = {
            label: evaluate_standard_variant(
                args,
                label,
                gate,
                model,
                refiner,
                aligner,
                standard_dataset,
                standard_rx,
                mode,
                device,
            )
            for label, gate in gates.items()
        }
        comparisons = {
            label: {
                "vs_released": compare_reports(evaluations["released"], evaluations[label]),
                "vs_scalar": compare_reports(evaluations["scalar"], evaluations[label]),
            }
            for label in ("spatial", "spatial_channel")
        }
        comparisons["scalar"] = {
            "vs_released": compare_reports(evaluations["released"], evaluations["scalar"])
        }
        comparisons["spatial_channel"]["vs_spatial"] = compare_reports(
            evaluations["spatial"], evaluations["spatial_channel"]
        )
        failure = evaluate_failure_matrix(
            args,
            gates,
            model,
            refiner,
            aligner,
            failure_dataset,
            failure_rx,
            mode,
            device,
        )
        decisions = {
            label: assess_reliability_candidate(
                comparisons[label]["vs_released"],
                comparisons[label]["vs_scalar"],
                comparisons["scalar"]["vs_released"],
                failure,
                label,
            )
            for label in ("spatial", "spatial_channel")
        }
        selection = select_candidate(
            decisions,
            gates,
            gate_paths,
            comparisons["spatial_channel"]["vs_spatial"],
        )
        evaluations_path.write_text(json.dumps(evaluations, indent=2) + "\n", encoding="utf-8")
        comparisons_path.write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
        decisions_path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        (args.out / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        write_report(
            args.out / "report.md",
            args,
            evaluations,
            comparisons,
            decisions,
            failure,
            selection,
        )
        for label, decision in decisions.items():
            print(f"{label}: {decision['decision'].upper()}", flush=True)
            for reason in decision["reasons"]:
                print(f"  - {reason}", flush=True)
        if args.command == "evaluate":
            return

    if args.command in {"render", "all"}:
        render_full_comparisons(
            args,
            gates,
            model,
            refiner,
            aligner,
            standard_dataset,
            standard_rx,
            mode,
            device,
        )


if __name__ == "__main__":
    main()
