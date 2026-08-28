#!/usr/bin/env python3
"""Controlled two-GOP boundary fine-tune and paired V8 evaluation.

Every example is exactly 12 contiguous frames.  Frames 0..5 and 6..11 are
encoded independently, carried through the same continuous TX conditioner,
stateful GUI channel emulator, and streaming demodulator used by station
loopback, then decoded independently.  The two reconstructions are joined only
for source-referenced losses and metrics.

The ``all`` command precomputes immutable received-latent caches, trains the
1x/2x/4x/8x boundary-loss sweep from the released checkpoint, evaluates the
baseline and every candidate on the same 32 sequences/channel realizations,
applies LPIPS/motion-blur rejection gates, and renders all 384 held-out frames
per channel with GUI boundary blending disabled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics as st
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES, AETVModeSpec  # noqa: E402
from aetv.hfchannel import (  # noqa: E402
    CHANNEL_PROFILES,
    ChannelProfile,
    StreamingChannelEmulator,
)
from aetv.models import AETVAutoencoder  # noqa: E402
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402
from train import (  # noqa: E402
    lpips_metric,
    simulate_transmission as simulate_transmission,
)


FRAMES_PER_EXAMPLE = 12
TX_LEVEL = 0.7
MEASURED_PROFILE = ChannelProfile(
    "measured_hf_ota40m_5db",
    "Measured OTA40m 5 dB",
    5.0,
    "ota40m",
    "K9CZI-1-like measured 40 m path: 0.6 ms / 0.24 Hz at 5 dB",
)


@dataclass(frozen=True)
class ChannelCell:
    label: str
    profile_key: str
    snr_db: float | None
    fading: str


DEFAULT_CELLS = (
    ChannelCell("clean", "clean", None, "none"),
    ChannelCell("awgn_6db", "awgn6", 6.0, "none"),
    ChannelCell("mpp_12db", "mpp12", 12.0, "mpp"),
    ChannelCell("measured_hf", MEASURED_PROFILE.key, 5.0, "ota40m"),
)


def cache_name(mode: AETVModeSpec, gops: int, split: str) -> str:
    """Return the historical sequence-cache directory name used by GOP experiments."""
    if gops < 1:
        raise ValueError("gops must be positive")
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    return f"{mode.name.lower()}_{mode.width}x{mode.height}_{gops}gop_{split}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class SequenceCache(Dataset):
    """Read the first 12 contiguous frames from a fixed sequence cache."""

    def __init__(self, root: Path, *, limit: int | None = None, max_frames: int = FRAMES_PER_EXAMPLE):
        self.root = root
        self.max_frames = max_frames
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
        value = torch.load(self.files[index], map_location="cpu").float()
        if value.ndim != 4 or value.shape[0] != 3:
            raise ValueError(f"expected CTHW tensor in {self.files[index]}, got {tuple(value.shape)}")
        if value.shape[1] < self.max_frames:
            raise ValueError(
                f"{self.files[index]} has {value.shape[1]} frames; need {self.max_frames}"
            )
        value = value[:, :self.max_frames]
        if value.max() > 1.0:
            value = value.div(255.0)
        return value.contiguous()

    def manifest(self) -> list[dict]:
        records = []
        for index in range(len(self)):
            value = self[index]
            records.append(
                {
                    "index": index,
                    "file": str(self.files[index].resolve()),
                    "source_file_sha256": sha256_file(self.files[index]),
                    "frame_slice": [0, FRAMES_PER_EXAMPLE],
                    "pixel_sha256": tensor_sha256(value),
                    "shape": list(value.shape),
                }
            )
        return records


class IndexedDataset(Dataset):
    def __init__(self, dataset: SequenceCache):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return index, self.dataset[index]


def checkpoint_config(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not an AETV checkpoint dictionary")
    saved = payload.get("args", {}) or {}
    if hasattr(saved, "__dict__"):
        saved = vars(saved)
    return payload, dict(saved)


def load_model(path: Path, mode: AETVModeSpec, device: torch.device) -> AETVAutoencoder:
    payload, saved = checkpoint_config(path)
    checkpoint_mode = payload.get("mode") or saved.get("mode")
    if checkpoint_mode and checkpoint_mode != mode.name:
        raise ValueError(f"checkpoint mode {checkpoint_mode} does not match --mode {mode.name}")
    model = AETVAutoencoder(
        mode=mode,
        width=int(saved.get("model_width", 128)),
        latent_channels=int(saved.get("latent_channels", 3)),
        compact=bool(saved.get("compact", False)),
        causal=mode.causal,
    ).to(device)
    state = payload.get("model_state_dict") or payload.get("model")
    if state is None:
        raise KeyError(f"{path} has no model state")
    model.load_state_dict(state, strict=True)
    return model


def split_gops(sequence: torch.Tensor, frames_per_gop: int) -> torch.Tensor:
    """Convert B,C,G*T,H,W to B*G,C,T,H,W without sharing model state."""
    if sequence.ndim != 5:
        raise ValueError(f"expected BCTHW sequence, got {tuple(sequence.shape)}")
    batch, channels, frames, height, width = sequence.shape
    if frames != FRAMES_PER_EXAMPLE:
        raise ValueError(f"experiment requires exactly {FRAMES_PER_EXAMPLE} frames, got {frames}")
    if frames % frames_per_gop:
        raise ValueError(f"{frames} frames is not divisible by GOP length {frames_per_gop}")
    count = frames // frames_per_gop
    if count != 2:
        raise ValueError(f"experiment requires two GOPs, got {count}")
    return (
        sequence.reshape(batch, channels, count, frames_per_gop, height, width)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * count, channels, frames_per_gop, height, width)
    )


def join_gops(gops: torch.Tensor, batch: int, count: int = 2) -> torch.Tensor:
    """Convert B*G,C,T,H,W back to B,C,G*T,H,W."""
    _, channels, frames, height, width = gops.shape
    return (
        gops.reshape(batch, count, channels, frames, height, width)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(batch, channels, count * frames, height, width)
    )


def encode_independent_gops(
    model: AETVAutoencoder,
    separated: torch.Tensor,
) -> torch.Tensor:
    """Use one encoder invocation per GOP, matching the live codec contract."""
    if separated.shape[0] != 2:
        raise ValueError(f"expected two GOP samples, got {separated.shape[0]}")
    return torch.cat(
        (model.encoder(separated[0:1]), model.encoder(separated[1:2])),
        dim=0,
    )


def decode_independent_gops(
    model: AETVAutoencoder,
    received: torch.Tensor,
    weights: torch.Tensor,
    mode: AETVModeSpec,
) -> torch.Tensor:
    """Use one decoder invocation per GOP position for every batch example."""
    if received.ndim != 3 or received.shape[1] != 2:
        raise ValueError(f"expected B,2,N received latents, got {tuple(received.shape)}")
    if weights.shape != received.shape:
        raise ValueError("received latent/confidence shapes differ")
    decoded = [
        model.decoder(
            received[:, index],
            weights[:, index],
            output_shape=(mode.gop_frames, mode.height, mode.width),
        )
        for index in range(2)
    ]
    return torch.stack(decoded, dim=1).flatten(0, 1)


def boundary_indices(total_frames: int, frames_per_gop: int) -> list[int]:
    return list(range(frames_per_gop, total_frames, frames_per_gop))


def temporal_delta_error(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (recon[:, :, 1:] - recon[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])


def rgb_to_ycbcr_delta(value: torch.Tensor) -> torch.Tensor:
    """Linear RGB-to-YCbCr transform for signed differences (no offsets)."""
    red, green, blue = value.unbind(dim=1)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue
    return torch.stack((y, cb, cr), dim=1)


def lowpass_video(value: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    batch, channels, frames, height, width = value.shape
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    pooled = F.avg_pool2d(flat, kernel_size, stride=1, padding=kernel_size // 2)
    return pooled.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def spatial_gradient_l1(value: torch.Tensor) -> torch.Tensor:
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def spatial_gradient_energy(value: torch.Tensor) -> torch.Tensor:
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def boundary_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    frames_per_gop: int,
) -> dict[str, torch.Tensor]:
    """Source-referenced 5->6 losses plus within-GOP guardrails."""
    boundaries = boundary_indices(recon.shape[2], frames_per_gop)
    if boundaries != [frames_per_gop]:
        raise ValueError(f"expected one two-GOP boundary, got {boundaries}")
    boundary = boundaries[0]
    delta_error = temporal_delta_error(recon, target)
    boundary_delta_error = delta_error[:, :, boundary - 1 : boundary]

    lowpass_ycc = lowpass_video(rgb_to_ycbcr_delta(boundary_delta_error))
    lowpass_y = lowpass_ycc[:, :1].abs().mean()
    lowpass_chroma = lowpass_ycc[:, 1:].abs().mean()

    # Both second-order triplets that touch the join:
    # (4,5,6): d_5 - d_4 and (5,6,7): d_6 - d_5.
    acceleration_error = delta_error[:, :, 1:] - delta_error[:, :, :-1]
    acceleration = torch.stack(
        (
            acceleration_error[:, :, boundary - 2],
            acceleration_error[:, :, boundary - 1],
        ),
        dim=2,
    ).abs().mean()

    mask = torch.ones(delta_error.shape[2], dtype=torch.bool, device=recon.device)
    mask[boundary - 1] = False
    within = delta_error[:, :, mask]
    source_delta = target[:, :, 1:] - target[:, :, :-1]
    recon_delta = recon[:, :, 1:] - recon[:, :, :-1]
    source_within = source_delta[:, :, mask].abs().mean()
    recon_within = recon_delta[:, :, mask].abs().mean()

    boundary_rgb = boundary_delta_error.abs().mean()
    within_error = within.abs().mean()
    return {
        "boundary_rgb_delta": boundary_rgb,
        "boundary_lowpass_y": lowpass_y,
        "boundary_lowpass_chroma": lowpass_chroma,
        "boundary_lowpass_step": 0.5 * (lowpass_y + lowpass_chroma),
        "boundary_gradient_delta": spatial_gradient_l1(boundary_delta_error),
        "boundary_acceleration": acceleration,
        "within_gop_temporal_error": within_error,
        "boundary_excess": boundary_rgb - within_error,
        "boundary_error_ratio": boundary_rgb / within_error.clamp_min(1e-12),
        "within_motion_ratio": recon_within / source_within.clamp_min(1e-12),
        "spatial_detail_ratio": spatial_gradient_energy(recon)
        / spatial_gradient_energy(target).clamp_min(1e-12),
    }


def simple_ssim(recon: torch.Tensor, target: torch.Tensor) -> float:
    r = recon.float().permute(0, 2, 1, 3, 4).flatten(0, 1)
    t = target.float().permute(0, 2, 1, 3, 4).flatten(0, 1)
    mu_r = F.avg_pool2d(r, 7, stride=1, padding=3)
    mu_t = F.avg_pool2d(t, 7, stride=1, padding=3)
    var_r = F.avg_pool2d(r * r, 7, stride=1, padding=3) - mu_r.square()
    var_t = F.avg_pool2d(t * t, 7, stride=1, padding=3) - mu_t.square()
    cov = F.avg_pool2d(r * t, 7, stride=1, padding=3) - mu_r * mu_t
    value = ((2 * mu_r * mu_t + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mu_r.square() + mu_t.square() + 0.01**2)
        * (var_r + var_t + 0.03**2)
    ).clamp_min(1e-12)
    return float(value.mean().item())


def sequence_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    frames_per_gop: int,
    device: torch.device,
    *,
    include_lpips: bool,
) -> dict[str, float]:
    r = recon.float().clamp(0, 1)
    t = target.float().clamp(0, 1)
    mse = F.mse_loss(r, t).item()
    losses = boundary_losses(r, t, frames_per_gop)
    result = {
        "psnr": float(10 * math.log10(1.0 / max(mse, 1e-12))),
        "ssim": simple_ssim(r, t),
        **{name: float(value.item()) for name, value in losses.items()},
    }
    if include_lpips:
        result["lpips"] = lpips_metric(r, t, device)
        boundary = frames_per_gop
        r_delta = r[:, :, boundary : boundary + 1] - r[:, :, boundary - 1 : boundary]
        t_delta = t[:, :, boundary : boundary + 1] - t[:, :, boundary - 1 : boundary]
        result["boundary_delta_lpips"] = lpips_metric(
            (0.5 * r_delta + 0.5).clamp(0, 1),
            (0.5 * t_delta + 0.5).clamp(0, 1),
            device,
        )
    return result


def profile_for_cell(cell: ChannelCell) -> ChannelProfile:
    if cell.profile_key == MEASURED_PROFILE.key:
        return MEASURED_PROFILE
    return CHANNEL_PROFILES[cell.profile_key]


def runtime_channel_seed(base_seed: int, sequence_index: int, cell_index: int) -> int:
    return int(base_seed + 1009 * sequence_index + 9176 * cell_index)


def runtime_retry_seed(initial_seed: int, attempt: int) -> int:
    """Deterministically advance to another channel realization after a drop."""
    return int(initial_seed + 104729 * attempt)


def runtime_transmit_gops(
    latents: np.ndarray,
    mode: AETVModeSpec,
    cell: ChannelCell,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run two GOPs through the exact GUI loopback waveform/channel/RX path."""
    values = np.asarray(latents, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"expected two latent vectors, got {values.shape}")
    channel = StreamingChannelEmulator(profile_for_cell(cell), seed=seed, fs=mode.geometry.fs)
    demodulator = StreamingDemodulator(
        mode.band,
        continuous=True,
        mode_name=mode.name,
    )
    received: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    diagnostics: list[dict] = []
    block_samples = max(1, mode.geometry.fs // 10)
    chunks = modulate_continuous_chunks(values, mode_name=mode.name, callsign="EVAL")
    for clean in chunks:
        clean = np.asarray(clean, dtype=np.float32).copy()
        clean_peak = float(np.max(np.abs(clean))) if clean.size else 0.0
        if clean_peak > 0:
            clean *= TX_LEVEL / clean_peak
        impaired = channel.process(clean)
        impaired_peak = float(np.max(np.abs(impaired))) if impaired.size else 0.0
        if impaired_peak > 0:
            impaired *= TX_LEVEL / impaired_peak
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
    if len(received) != 2:
        raise RuntimeError(
            f"runtime path recovered {len(received)}/2 GOPs for {cell.label} seed={seed}"
        )
    return np.stack(received), np.stack(weights), diagnostics


def rx_cache_metadata(
    checkpoint: Path,
    mode: AETVModeSpec,
    dataset: SequenceCache,
    *,
    seed: int,
    split_name: str,
) -> dict:
    return {
        "schema": 3,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "mode": mode.name,
        "split": split_name,
        "frames_per_sequence": FRAMES_PER_EXAMPLE,
        "gops_per_sequence": 2,
        "tx_level": TX_LEVEL,
        "channel_path": [
            "AETVAutoencoder.encoder (two explicit independent calls)",
            "modulate_continuous_chunks",
            "StreamingChannelEmulator",
            "StreamingDemodulator(continuous=True)",
            "AETVAutoencoder.decoder (two explicit independent calls)",
        ],
        "channel_seed_base": seed,
        "channel_seed_formula": "base + 1009*sequence_index + 9176*cell_index",
        "decode_retry_formula": "initial_seed + 104729*attempt; first realization recovering 2/2 GOPs",
        "cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "sequences": dataset.manifest(),
    }


def precompute_runtime_rx(
    checkpoint: Path,
    destination: Path,
    dataset: SequenceCache,
    mode: AETVModeSpec,
    device: torch.device,
    *,
    seed: int,
    split_name: str,
) -> dict:
    expected = rx_cache_metadata(checkpoint, mode, dataset, seed=seed, split_name=split_name)
    if destination.is_file():
        cached = torch.load(destination, map_location="cpu", weights_only=False)
        if cached.get("metadata") == expected:
            print(f"Reusing fixed {split_name} runtime RX cache: {destination}", flush=True)
            return cached
        raise RuntimeError(f"stale/incompatible RX cache exists: {destination}")

    model = load_model(checkpoint, mode, device).eval()
    received = {
        cell.label: torch.empty(len(dataset), 2, mode.latents_per_gop, dtype=torch.float32)
        for cell in DEFAULT_CELLS
    }
    weights = {label: torch.empty_like(value) for label, value in received.items()}
    diagnostics = {cell.label: [] for cell in DEFAULT_CELLS}
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index].unsqueeze(0).to(device)
            separated = split_gops(source, mode.gop_frames)
            encoded = encode_independent_gops(model, separated).float().cpu().numpy()
            for cell_index, cell in enumerate(DEFAULT_CELLS):
                initial_seed = runtime_channel_seed(seed, index, cell_index)
                failures = []
                for attempt in range(32):
                    path_seed = runtime_retry_seed(initial_seed, attempt)
                    try:
                        rx, confidence, detail = runtime_transmit_gops(
                            encoded,
                            mode,
                            cell,
                            seed=path_seed,
                        )
                        break
                    except RuntimeError as error:
                        failures.append(str(error))
                else:
                    raise RuntimeError(
                        f"no 2/2-GOP runtime decode for {cell.label} sequence {index} "
                        f"after 32 deterministic realizations; last={failures[-1]}"
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
            print(
                f"  {split_name} runtime path {index + 1:>3}/{len(dataset)}",
                flush=True,
            )
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


def decode_cached_sequence(
    model: AETVAutoencoder,
    rx_cache: dict,
    cell: ChannelCell,
    index: int,
    mode: AETVModeSpec,
    device: torch.device,
) -> torch.Tensor:
    received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
    weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
    reconstructed = decode_independent_gops(
        model,
        received,
        weights,
        mode,
    )
    return join_gops(reconstructed, batch=1, count=2)


def save_candidate(
    model: AETVAutoencoder,
    source_path: Path,
    destination: Path,
    experiment: dict,
) -> None:
    payload, saved = checkpoint_config(source_path)
    output = dict(payload)
    output["model_state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    output["source_checkpoint"] = str(source_path.resolve())
    output["source_checkpoint_sha256"] = sha256_file(source_path)
    output["experiment"] = experiment
    saved = dict(saved)
    saved["gop_boundary_experiment"] = True
    output["args"] = saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)


def train_candidate(
    checkpoint: Path,
    destination: Path,
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
    args: argparse.Namespace,
    multiplier: float,
) -> list[dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = load_model(checkpoint, mode, device)
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    iterator = iter(loader)
    history: list[dict] = []
    model.train()
    for step in range(1, args.steps + 1):
        try:
            indices, source = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            indices, source = next(iterator)
        source = source.to(device, non_blocking=True)
        cell = DEFAULT_CELLS[(step - 1) % len(DEFAULT_CELLS)]
        received = rx_cache["received"][cell.label][indices].to(device)
        confidence = rx_cache["weights"][cell.label][indices].to(device)

        with torch.no_grad():
            reference_gops = decode_independent_gops(
                reference,
                received,
                confidence,
                mode,
            )
            reference_sequence = join_gops(reference_gops, source.shape[0], count=2)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            candidate_gops = decode_independent_gops(
                model,
                received,
                confidence,
                mode,
            )
            candidate = join_gops(candidate_gops, source.shape[0], count=2)
            cross = boundary_losses(candidate, source, mode.gop_frames)
            source_l1 = F.l1_loss(candidate, source)
            anchor_l1 = F.l1_loss(candidate, reference_sequence)
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
                + multiplier * boundary_group
            )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        row = {
            "step": step,
            "cell": cell.label,
            "indices": [int(index) for index in indices],
            "total": float(total.detach()),
            "source_l1": float(source_l1.detach()),
            "anchor_l1": float(anchor_l1.detach()),
            "boundary_group": float(boundary_group.detach()),
            **{name: float(value.detach()) for name, value in cross.items()},
        }
        history.append(row)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"{multiplier:g}x step {step:>4}/{args.steps} {cell.label:<11} "
                f"total={row['total']:.5f} boundary={row['boundary_rgb_delta']:.5f} "
                f"lowpass={row['boundary_lowpass_step']:.5f} "
                f"gradient={row['boundary_gradient_delta']:.5f} "
                f"anchor={row['anchor_l1']:.5f}",
                flush=True,
            )

    experiment = {
        "kind": "two-independent-gop-source-referenced-boundary-finetune",
        "frames": FRAMES_PER_EXAMPLE,
        "gops": 2,
        "steps": args.steps,
        "lr": args.lr,
        "batch": args.batch,
        "seed": args.seed,
        "boundary_multiplier": multiplier,
        "loss_weights": {
            "source": args.source_weight,
            "released_anchor": args.anchor_weight,
            "within_gop_temporal": args.within_weight,
            "boundary_rgb": multiplier,
            "boundary_lowpass_y_and_chroma": multiplier * args.lowpass_term_weight,
            "boundary_gradient_delta": multiplier * args.gradient_term_weight,
            "boundary_acceleration_both_triplets": multiplier
            * args.acceleration_term_weight,
        },
        "training_cells": [cell.label for cell in DEFAULT_CELLS],
        "training_cell_schedule": "round_robin",
        "runtime_rx_cache_sha256": sha256_file(args.train_rx_cache),
        "encoder_frozen": True,
        "decoder_trainable": True,
        "wire_contract_changed": False,
        "gui_blending_used": False,
    }
    save_candidate(model, checkpoint, destination, experiment)
    del reference, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def evaluate_model(
    model: AETVAutoencoder,
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
    *,
    include_lpips: bool,
) -> dict:
    model.eval()
    rows = {cell.label: [] for cell in DEFAULT_CELLS}
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index].unsqueeze(0).to(device)
            for cell in DEFAULT_CELLS:
                recon = decode_cached_sequence(model, rx_cache, cell, index, mode, device)
                rows[cell.label].append(
                    sequence_metrics(
                        recon,
                        source,
                        mode.gop_frames,
                        device,
                        include_lpips=include_lpips,
                    )
                )
            print(f"  evaluated sequence {index + 1:>2}/{len(dataset)}", flush=True)
    return {
        "cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "sequences": rows,
    }


def paired_delta(candidate: Iterable[float], baseline: Iterable[float]) -> dict[str, float]:
    differences = [new - old for new, old in zip(candidate, baseline)]
    mean = st.mean(differences)
    se = st.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
    return {"mean": mean, "se": se, "two_se": 2 * se}


def compare_reports(baseline: dict, candidate: dict) -> dict:
    output: dict = {"cells": {}}
    for cell in baseline["sequences"]:
        old_rows = baseline["sequences"][cell]
        new_rows = candidate["sequences"][cell]
        if len(old_rows) != len(new_rows):
            raise ValueError(f"unpaired report length for {cell}")
        output["cells"][cell] = {}
        for metric in old_rows[0]:
            old_values = [row[metric] for row in old_rows]
            new_values = [row[metric] for row in new_rows]
            output["cells"][cell][metric] = {
                "baseline_mean": st.mean(old_values),
                "candidate_mean": st.mean(new_values),
                "paired_delta": paired_delta(new_values, old_values),
            }
    return output


def significant_regression(values: dict, tolerance: float = 0.0) -> bool:
    delta = values["paired_delta"]
    return delta["mean"] - delta["two_se"] > tolerance


def significant_drop(values: dict, tolerance: float = 0.0) -> bool:
    delta = values["paired_delta"]
    return delta["mean"] + delta["two_se"] < -tolerance


def assess_candidate(comparison: dict) -> dict:
    reasons: list[str] = []
    clean = comparison["cells"]["clean"]
    seam_metrics = (
        "boundary_excess",
        "boundary_error_ratio",
        "boundary_delta_lpips",
        "boundary_lowpass_step",
        "boundary_acceleration",
    )
    improved = [
        metric
        for metric in seam_metrics
        if comparison["cells"]["clean"][metric]["paired_delta"]["mean"] < 0
    ]
    if len(improved) < 3:
        reasons.append(f"only {len(improved)}/5 clean seam metrics improved")

    # LPIPS gate: reject a regression whose 95%-style lower bound exceeds
    # 0.001 in clean or any requested channel cell.
    for cell, metrics in comparison["cells"].items():
        if significant_regression(metrics["lpips"], tolerance=0.001):
            delta = metrics["lpips"]["paired_delta"]
            reasons.append(
                f"{cell} LPIPS regressed {delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
            )

    # Blur/motion gate: a seam gain is invalid if ordinary motion fidelity
    # worsens, or if source-normalized within-GOP motion/detail fall materially.
    if significant_regression(clean["within_gop_temporal_error"]):
        delta = clean["within_gop_temporal_error"]["paired_delta"]
        reasons.append(
            f"within-GOP temporal error regressed {delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
        )
    if significant_drop(clean["within_motion_ratio"], tolerance=0.02):
        delta = clean["within_motion_ratio"]["paired_delta"]
        reasons.append(
            f"within-GOP motion ratio fell {delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
        )
    baseline_detail = clean["spatial_detail_ratio"]["baseline_mean"]
    if baseline_detail <= 1.02 and significant_drop(clean["spatial_detail_ratio"], tolerance=0.02):
        delta = clean["spatial_detail_ratio"]["paired_delta"]
        reasons.append(
            f"spatial detail ratio fell {delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
        )
    return {
        "accepted": not reasons,
        "decision": "retain" if not reasons else "reject",
        "clean_seam_metrics_improved": improved,
        "reasons": reasons,
        "lpips_tolerance": 0.001,
        "motion_ratio_tolerance": 0.02,
        "detail_ratio_tolerance": 0.02,
    }


def candidate_label(multiplier: float) -> str:
    return f"boundary_{multiplier:g}x"


def candidate_path(out: Path, multiplier: float) -> Path:
    return out / f"candidate-boundary-{multiplier:g}x.pt"


REPORT_METRICS = (
    "boundary_excess",
    "boundary_error_ratio",
    "boundary_delta_lpips",
    "boundary_lowpass_step",
    "boundary_acceleration",
    "within_gop_temporal_error",
    "psnr",
    "ssim",
    "lpips",
)


def write_markdown_report(
    destination: Path,
    evaluation: dict,
    comparisons: dict,
    decisions: dict,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# V8 controlled two-GOP boundary fine-tune",
        "",
        f"Released checkpoint SHA-256: `{sha256_file(args.checkpoint)}`",
        "",
        (
            f"Training: {args.steps} steps per candidate, batch {args.batch}, LR {args.lr:g}; "
            "the only sweep variable is the 1x/2x/4x/8x multiplier applied to the complete "
            "source-referenced boundary group."
        ),
        "",
        (
            "Path: two independent six-frame encoder calls -> continuous production TX "
            "conditioner -> stateful runtime channel -> streaming RX modem -> two independent "
            "decoder calls. GUI boundary blending was disabled."
        ),
        "",
        "## Promotion decisions",
        "",
        "| Candidate | Decision | Reasons |",
        "|---|---|---|",
    ]
    for label, decision in decisions.items():
        reasons = "; ".join(decision["reasons"]) or "all seam, LPIPS, and motion gates passed"
        lines.append(f"| {label} | **{decision['decision'].upper()}** | {reasons} |")

    for cell in (item.label for item in DEFAULT_CELLS):
        lines.extend(
            (
                "",
                f"## {cell}",
                "",
                "| Model | " + " | ".join(REPORT_METRICS) + " |",
                "|---|" + "---:|" * len(REPORT_METRICS),
            )
        )
        baseline_rows = evaluation["baseline"]["sequences"][cell]
        baseline_values = [st.mean(row[m] for row in baseline_rows) for m in REPORT_METRICS]
        lines.append("| released | " + " | ".join(f"{v:.6f}" for v in baseline_values) + " |")
        for label in decisions:
            rows = evaluation[label]["sequences"][cell]
            values = [st.mean(row[m] for row in rows) for m in REPORT_METRICS]
            lines.append(f"| {label} | " + " | ".join(f"{v:.6f}" for v in values) + " |")

    lines.extend(
        (
            "",
            "## Paired uncertainty",
            "",
            "`comparison.json` contains the per-sequence paired mean delta, standard error, "
            "and 2x standard error for every metric/cell. Lower is better for seam errors and "
            "LPIPS; higher is better for PSNR/SSIM.",
            "",
        )
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def render_full_comparisons(
    checkpoints: dict[str, Path],
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
    destination: Path,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    render_manifest = {
        "gui_boundary_blending": False,
        "frames_per_sequence": FRAMES_PER_EXAMPLE,
        "seam_transition": "frame 5 -> frame 6",
        "sequences": len(dataset),
        "total_frames_per_render": len(dataset) * FRAMES_PER_EXAMPLE,
        "fps": mode.fps,
        "sequence_frame_intervals": [
            {"sequence": index, "start": index * FRAMES_PER_EXAMPLE, "end": (index + 1) * FRAMES_PER_EXAMPLE}
            for index in range(len(dataset))
        ],
        "files": {},
    }
    sources = torch.cat([dataset[index] for index in range(len(dataset))], dim=1).unsqueeze(0)
    for cell in DEFAULT_CELLS:
        panels: list[tuple[str, torch.Tensor]] = [("Source", sources)]
        for label, checkpoint in checkpoints.items():
            model = load_model(checkpoint, mode, device).eval()
            clips = []
            with torch.inference_mode():
                for index in range(len(dataset)):
                    clips.append(
                        decode_cached_sequence(model, rx_cache, cell, index, mode, device).cpu()
                    )
            panels.append((label, torch.cat(clips, dim=2)))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        output = destination / f"full-paired-32-{cell.label}-no-gui-blend.mp4"
        write_labeled_grid_mp4(panels, output, fps=mode.fps, columns=3)
        render_manifest["files"][cell.label] = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
        }
        print(f"Rendered {output}", flush=True)
    (destination / "render-manifest.json").write_text(
        json.dumps(render_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return render_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("precompute", "train", "evaluate", "render", "all"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/v8-two-gop-boundary-sweep-20260826"))
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path("runs/gop-boundary-data/v8_192x108_3gop_train"),
    )
    parser.add_argument(
        "--eval-cache",
        type=Path,
        default=Path("runs/gop-boundary-data/v8_192x108_3gop_eval"),
    )
    parser.add_argument("--train-rx-cache", type=Path)
    parser.add_argument("--eval-rx-cache", type=Path)
    parser.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--source-weight", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=4.0)
    parser.add_argument("--within-weight", type=float, default=1.0)
    parser.add_argument("--lowpass-term-weight", type=float, default=0.5)
    parser.add_argument("--gradient-term-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-term-weight", type=float, default=0.25)
    parser.add_argument("--multipliers", type=float, nargs="+", default=(1.0, 2.0, 4.0, 8.0))
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--eval-seed", type=int, default=20260827)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if tuple(args.multipliers) != (1.0, 2.0, 4.0, 8.0):
        raise SystemExit("controlled sweep requires exactly --multipliers 1 2 4 8")
    if args.skip_lpips and args.command in {"evaluate", "all"}:
        raise SystemExit("LPIPS cannot be skipped for promotion-gated evaluation")
    mode = AETV_MODES[args.mode]
    if mode.gop_frames != 6:
        raise SystemExit(f"two-GOP experiment requires six-frame GOPs, got {mode.gop_frames}")
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    args.train_rx_cache = args.train_rx_cache or args.out / "train-runtime-rx.pt"
    args.eval_rx_cache = args.eval_rx_cache or args.out / "eval-runtime-rx.pt"
    train_dataset = SequenceCache(args.train_cache, limit=args.train_sequences)
    eval_dataset = SequenceCache(args.eval_cache, limit=args.eval_sequences)
    if len(train_dataset) != args.train_sequences:
        raise SystemExit(f"need {args.train_sequences} train sequences, found {len(train_dataset)}")
    if len(eval_dataset) != args.eval_sequences:
        raise SystemExit(f"need {args.eval_sequences} eval sequences, found {len(eval_dataset)}")

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "mode": mode.name,
        "frames": FRAMES_PER_EXAMPLE,
        "gops": 2,
        "boundary": "5->6",
        "train_sequences": train_dataset.manifest(),
        "eval_sequences": eval_dataset.manifest(),
        "cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "multipliers": list(args.multipliers),
        "gui_blending": False,
    }
    (args.out / "experiment-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    train_rx = None
    eval_rx = None
    if args.command in {"precompute", "train", "evaluate", "render", "all"}:
        train_rx = precompute_runtime_rx(
            args.checkpoint,
            args.train_rx_cache,
            train_dataset,
            mode,
            device,
            seed=args.seed,
            split_name="train",
        )
        eval_rx = precompute_runtime_rx(
            args.checkpoint,
            args.eval_rx_cache,
            eval_dataset,
            mode,
            device,
            seed=args.eval_seed,
            split_name="eval",
        )
        if args.command == "precompute":
            return

    if args.command in {"train", "all"}:
        assert train_rx is not None
        for multiplier in args.multipliers:
            destination = candidate_path(args.out, multiplier)
            started = time.time()
            history = train_candidate(
                args.checkpoint,
                destination,
                train_dataset,
                train_rx,
                mode,
                device,
                args,
                multiplier,
            )
            (args.out / f"training-{multiplier:g}x.json").write_text(
                json.dumps(
                    {
                        "elapsed_s": time.time() - started,
                        "multiplier": multiplier,
                        "steps": history,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.command == "train":
            return

    evaluation_path = args.out / "evaluation.json"
    comparison_path = args.out / "comparison.json"
    decisions_path = args.out / "decisions.json"
    if args.command in {"evaluate", "all"}:
        assert eval_rx is not None
        checkpoint_map = {"baseline": args.checkpoint}
        checkpoint_map.update(
            {candidate_label(m): candidate_path(args.out, m) for m in args.multipliers}
        )
        for label, path in checkpoint_map.items():
            if not path.is_file():
                raise SystemExit(f"missing {label} checkpoint: {path}")
        evaluation = {}
        for label, path in checkpoint_map.items():
            print(f"Evaluating {label}: {path}", flush=True)
            model = load_model(path, mode, device).eval()
            evaluation[label] = evaluate_model(
                model,
                eval_dataset,
                eval_rx,
                mode,
                device,
                include_lpips=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        comparisons = {
            label: compare_reports(evaluation["baseline"], evaluation[label])
            for label in evaluation
            if label != "baseline"
        }
        decisions = {label: assess_candidate(value) for label, value in comparisons.items()}
        evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        comparison_path.write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
        decisions_path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        write_markdown_report(
            args.out / "report.md", evaluation, comparisons, decisions, args
        )
        for label, decision in decisions.items():
            print(f"{label}: {decision['decision'].upper()}", flush=True)
            for reason in decision["reasons"]:
                print(f"  - {reason}", flush=True)
        if args.command == "evaluate":
            return

    if args.command in {"render", "all"}:
        assert eval_rx is not None
        checkpoint_map = {"released": args.checkpoint}
        checkpoint_map.update(
            {f"{m:g}x": candidate_path(args.out, m) for m in args.multipliers}
        )
        render_full_comparisons(
            checkpoint_map,
            eval_dataset,
            eval_rx,
            mode,
            device,
            args.out / "renders",
        )


if __name__ == "__main__":
    main()
