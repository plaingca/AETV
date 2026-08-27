#!/usr/bin/env python3
"""Fine-tune V8 boundary feature context with source-derived optical flow.

The released codec, RF payload, modem path, and selected reliability gate stay
frozen.  Five copies of the retained feature refiner are fine-tuned on the
same immutable two-GOP runtime receive cache.  The only sweep variable is the
weight on an occlusion-masked, source-referenced motion-compensated residual
at the independent frame 5->6 boundary.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics as st
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES, AETVModeSpec  # noqa: E402
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from eval import write_labeled_grid_mp4  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    DEFAULT_CELLS,
    REPORT_METRICS,
    ChannelCell,
    SequenceCache,
    boundary_losses,
    compare_reports,
    load_model,
    sequence_metrics,
    sha256_file,
)
from experiment_gop_feature_context import (  # noqa: E402
    FeatureContextRefiner,
    load_refiner,
)
from experiment_gop_flow import (  # noqa: E402
    RAFTAligner,
    sample_padded_reference,
)
from experiment_gop_reliability_gate import (  # noqa: E402
    MultiGOPSequenceCache,
    apply_reliability_sequence,
    cache_diagnostics_for_sequence,
    decode_received_features,
    flow_in_bounds,
    join_many_gops,
    load_gate,
    multi_boundary_losses,
    prepare_event_batch,
)


FLOW_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
FLOW_METRICS = (
    "boundary_mc_residual",
    "boundary_flow_epe",
    "boundary_flow_cosine",
    "boundary_flow_magnitude_ratio",
    "source_flow_valid_fraction",
)
SEAM_METRICS = (
    "boundary_excess",
    "boundary_error_ratio",
    "boundary_delta_lpips",
    "boundary_lowpass_step",
    "boundary_acceleration",
)


class IndexedDataset(Dataset):
    def __init__(self, dataset: SequenceCache):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return index, self.dataset[index]


def warp_reference(reference: torch.Tensor, target_to_reference: torch.Tensor) -> torch.Tensor:
    return sample_padded_reference(
        reference,
        target_to_reference,
        left=0,
        top=0,
        output_height=target_to_reference.shape[-2],
        output_width=target_to_reference.shape[-1],
    )


def forward_backward_validity(
    target_to_reference: torch.Tensor,
    reference_to_target: torch.Tensor,
    *,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> torch.Tensor:
    """Forward/backward-consistency and image-bounds validity at target pixels."""
    reverse_at_target = warp_reference(reference_to_target, target_to_reference)
    disagreement = (target_to_reference + reverse_at_target).square().sum(dim=1, keepdim=True)
    scale = alpha * (
        target_to_reference.square().sum(dim=1, keepdim=True)
        + reverse_at_target.square().sum(dim=1, keepdim=True)
    ) + beta
    return (disagreement <= scale).to(target_to_reference.dtype) * flow_in_bounds(
        target_to_reference
    )


def source_boundary_geometry(
    source: torch.Tensor,
    aligner: RAFTAligner,
    frames_per_gop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    previous = source[:, :, frames_per_gop - 1]
    current = source[:, :, frames_per_gop]
    with torch.no_grad():
        target_to_reference = aligner.estimate_flow(previous, current)
        reference_to_target = aligner.estimate_flow(current, previous)
        valid = forward_backward_validity(target_to_reference, reference_to_target)
        source_residual = current - warp_reference(previous, target_to_reference)
    return target_to_reference, valid, source_residual


def motion_compensated_residual_loss(
    reconstruction: torch.Tensor,
    source_residual: torch.Tensor,
    target_to_reference: torch.Tensor,
    valid: torch.Tensor,
    frames_per_gop: int,
    *,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    previous = reconstruction[:, :, frames_per_gop - 1]
    current = reconstruction[:, :, frames_per_gop]
    reconstructed_residual = current - warp_reference(previous, target_to_reference)
    difference = reconstructed_residual - source_residual
    robust = torch.sqrt(difference.square() + epsilon**2) - epsilon
    expanded = valid.expand_as(robust)
    return (robust * expanded).sum() / expanded.sum().clamp_min(1.0)


def flow_agreement_metrics(
    reconstruction: torch.Tensor,
    source_residual: torch.Tensor,
    target_to_reference: torch.Tensor,
    valid: torch.Tensor,
    aligner: RAFTAligner,
    frames_per_gop: int,
) -> dict[str, float]:
    with torch.no_grad():
        predicted = aligner.estimate_flow(
            reconstruction[:, :, frames_per_gop - 1],
            reconstruction[:, :, frames_per_gop],
        )
        epe = (predicted - target_to_reference).square().sum(dim=1, keepdim=True).sqrt()
        source_magnitude = target_to_reference.square().sum(dim=1, keepdim=True).sqrt()
        predicted_magnitude = predicted.square().sum(dim=1, keepdim=True).sqrt()
        motion_valid = valid * (source_magnitude >= 0.25).to(valid.dtype)
        if float(motion_valid.sum()) < 1:
            motion_valid = valid
        dot = (predicted * target_to_reference).sum(dim=1, keepdim=True)
        cosine = dot / (predicted_magnitude * source_magnitude).clamp_min(1e-6)
        denominator = motion_valid.sum().clamp_min(1.0)
        mc = motion_compensated_residual_loss(
            reconstruction,
            source_residual,
            target_to_reference,
            valid,
            frames_per_gop,
        )
        return {
            "boundary_mc_residual": float(mc),
            "boundary_flow_epe": float((epe * motion_valid).sum() / denominator),
            "boundary_flow_cosine": float((cosine * motion_valid).sum() / denominator),
            "boundary_flow_magnitude_ratio": float(
                (predicted_magnitude * motion_valid).sum()
                / (source_magnitude * motion_valid).sum().clamp_min(1e-6)
            ),
            "source_flow_valid_fraction": float(valid.mean()),
        }


def flow_cache_metadata(dataset: SequenceCache, checkpoint: Path, mode: AETVModeSpec) -> dict:
    return {
        "schema": 1,
        "kind": "source-boundary-raft-small-forward-backward-consistency",
        "checkpoint_sha256": sha256_file(checkpoint),
        "mode": mode.name,
        "frames_per_gop": mode.gop_frames,
        "boundary": "5->6",
        "raft": "Raft_Small_Weights.DEFAULT",
        "occlusion": {"alpha": 0.01, "beta": 0.5, "in_bounds": True},
        "sequences": dataset.manifest(),
    }


def precompute_source_flow(
    destination: Path,
    dataset: SequenceCache,
    checkpoint: Path,
    mode: AETVModeSpec,
    device: torch.device,
    *,
    batch_size: int,
) -> dict:
    expected = flow_cache_metadata(dataset, checkpoint, mode)
    if destination.is_file():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        if payload.get("metadata") == expected:
            print(f"Reusing source-flow cache: {destination}", flush=True)
            return payload
        raise RuntimeError(f"stale/incompatible source-flow cache: {destination}")
    aligner = RAFTAligner(device).eval()
    flows, masks, residuals = [], [], []
    loader = DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=False)
    for indices, source in loader:
        flow, mask, residual = source_boundary_geometry(
            source.to(device), aligner, mode.gop_frames
        )
        flows.append(flow.cpu())
        masks.append(mask.cpu())
        residuals.append(residual.cpu())
        print(
            f"  source flow {int(indices[-1]) + 1:>3}/{len(dataset)} "
            f"valid={float(mask.mean()):.3f}",
            flush=True,
        )
    payload = {
        "metadata": expected,
        "target_to_reference": torch.cat(flows),
        "valid": torch.cat(masks),
        "source_residual": torch.cat(residuals),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    del aligner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def verify_rx_cache(path: Path, dataset: SequenceCache, checkpoint: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"RX cache checkpoint mismatch: {path}")
    if metadata.get("gops_per_sequence") != 2:
        raise RuntimeError(f"RX cache is not two-GOP: {path}")
    expected = [row["pixel_sha256"] for row in metadata.get("sequences", [])]
    actual = [row["pixel_sha256"] for row in dataset.manifest()]
    if expected[: len(actual)] != actual:
        raise RuntimeError(f"RX cache sequence pairing mismatch: {path}")
    return payload


def verify_failure_rx_cache(
    path: Path,
    dataset: MultiGOPSequenceCache,
    checkpoint: Path,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"failure RX cache checkpoint mismatch: {path}")
    if metadata.get("gops_per_sequence") != dataset.gops:
        raise RuntimeError(f"failure RX cache GOP count mismatch: {path}")
    expected = [row["pixel_sha256"] for row in metadata.get("sequences", [])]
    actual = [row["pixel_sha256"] for row in dataset.manifest()]
    if expected[: len(actual)] != actual:
        raise RuntimeError(f"failure RX cache sequence pairing mismatch: {path}")
    return payload


def batch_diagnostics(
    rx_cache: dict,
    cell: ChannelCell,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    snr, coherence = [], []
    for index in indices.tolist():
        rows = rx_cache["diagnostics"][cell.label][index]["gops"]
        snr.append([float(row["snr_db"]) for row in rows])
        coherence.append([float(row["pilot_coherence"]) for row in rows])
    return torch.tensor(snr, device=device), torch.tensor(coherence, device=device)


def apply_refiner(
    args: argparse.Namespace,
    model,
    refiner: FeatureContextRefiner,
    gate,
    aligner: RAFTAligner,
    received: torch.Tensor,
    weights: torch.Tensor,
    snr: torch.Tensor,
    coherence: torch.Tensor,
    mode: AETVModeSpec,
    *,
    train_refiner: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    base, features, skips = decode_received_features(model, received, weights, mode)
    valid = torch.ones(base.shape[:2], dtype=torch.bool, device=base.device)
    reset = torch.zeros_like(valid)
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
        detach_refiner=not train_refiner,
    )
    return base, corrected


def weight_token(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def candidate_label(weight: float) -> str:
    return f"flow_{weight:g}"


def candidate_path(out: Path, weight: float) -> Path:
    return out / f"refiner-flow-{weight_token(weight)}.pt"


def save_candidate(
    refiner: FeatureContextRefiner,
    source_payload: dict,
    destination: Path,
    args: argparse.Namespace,
    weight: float,
    elapsed_s: float,
) -> None:
    payload = copy.deepcopy(source_payload)
    payload["refiner_state_dict"] = {
        name: value.detach().cpu() for name, value in refiner.state_dict().items()
    }
    payload["source_refiner"] = str(args.refiner.resolve())
    payload["source_refiner_sha256"] = sha256_file(args.refiner)
    payload["source_flow_experiment"] = {
        "weight": weight,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "elapsed_s": elapsed_s,
        "gate": str(args.gate.resolve()),
        "gate_sha256": sha256_file(args.gate),
        "train_runtime_rx_sha256": sha256_file(args.train_rx_cache),
        "train_source_flow_sha256": sha256_file(args.train_flow_cache),
        "codec_frozen": True,
        "gate_frozen": True,
        "wire_contract_changed": False,
        "gui_blending": False,
        "loss_weights": {
            "source_l1": args.source_weight,
            "starting_adapter_anchor_l1": args.anchor_weight,
            "boundary_group": args.boundary_weight,
            "within_gop": args.within_weight,
            "vgg_source": args.vgg_source_weight,
            "vgg_anchor": args.vgg_anchor_weight,
            "source_flow_residual": weight,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def train_candidate(
    args: argparse.Namespace,
    weight: float,
    destination: Path,
    dataset: SequenceCache,
    rx_cache: dict,
    flow_cache: dict,
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
    starting, starting_payload = load_refiner(args.refiner, device)
    starting.eval()
    for parameter in starting.parameters():
        parameter.requires_grad_(False)
    candidate = copy.deepcopy(starting).train()
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    gate, gate_payload = load_gate(args.gate, device)
    if gate_payload["refiner_sha256"] != sha256_file(args.refiner):
        raise RuntimeError("selected gate was trained against another refiner")
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    aligner = RAFTAligner(device).eval()
    perceptual = MultiLayerVGGPerceptualLoss().to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        IndexedDataset(dataset),
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
        cell = DEFAULT_CELLS[(step - 1) % len(DEFAULT_CELLS)]
        received = rx_cache["received"][cell.label][indices].to(device)
        weights = rx_cache["weights"][cell.label][indices].to(device)
        snr, coherence = batch_diagnostics(rx_cache, cell, indices, device)
        target_flow = flow_cache["target_to_reference"][indices].to(device)
        valid_flow = flow_cache["valid"][indices].to(device)
        source_residual = flow_cache["source_residual"][indices].to(device)
        with torch.no_grad():
            base, anchor_gops = apply_refiner(
                args,
                model,
                starting,
                gate,
                aligner,
                received,
                weights,
                snr,
                coherence,
                mode,
                train_refiner=False,
            )
            base_sequence = join_many_gops(base)
            anchor = join_many_gops(anchor_gops)
        optimizer.zero_grad(set_to_none=True)
        _, corrected_gops = apply_refiner(
            args,
            model,
            candidate,
            gate,
            aligner,
            received,
            weights,
            snr,
            coherence,
            mode,
            train_refiner=True,
        )
        corrected = join_many_gops(corrected_gops)
        cross = boundary_losses(corrected, source, mode.gop_frames)
        source_l1 = F.l1_loss(corrected, source)
        anchor_l1 = F.l1_loss(corrected, anchor)
        vgg_source = perceptual(corrected, source)
        vgg_anchor = perceptual(corrected, anchor)
        flow_loss = motion_compensated_residual_loss(
            corrected,
            source_residual,
            target_flow,
            valid_flow,
            mode.gop_frames,
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
            + args.boundary_weight * boundary_group
            + args.within_weight * cross["within_gop_temporal_error"]
            + args.vgg_source_weight * vgg_source
            + args.vgg_anchor_weight * vgg_anchor
            + weight * flow_loss
        )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), args.clip_grad)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at flow {weight:g} step {step}")
        optimizer.step()
        row = {
            "step": step,
            "cell": cell.label,
            "indices": [int(index) for index in indices],
            "total": float(total.detach()),
            "source_l1": float(source_l1.detach()),
            "anchor_l1": float(anchor_l1.detach()),
            "base_anchor_l1": float(F.l1_loss(anchor, base_sequence)),
            "vgg_source": float(vgg_source.detach()),
            "vgg_anchor": float(vgg_anchor.detach()),
            "source_flow_residual": float(flow_loss.detach()),
            "flow_valid_fraction": float(valid_flow.mean()),
            "grad_norm": float(grad_norm.detach()),
            **{name: float(value.detach()) for name, value in cross.items()},
        }
        history.append(row)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"flow {weight:<4g} step {step:>4}/{args.steps} {cell.label:<11} "
                f"total={row['total']:.5f} flow={row['source_flow_residual']:.5f} "
                f"boundary={row['boundary_rgb_delta']:.5f} anchor={row['anchor_l1']:.5f}",
                flush=True,
            )
    save_candidate(
        candidate,
        starting_payload,
        destination,
        args,
        weight,
        time.time() - started,
    )
    del model, starting, candidate, gate, aligner, perceptual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def evaluate_variants(
    args: argparse.Namespace,
    refiners: dict[str, FeatureContextRefiner | None],
    dataset: SequenceCache,
    rx_cache: dict,
    flow_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    model = load_model(args.checkpoint, mode, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate, _ = load_gate(args.gate, device)
    gate.eval()
    aligner = RAFTAligner(device).eval()
    rows = {
        label: {cell.label: [] for cell in DEFAULT_CELLS} for label in refiners
    }
    timings = {label: [] for label, refiner in refiners.items() if refiner is not None}
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index].unsqueeze(0).to(device)
            target_flow = flow_cache["target_to_reference"][index : index + 1].to(device)
            valid_flow = flow_cache["valid"][index : index + 1].to(device)
            source_residual = flow_cache["source_residual"][index : index + 1].to(device)
            for cell in DEFAULT_CELLS:
                received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
                weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
                snr, coherence = cache_diagnostics_for_sequence(
                    rx_cache, cell, index, device
                )
                base, features, skips = decode_received_features(
                    model, received, weights, mode
                )
                for label, refiner in refiners.items():
                    if refiner is None:
                        gops = base
                    else:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        started = time.perf_counter()
                        valid = torch.ones(1, 2, dtype=torch.bool, device=device)
                        reset = torch.zeros_like(valid)
                        gops, _ = apply_reliability_sequence(
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
                        timings[label].append(time.perf_counter() - started)
                    reconstruction = join_many_gops(gops)
                    metrics = sequence_metrics(
                        reconstruction,
                        source,
                        mode.gop_frames,
                        device,
                        include_lpips=True,
                    )
                    metrics.update(
                        flow_agreement_metrics(
                            reconstruction,
                            source_residual,
                            target_flow,
                            valid_flow,
                            aligner,
                            mode.gop_frames,
                        )
                    )
                    rows[label][cell.label].append(metrics)
            print(f"  evaluated all variants {index + 1:>2}/{len(dataset)}", flush=True)
    output = {}
    for label in refiners:
        output[label] = {
            "cells": [asdict(cell) for cell in DEFAULT_CELLS],
            "sequences": rows[label],
        }
        if label in timings:
            output[label]["runtime_ms_per_boundary"] = 1000 * st.mean(timings[label])
    return output


def evaluate_failures(
    args: argparse.Namespace,
    refiners: dict[str, FeatureContextRefiner | None],
    dataset: MultiGOPSequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    model = load_model(args.checkpoint, mode, device).eval()
    gate, _ = load_gate(args.gate, device)
    gate.eval()
    aligner = RAFTAligner(device).eval()
    events = (
        "good_fade_good",
        "false_high_confidence",
        "false_low_confidence",
        "missing_gop",
        "random_reset",
        "scene_cut",
    )
    output = {event: {label: [] for label in refiners} for event in events}
    with torch.inference_mode():
        for event_name in events:
            for start in range(0, len(dataset), args.batch):
                indices_list = list(range(start, min(start + args.batch, len(dataset))))
                if event_name == "scene_cut" and len(indices_list) < 2:
                    indices_list = [indices_list[0], (indices_list[0] + 1) % len(dataset)]
                indices = torch.tensor(indices_list, dtype=torch.long)
                source = torch.stack([dataset[index] for index in indices_list]).to(device)
                event = prepare_event_batch(
                    event_name, source, indices, rx_cache, mode, device
                )
                base, features, skips = decode_received_features(
                    model, event.received, event.decode_weights, mode
                )
                released = join_many_gops(base)
                start_frame = event.recovery_gop * mode.gop_frames
                end_frame = start_frame + mode.gop_frames
                for label, refiner in refiners.items():
                    if refiner is None:
                        reconstruction = released
                    else:
                        corrected, _ = apply_reliability_sequence(
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
                        reconstruction = join_many_gops(corrected)
                    for row, sequence_index in enumerate(indices_list):
                        reference_error = F.l1_loss(
                            released[row, :, start_frame:end_frame],
                            event.source[row, :, start_frame:end_frame],
                        )
                        candidate_error = F.l1_loss(
                            reconstruction[row, :, start_frame:end_frame],
                            event.source[row, :, start_frame:end_frame],
                        )
                        cross = multi_boundary_losses(
                            reconstruction[row : row + 1],
                            event.source[row : row + 1],
                            mode.gop_frames,
                        )
                        output[event_name][label].append(
                            {
                                "sequence": sequence_index,
                                "recovery_error_ratio": float(
                                    candidate_error / reference_error.clamp_min(1e-12)
                                ),
                                "full_l1": float(
                                    F.l1_loss(reconstruction[row], event.source[row])
                                ),
                                "boundary_rgb_delta": float(cross["boundary_rgb_delta"]),
                                "boundary_lowpass_step": float(
                                    cross["boundary_lowpass_step"]
                                ),
                                "boundary_acceleration": float(
                                    cross["boundary_acceleration"]
                                ),
                            }
                        )
            print(f"  failure case {event_name}", flush=True)
    return output


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def outlier_summary(report: dict) -> dict:
    output = {}
    for cell, rows in report["sequences"].items():
        output[cell] = {}
        for metric in ("boundary_error_ratio", "boundary_delta_lpips"):
            values = [row[metric] for row in rows]
            maximum = max(values)
            output[cell][metric] = {
                "p95": percentile(values, 0.95),
                "max": maximum,
                "max_sequence": values.index(maximum),
            }
    return output


def statistically_worse(metric: dict, tolerance: float = 0.0) -> bool:
    delta = metric["paired_delta"]
    return delta["mean"] - delta["two_se"] > tolerance


def statistically_lower(metric: dict, tolerance: float = 0.0) -> bool:
    delta = metric["paired_delta"]
    return delta["mean"] + delta["two_se"] < -tolerance


def worst_ratio_drop(
    candidate: dict,
    reference: dict,
    cell: str,
    metric: str,
) -> tuple[float, int, float, float]:
    rows = candidate["sequences"][cell]
    baselines = reference["sequences"][cell]
    eligible = [
        (row[metric] - base[metric], index, base[metric], row[metric])
        for index, (row, base) in enumerate(zip(rows, baselines))
        if row[metric] < 1.0
    ]
    return min(eligible, default=(0.0, -1, 0.0, 0.0))


def assess_candidate(
    label: str,
    vs_spatial: dict,
    vs_released: dict,
    candidate: dict,
    spatial: dict,
    outliers: dict,
    failure: dict,
    args: argparse.Namespace,
    vs_zero_flow: dict | None = None,
) -> dict:
    reasons = []
    improved = [
        metric
        for metric in SEAM_METRICS
        if vs_spatial["cells"]["clean"][metric]["paired_delta"]["mean"] < 0
    ]
    if len(improved) < 3:
        reasons.append(f"only {len(improved)}/5 clean seam metrics improved versus spatial")
    flow_delta = vs_spatial["cells"]["clean"]["boundary_mc_residual"]["paired_delta"]
    if flow_delta["mean"] + flow_delta["two_se"] >= 0:
        reasons.append(
            "clean motion-compensated residual did not improve significantly "
            f"({flow_delta['mean']:+.6f} +/- {flow_delta['two_se']:.6f})"
        )
    if vs_zero_flow is not None:
        isolated = vs_zero_flow["cells"]["clean"]["boundary_mc_residual"][
            "paired_delta"
        ]
        if isolated["mean"] + isolated["two_se"] >= 0:
            reasons.append(
                "source-flow term did not beat the zero-flow fine-tune control "
                f"({isolated['mean']:+.6f} +/- {isolated['two_se']:.6f})"
            )
    for comparison_name, comparison in (
        ("released", vs_released),
        ("spatial", vs_spatial),
    ):
        for cell, metrics in comparison["cells"].items():
            if statistically_worse(metrics["lpips"], tolerance=0.001):
                delta = metrics["lpips"]["paired_delta"]
                reasons.append(
                    f"{cell} LPIPS regressed versus {comparison_name} "
                    f"{delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
                )
    for cell, metrics in vs_spatial["cells"].items():
        if statistically_worse(metrics["within_gop_temporal_error"]):
            delta = metrics["within_gop_temporal_error"]["paired_delta"]
            reasons.append(
                f"{cell} within-GOP temporal error regressed versus spatial "
                f"{delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
            )
        if statistically_worse(metrics["boundary_mc_residual"]):
            delta = metrics["boundary_mc_residual"]["paired_delta"]
            reasons.append(
                f"{cell} motion-compensated residual regressed "
                f"{delta['mean']:+.6f} +/- {delta['two_se']:.6f}"
            )
        if statistically_worse(metrics["boundary_flow_epe"], tolerance=0.02):
            delta = metrics["boundary_flow_epe"]["paired_delta"]
            reasons.append(f"{cell} boundary flow EPE regressed {delta['mean']:+.4f}")
        for metric, description in (
            ("within_motion_ratio", "motion ratio"),
            ("spatial_detail_ratio", "detail ratio"),
        ):
            drop, sequence, before, after = worst_ratio_drop(
                candidate, spatial, cell, metric
            )
            if drop < -0.02:
                reasons.append(
                    f"{cell} sequence {sequence} {description} fell "
                    f"{before:.3f}->{after:.3f}"
                )
        ratio = outliers[cell]["boundary_error_ratio"]
        if ratio["max"] > args.max_boundary_ratio:
            reasons.append(
                f"{cell} max boundary ratio {ratio['max']:.3f} at sequence "
                f"{ratio['max_sequence']} exceeds {args.max_boundary_ratio:.3f}"
            )
        delta_lpips = outliers[cell]["boundary_delta_lpips"]
        if delta_lpips["max"] > args.max_boundary_lpips:
            reasons.append(
                f"{cell} max boundary LPIPS {delta_lpips['max']:.3f} at sequence "
                f"{delta_lpips['max_sequence']} exceeds {args.max_boundary_lpips:.3f}"
            )
    for event in (
        "good_fade_good",
        "false_high_confidence",
        "missing_gop",
        "random_reset",
        "scene_cut",
    ):
        worst = max(row["recovery_error_ratio"] for row in failure[event][label])
        if worst > 1.05:
            reasons.append(f"{event} worst recovery ratio {worst:.4f} exceeds 1.05")
    return {
        "candidate": label,
        "accepted": not reasons,
        "decision": "retain" if not reasons else "reject",
        "clean_seam_metrics_improved_vs_spatial": improved,
        "reasons": reasons,
        "limits": {
            "lpips_regression": 0.001,
            "per_sequence_motion_or_detail_drop": 0.02,
            "max_boundary_ratio": args.max_boundary_ratio,
            "max_boundary_lpips": args.max_boundary_lpips,
            "failure_recovery_ratio": 1.05,
        },
    }


def write_report(
    destination: Path,
    args: argparse.Namespace,
    evaluations: dict,
    comparisons: dict,
    decisions: dict,
    outliers: dict,
    failure: dict,
) -> None:
    metrics = tuple(dict.fromkeys(REPORT_METRICS + FLOW_METRICS))
    lines = [
        "# V8 source-flow-supervised GOP boundary adapter",
        "",
        f"Released checkpoint SHA-256: `{sha256_file(args.checkpoint)}`",
        f"Starting refiner SHA-256: `{sha256_file(args.refiner)}`",
        f"Frozen reliability gate SHA-256: `{sha256_file(args.gate)}`",
        "",
        (
            "The codec, modem, RF payload, and reliability gate are unchanged. Five copies "
            "of the retained feature refiner were fine-tuned; only the source-flow loss "
            "weight changed, including a zero-flow fine-tune control. GUI blending was disabled."
        ),
        "",
        "## Decisions",
        "",
        "| Candidate | Decision | Reasons |",
        "|---|---|---|",
    ]
    for label, decision in decisions.items():
        reason = "; ".join(decision["reasons"]) or "all seam, flow, LPIPS, motion, and recovery gates passed"
        lines.append(f"| {label} | **{decision['decision'].upper()}** | {reason} |")
    for cell in (item.label for item in DEFAULT_CELLS):
        lines.extend(
            (
                "",
                f"## {cell}",
                "",
                "| Model | " + " | ".join(metrics) + " |",
                "|---|" + "---:|" * len(metrics),
            )
        )
        for label, report in evaluations.items():
            rows = report["sequences"][cell]
            values = [st.mean(row[metric] for row in rows) for metric in metrics]
            lines.append(
                f"| {label} | " + " | ".join(f"{value:.6f}" for value in values) + " |"
            )
        lines.extend(
            (
                "",
                "| Model | ratio p95 | ratio max (seq) | boundary LPIPS p95 | boundary LPIPS max (seq) |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for label in evaluations:
            ratio = outliers[label][cell]["boundary_error_ratio"]
            lpips = outliers[label][cell]["boundary_delta_lpips"]
            lines.append(
                f"| {label} | {ratio['p95']:.4f} | {ratio['max']:.4f} "
                f"({ratio['max_sequence']}) | {lpips['p95']:.4f} | "
                f"{lpips['max']:.4f} ({lpips['max_sequence']}) |"
            )
    lines.extend(
        (
            "",
            "## Failure recovery",
            "",
            "| Event | Model | mean recovery ratio | worst recovery ratio | mean full L1 |",
            "|---|---|---:|---:|---:|",
        )
    )
    for event, variants in failure.items():
        for label, rows in variants.items():
            ratios = [row["recovery_error_ratio"] for row in rows]
            full = [row["full_l1"] for row in rows]
            lines.append(
                f"| {event} | {label} | {st.mean(ratios):.6f} | "
                f"{max(ratios):.6f} | {st.mean(full):.6f} |"
            )
    lines.extend(
        (
            "",
            "## Paired uncertainty",
            "",
            "`comparisons.json` contains paired deltas and two-standard-error intervals "
            "versus both released and the unchanged starting spatial adapter.",
            "",
        )
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def render_full(
    args: argparse.Namespace,
    refiners: dict[str, FeatureContextRefiner | None],
    dataset: SequenceCache,
    rx_cache: dict,
    mode: AETVModeSpec,
    device: torch.device,
) -> dict:
    model = load_model(args.checkpoint, mode, device).eval()
    gate, _ = load_gate(args.gate, device)
    gate.eval()
    aligner = RAFTAligner(device).eval()
    destination = args.out / "renders"
    destination.mkdir(parents=True, exist_ok=True)
    source = torch.cat([dataset[index] for index in range(len(dataset))], dim=1).unsqueeze(0)
    manifest = {
        "gui_boundary_blending": False,
        "frames_per_sequence": 12,
        "sequences": len(dataset),
        "total_frames_per_render": 12 * len(dataset),
        "fps": mode.fps,
        "files": {},
    }
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            clips = {label: [] for label in refiners}
            for index in range(len(dataset)):
                received = rx_cache["received"][cell.label][index].unsqueeze(0).to(device)
                weights = rx_cache["weights"][cell.label][index].unsqueeze(0).to(device)
                snr, coherence = cache_diagnostics_for_sequence(
                    rx_cache, cell, index, device
                )
                base, features, skips = decode_received_features(
                    model, received, weights, mode
                )
                valid = torch.ones(1, 2, dtype=torch.bool, device=device)
                reset = torch.zeros_like(valid)
                for label, refiner in refiners.items():
                    if refiner is None:
                        gops = base
                    else:
                        gops, _ = apply_reliability_sequence(
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
                    clips[label].append(join_many_gops(gops).cpu())
            panels = [("Source", source)] + [
                (label, torch.cat(clips[label], dim=2)) for label in refiners
            ]
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
    parser.add_argument("command", choices=("precompute", "train", "evaluate", "render", "all"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    parser.add_argument("--refiner", type=Path, default=Path("runs/gop-feature-context-v8/refiner.pt"))
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("runs/v8-reliability-gated-feature-context-safety-20260826/gate-spatial.pt"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/v8-source-flow-boundary-20260826")
    )
    parser.add_argument(
        "--train-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_3gop_train")
    )
    parser.add_argument(
        "--eval-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_3gop_eval")
    )
    parser.add_argument(
        "--train-rx-cache",
        type=Path,
        default=Path("runs/v8-two-gop-boundary-sweep-explicit-20260826/train-runtime-rx.pt"),
    )
    parser.add_argument(
        "--eval-rx-cache",
        type=Path,
        default=Path("runs/v8-two-gop-boundary-sweep-explicit-20260826/eval-runtime-rx.pt"),
    )
    parser.add_argument(
        "--failure-cache", type=Path, default=Path("runs/gop-boundary-data/v8_192x108_5gop_eval")
    )
    parser.add_argument(
        "--failure-rx-cache",
        type=Path,
        default=Path("runs/v8-reliability-gated-feature-context-20260826/failure-eval-runtime-rx-5gop.pt"),
    )
    parser.add_argument("--train-flow-cache", type=Path)
    parser.add_argument("--eval-flow-cache", type=Path)
    parser.add_argument("--mode", default="V8", choices=tuple(AETV_MODES))
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--failure-sequences", type=int, default=8)
    parser.add_argument("--flow-weights", type=float, nargs="+", default=FLOW_WEIGHTS)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--flow-cache-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=4.0)
    parser.add_argument("--boundary-weight", type=float, default=0.25)
    parser.add_argument("--within-weight", type=float, default=4.0)
    parser.add_argument("--lowpass-term-weight", type=float, default=0.5)
    parser.add_argument("--gradient-term-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-term-weight", type=float, default=0.25)
    parser.add_argument("--vgg-source-weight", type=float, default=0.5)
    parser.add_argument("--vgg-anchor-weight", type=float, default=0.25)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--output-flow-strength", type=float, default=0.10)
    parser.add_argument("--photometric-threshold", type=float, default=0.10)
    parser.add_argument("--photometric-softness", type=float, default=0.02)
    parser.add_argument("--scene-threshold-multiplier", type=float, default=1.5)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.15)
    parser.add_argument("--min-previous-snr-db", type=float, default=-2.0)
    parser.add_argument("--min-previous-pilot-coherence", type=float, default=0.25)
    parser.add_argument("--max-boundary-ratio", type=float, default=3.0)
    parser.add_argument("--max-boundary-lpips", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if tuple(args.flow_weights) != FLOW_WEIGHTS:
        raise SystemExit("controlled sweep requires --flow-weights 0 0.1 0.25 0.5 1.0")
    mode = AETV_MODES[args.mode]
    if mode.gop_frames != 6:
        raise SystemExit("source-flow experiment requires six-frame GOPs")
    for path in (
        args.checkpoint,
        args.refiner,
        args.gate,
        args.train_rx_cache,
        args.eval_rx_cache,
        args.failure_rx_cache,
    ):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    args.train_flow_cache = args.train_flow_cache or args.out / "train-source-flow.pt"
    args.eval_flow_cache = args.eval_flow_cache or args.out / "eval-source-flow.pt"
    train_dataset = SequenceCache(args.train_cache, limit=args.train_sequences)
    eval_dataset = SequenceCache(args.eval_cache, limit=args.eval_sequences)
    failure_dataset = MultiGOPSequenceCache(
        args.failure_cache, 5, mode.gop_frames, args.failure_sequences
    )
    if len(train_dataset) != args.train_sequences:
        raise SystemExit(f"need {args.train_sequences} train sequences, found {len(train_dataset)}")
    if len(eval_dataset) != args.eval_sequences:
        raise SystemExit(f"need {args.eval_sequences} eval sequences, found {len(eval_dataset)}")
    if len(failure_dataset) != args.failure_sequences:
        raise SystemExit(f"need {args.failure_sequences} failure sequences, found {len(failure_dataset)}")
    train_rx = verify_rx_cache(args.train_rx_cache, train_dataset, args.checkpoint)
    eval_rx = verify_rx_cache(args.eval_rx_cache, eval_dataset, args.checkpoint)
    failure_rx = verify_failure_rx_cache(
        args.failure_rx_cache, failure_dataset, args.checkpoint
    )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "starting_refiner": str(args.refiner.resolve()),
        "starting_refiner_sha256": sha256_file(args.refiner),
        "frozen_gate": str(args.gate.resolve()),
        "frozen_gate_sha256": sha256_file(args.gate),
        "flow_weights": list(args.flow_weights),
        "frames": 12,
        "gops": 2,
        "boundary": "5->6",
        "training_cells": [asdict(cell) for cell in DEFAULT_CELLS],
        "runtime_channel_cache": {
            "train": str(args.train_rx_cache.resolve()),
            "eval": str(args.eval_rx_cache.resolve()),
        },
        "wire_contract_changed": False,
        "gui_blending": False,
    }
    (args.out / "experiment-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    train_flow = precompute_source_flow(
        args.train_flow_cache,
        train_dataset,
        args.checkpoint,
        mode,
        device,
        batch_size=args.flow_cache_batch,
    )
    eval_flow = precompute_source_flow(
        args.eval_flow_cache,
        eval_dataset,
        args.checkpoint,
        mode,
        device,
        batch_size=args.flow_cache_batch,
    )
    if args.command == "precompute":
        return
    if args.command in {"train", "all"}:
        for weight in args.flow_weights:
            started = time.time()
            history = train_candidate(
                args,
                weight,
                candidate_path(args.out, weight),
                train_dataset,
                train_rx,
                train_flow,
                mode,
                device,
            )
            (args.out / f"training-flow-{weight_token(weight)}.json").write_text(
                json.dumps(
                    {
                        "weight": weight,
                        "elapsed_s": time.time() - started,
                        "steps": history,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.command == "train":
            return
    refiners: dict[str, FeatureContextRefiner | None] = {"released": None}
    spatial, _ = load_refiner(args.refiner, device)
    refiners["spatial"] = spatial.eval()
    for weight in args.flow_weights:
        path = candidate_path(args.out, weight)
        if not path.is_file():
            raise SystemExit(f"missing trained candidate: {path}")
        candidate, payload = load_refiner(path, device)
        if payload.get("source_refiner_sha256") != sha256_file(args.refiner):
            raise SystemExit(f"candidate provenance mismatch: {path}")
        refiners[candidate_label(weight)] = candidate.eval()
    if args.command in {"evaluate", "all"}:
        evaluations = evaluate_variants(
            args, refiners, eval_dataset, eval_rx, eval_flow, mode, device
        )
        comparisons = {
            label: {
                "vs_released": compare_reports(evaluations["released"], evaluations[label]),
                "vs_spatial": compare_reports(evaluations["spatial"], evaluations[label]),
            }
            for label in refiners
            if label not in {"released", "spatial"}
        }
        zero_label = candidate_label(0.0)
        for label in comparisons:
            if label != zero_label:
                comparisons[label]["vs_zero_flow"] = compare_reports(
                    evaluations[zero_label], evaluations[label]
                )
        outliers = {
            label: outlier_summary(report) for label, report in evaluations.items()
        }
        failure = evaluate_failures(
            args, refiners, failure_dataset, failure_rx, mode, device
        )
        decisions = {
            label: assess_candidate(
                label,
                comparisons[label]["vs_spatial"],
                comparisons[label]["vs_released"],
                evaluations[label],
                evaluations["spatial"],
                outliers[label],
                failure,
                args,
                comparisons[label].get("vs_zero_flow"),
            )
            for label in comparisons
        }
        for name, value in (
            ("evaluation.json", evaluations),
            ("comparisons.json", comparisons),
            ("outliers.json", outliers),
            ("failure-evaluation.json", failure),
            ("decisions.json", decisions),
        ):
            (args.out / name).write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )
        write_report(
            args.out / "report.md",
            args,
            evaluations,
            comparisons,
            decisions,
            outliers,
            failure,
        )
        for label, decision in decisions.items():
            print(f"{label}: {decision['decision'].upper()}", flush=True)
            for reason in decision["reasons"]:
                print(f"  - {reason}", flush=True)
        if args.command == "evaluate":
            return
    if args.command in {"render", "all"}:
        render_full(args, refiners, eval_dataset, eval_rx, mode, device)


if __name__ == "__main__":
    main()
