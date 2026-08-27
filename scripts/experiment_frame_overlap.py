#!/usr/bin/env python3
"""Evaluate the training-free one-frame GOP overlap proposed at ICLR 2026.

The faithful method advances a six-frame V8 input window by five frames,
decodes the shared source frame twice, and averages the two reconstructions.
That costs 20% more V8 latent symbols per source second.  A second diagnostic
keeps the long-run symbol rate fixed by uniformly retaining only 5/6 of each
overlapping latent and supplying zero confidence for omitted coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aetv.config import AETV_MODES  # noqa: E402
from aetv.overlap_models import join_video_gops, split_video_gops  # noqa: E402
from experiment_gop_boundaries import (  # noqa: E402
    SequenceCache,
    load_model,
    simple_ssim,
)
from eval import write_labeled_grid_mp4  # noqa: E402
from render_overlap_checkpoint import extract_video  # noqa: E402
from train import lpips_metric  # noqa: E402


THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def overlapping_windows(
    sequence: torch.Tensor, window_frames: int = 6, stride_frames: int = 5
) -> torch.Tensor:
    """Convert BCTHW video to overlapping BGCTHW windows."""
    if sequence.ndim != 5:
        raise ValueError("sequence must be BCTHW")
    if not 0 < stride_frames < window_frames:
        raise ValueError("stride must be positive and smaller than the window")
    frames = sequence.shape[2]
    if frames < window_frames:
        raise ValueError("sequence is shorter than one GOP window")
    starts = range(0, frames - window_frames + 1, stride_frames)
    return torch.stack(
        [sequence[:, :, start : start + window_frames] for start in starts], dim=1
    )


def pad_for_overlap_coverage(
    sequence: torch.Tensor, window_frames: int = 6, stride_frames: int = 5
) -> torch.Tensor:
    """Repeat the terminal frame so overlapping windows cover the full clip."""
    if sequence.shape[2] < window_frames:
        raise ValueError("sequence is shorter than one GOP window")
    remainder = (sequence.shape[2] - window_frames) % stride_frames
    padding = (stride_frames - remainder) % stride_frames
    if not padding:
        return sequence
    tail = sequence[:, :, -1:].expand(-1, -1, padding, -1, -1)
    return torch.cat((sequence, tail), dim=2)


def fuse_overlapping_gops(decoded: torch.Tensor, theta: float = 0.5) -> torch.Tensor:
    """Deduplicate a one-frame overlap by fusing both transition decodes."""
    if decoded.ndim != 6:
        raise ValueError("decoded GOPs must be BGCTHW")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    _, count, _, frames, _, _ = decoded.shape
    if count < 1 or frames < 2:
        raise ValueError("at least one multi-frame GOP is required")
    if count == 1:
        return decoded[:, 0]
    pieces = [decoded[:, 0, :, :-1]]
    for index in range(1, count):
        transition = (
            theta * decoded[:, index - 1, :, -1:]
            + (1.0 - theta) * decoded[:, index, :, :1]
        )
        pieces.append(transition)
        end = frames - 1 if index < count - 1 else frames
        pieces.append(decoded[:, index, :, 1:end])
    return torch.cat(pieces, dim=2)


def fixed_rate_confidence(
    latents: torch.Tensor,
    *,
    window_frames: int = 6,
    stride_frames: int = 5,
    pattern: str = "uniform",
) -> torch.Tensor:
    """Keep exactly stride/window of coordinates over each six-GOP cycle."""
    if latents.ndim != 3:
        raise ValueError("latents must be BGL")
    if pattern not in {"uniform", "prefix"}:
        raise ValueError("pattern must be uniform or prefix")
    batch, count, budget = latents.shape
    numerator = budget * stride_frames
    low, remainder = divmod(numerator, window_frames)
    confidence = latents.new_zeros(latents.shape)
    for index in range(count):
        keep = low + (1 if index % window_frames < remainder else 0)
        if pattern == "prefix":
            selected = torch.arange(keep, device=latents.device)
        else:
            # Midpoint sampling distributes missing coordinates across every
            # channel, latent-time plane, and spatial row instead of deleting
            # most of the final feature channel.
            selected = torch.floor(
                (torch.arange(keep, device=latents.device) + 0.5) * budget / keep
            ).long()
        confidence[:, index, selected] = 1.0
    return confidence


def overlap_seams(total_frames: int, stride_frames: int = 5) -> list[int]:
    return list(range(stride_frames, total_frames, stride_frames))


def standard_seams(total_frames: int, gop_frames: int = 6) -> list[int]:
    return list(range(gop_frames, total_frames, gop_frames))


def transition_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    seams: list[int],
    device: torch.device,
    *,
    include_lpips: bool,
) -> dict[str, float]:
    """Score both temporal edges adjacent to each transition frame."""
    recon = recon.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    if recon.shape != target.shape:
        raise ValueError("reconstruction and target must have equal shapes")
    valid = [index for index in seams if 0 < index < recon.shape[2] - 1]
    if not valid:
        raise ValueError("no complete transition neighborhoods")
    residual = recon - target
    delta_error = (recon[:, :, 1:] - recon[:, :, :-1]) - (
        target[:, :, 1:] - target[:, :, :-1]
    )
    incoming = torch.stack([delta_error[:, :, index - 1] for index in valid], dim=2)
    outgoing = torch.stack([delta_error[:, :, index] for index in valid], dim=2)
    two_sided = torch.cat((incoming, outgoing), dim=2)
    acceleration_error = (delta_error[:, :, 1:] - delta_error[:, :, :-1])
    acceleration = torch.stack(
        [acceleration_error[:, :, index - 1] for index in valid], dim=2
    )
    lowpass = F.avg_pool2d(
        residual.permute(0, 2, 1, 3, 4).flatten(0, 1),
        kernel_size=9,
        stride=1,
        padding=4,
    ).reshape(residual.shape[0], residual.shape[2], residual.shape[1], *residual.shape[-2:])
    lowpass = lowpass.permute(0, 2, 1, 3, 4)
    lowpass_delta = lowpass[:, :, 1:] - lowpass[:, :, :-1]
    lowpass_edges = torch.cat(
        (
            torch.stack([lowpass_delta[:, :, index - 1] for index in valid], dim=2),
            torch.stack([lowpass_delta[:, :, index] for index in valid], dim=2),
        ),
        dim=2,
    )
    within_mask = torch.ones(delta_error.shape[2], dtype=torch.bool, device=device)
    excluded = sorted({edge for index in valid for edge in (index - 1, index)})
    within_mask[torch.tensor(excluded, device=device)] = False
    within = delta_error[:, :, within_mask]
    mse = float(F.mse_loss(recon, target))
    result = {
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": simple_ssim(recon, target),
        "l1": float(F.l1_loss(recon, target)),
        "temporal_delta": float(delta_error.abs().mean()),
        "seam_in_delta": float(incoming.abs().mean()),
        "seam_out_delta": float(outgoing.abs().mean()),
        "seam_two_sided_delta": float(two_sided.abs().mean()),
        "seam_lowpass_delta": float(lowpass_edges.abs().mean()),
        "seam_acceleration": float(acceleration.abs().mean()),
        "within_delta": float(within.abs().mean()),
    }
    result["seam_ratio"] = result["seam_two_sided_delta"] / max(
        result["within_delta"], 1e-12
    )
    if include_lpips:
        result["lpips"] = lpips_metric(recon, target, device)
        recon_deltas = torch.cat(
            (
                torch.stack(
                    [recon[:, :, index] - recon[:, :, index - 1] for index in valid],
                    dim=2,
                ),
                torch.stack(
                    [recon[:, :, index + 1] - recon[:, :, index] for index in valid],
                    dim=2,
                ),
            ),
            dim=2,
        )
        target_deltas = torch.cat(
            (
                torch.stack(
                    [target[:, :, index] - target[:, :, index - 1] for index in valid],
                    dim=2,
                ),
                torch.stack(
                    [target[:, :, index + 1] - target[:, :, index] for index in valid],
                    dim=2,
                ),
            ),
            dim=2,
        )
        result["seam_delta_lpips"] = lpips_metric(
            (0.5 * recon_deltas + 0.5).clamp(0, 1),
            (0.5 * target_deltas + 0.5).clamp(0, 1),
            device,
        )
    return result


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def decode_standard(model, source: torch.Tensor) -> torch.Tensor:
    mode = model.mode
    gops = split_video_gops(source, mode.gop_frames)
    batch, count = gops.shape[:2]
    latents = model.encoder(gops.flatten(0, 1))
    decoded = model.decoder(latents).reshape(batch, count, 3, mode.gop_frames, mode.height, mode.width)
    return join_video_gops(decoded)


def encode_overlapping(model, source: torch.Tensor) -> torch.Tensor:
    windows = overlapping_windows(source, model.mode.gop_frames, model.mode.gop_frames - 1)
    batch, count = windows.shape[:2]
    latents = model.encoder(windows.flatten(0, 1))
    return latents.reshape(batch, count, -1)


def decode_overlapping(model, latents: torch.Tensor, pattern: str | None) -> torch.Tensor:
    batch, count, budget = latents.shape
    if pattern is None:
        weights = torch.ones_like(latents)
    else:
        weights = fixed_rate_confidence(latents, pattern=pattern)
    decoded = model.decoder(
        latents.reshape(batch * count, budget),
        weights.reshape(batch * count, budget),
    )
    return decoded.reshape(batch, count, *decoded.shape[1:])


def evaluate_theta_sweep(model, dataset: SequenceCache, device: torch.device, limit: int) -> dict:
    rows = {
        pattern: {str(theta): [] for theta in THETAS}
        for pattern in ("full", "fixed_uniform", "fixed_prefix")
    }
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for index in range(min(limit, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            latents = encode_overlapping(model, source)
            decoded_by_pattern = {
                "full": decode_overlapping(model, latents, None),
                "fixed_uniform": decode_overlapping(model, latents, "uniform"),
                "fixed_prefix": decode_overlapping(model, latents, "prefix"),
            }
            for pattern, decoded in decoded_by_pattern.items():
                for theta in THETAS:
                    recon = fuse_overlapping_gops(decoded, theta)
                    target = source[:, :, : recon.shape[2]]
                    rows[pattern][str(theta)].append(
                        transition_metrics(
                            recon,
                            target,
                            overlap_seams(recon.shape[2]),
                            device,
                            include_lpips=False,
                        )
                    )
    summary = {
        pattern: {theta: aggregate(values) for theta, values in theta_rows.items()}
        for pattern, theta_rows in rows.items()
    }
    summary["selected_theta"] = {
        pattern: min(
            theta_rows,
            key=lambda theta: theta_rows[theta]["seam_two_sided_delta"],
        )
        for pattern, theta_rows in summary.items()
        if pattern != "selected_theta"
    }
    return summary


def evaluate_finalists(
    model,
    dataset: SequenceCache,
    device: torch.device,
    limit: int,
    selected: dict[str, str],
) -> dict:
    candidate_specs = {
        "paper_overlap_full": ("full", 0.5),
        "paper_overlap_fixed_uniform": ("fixed_uniform", 0.5),
        "paper_overlap_fixed_prefix": ("fixed_prefix", 0.5),
    }
    for pattern, theta in selected.items():
        value = float(theta)
        if value != 0.5:
            candidate_specs[f"best_{pattern}"] = (pattern, value)
    rows = {"released_v8": []} | {name: [] for name in candidate_specs}
    disagreement_rows = []
    model.eval()
    with torch.inference_mode():
        for index in range(min(limit, len(dataset))):
            source = dataset[index].unsqueeze(0).to(device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                standard = decode_standard(model, source)
                latents = encode_overlapping(model, source)
                decoded_by_pattern = {
                    "full": decode_overlapping(model, latents, None),
                    "fixed_uniform": decode_overlapping(model, latents, "uniform"),
                    "fixed_prefix": decode_overlapping(model, latents, "prefix"),
                }
            length = 1 + decoded_by_pattern["full"].shape[1] * (model.mode.gop_frames - 1)
            target = source[:, :, :length]
            rows["released_v8"].append(
                transition_metrics(
                    standard[:, :, :length],
                    target,
                    standard_seams(length),
                    device,
                    include_lpips=True,
                )
            )
            full_decoded = decoded_by_pattern["full"]
            disagreement_rows.append(
                float(F.l1_loss(full_decoded[:, :-1, :, -1], full_decoded[:, 1:, :, 0]))
            )
            for name, (pattern, theta) in candidate_specs.items():
                recon = fuse_overlapping_gops(decoded_by_pattern[pattern], theta)
                rows[name].append(
                    transition_metrics(
                        recon,
                        target,
                        overlap_seams(length),
                        device,
                        include_lpips=True,
                    )
                )
    report = {name: aggregate(values) for name, values in rows.items()}
    report["double_decode_transition_disagreement_l1"] = sum(disagreement_rows) / len(
        disagreement_rows
    )
    return report


def render_simpsons(
    model,
    source_path: Path,
    out: Path,
    device: torch.device,
) -> dict:
    mode = model.mode
    frames = int(60 * mode.fps)
    raw = extract_video(source_path, frames, int(mode.fps), mode.width, mode.height)
    source = torch.from_numpy(raw).permute(0, 3, 1, 2).unsqueeze(0)
    source = source.permute(0, 2, 1, 3, 4).to(device).float().div_(255)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        standard = decode_standard(model, source)
        overlap_source = pad_for_overlap_coverage(source)
        latents = encode_overlapping(model, overlap_source)
        full = fuse_overlapping_gops(decode_overlapping(model, latents, None), 0.5)
        fixed = fuse_overlapping_gops(
            decode_overlapping(model, latents, "uniform"), 0.5
        )
        full = full[:, :, :frames]
        fixed = fixed[:, :, :frames]
    metrics = {
        "released_v8": transition_metrics(
            standard, source, standard_seams(frames), device, include_lpips=False
        ),
        "paper_overlap_full": transition_metrics(
            full, source, overlap_seams(frames), device, include_lpips=False
        ),
        "paper_overlap_fixed_uniform": transition_metrics(
            fixed, source, overlap_seams(frames), device, include_lpips=False
        ),
    }
    panels = [
        ("SOURCE", source.cpu()),
        ("RELEASED V8", standard.cpu()),
        ("PAPER OVERLAP +20%", full.cpu()),
        ("OVERLAP FIXED RATE", fixed.cpu()),
    ]
    del latents, overlap_source, standard, full, fixed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    path = out / "simpsons_60s_paper_frame_overlap_clean.mp4"
    write_labeled_grid_mp4(
        panels,
        path,
        mode.fps,
        columns=2,
    )
    report = {
        "video": str(path),
        "frames": frames,
        "seconds": frames / mode.fps,
        "metrics": metrics,
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt")
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "/pool0/AETV-runs/gop-boundary-data/v8_192x108_8gop_train"
        ),
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
    parser.add_argument(
        "--out", type=Path, default=Path("/pool0/AETV-runs/v8-paper-frame-overlap")
    )
    parser.add_argument("--sweep-sequences", type=int, default=8)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    mode = AETV_MODES["V8"]
    model = load_model(args.checkpoint, mode, device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep = evaluate_theta_sweep(
        model, SequenceCache(args.train_cache), device, args.sweep_sequences
    )
    print(json.dumps({"theta_sweep": sweep}, indent=2), flush=True)
    final = evaluate_finalists(
        model,
        SequenceCache(args.eval_cache),
        device,
        args.eval_sequences,
        sweep["selected_theta"],
    )
    print(json.dumps({"fixed_eval": final}, indent=2), flush=True)
    simpsons = render_simpsons(model, args.simpsons, args.out, device)
    report = {
        "paper": {
            "title": "Perceptual Neural Video Compression with Video Variational Autoencoder at Low Bitrates",
            "status": "under review at ICLR 2026",
            "paper_gop_frames": 9,
            "paper_theta": 0.5,
        },
        "v8_contract": {
            "gop_frames": mode.gop_frames,
            "fps": mode.fps,
            "latents_per_gop": mode.latents_per_gop,
            "baseline_values_per_frame": mode.latents_per_gop / mode.gop_frames,
            "faithful_overlap_values_per_frame": mode.latents_per_gop
            / (mode.gop_frames - 1),
            "faithful_rate_increase": mode.gop_frames / (mode.gop_frames - 1) - 1,
            "fixed_overlap_values_per_gop_average": mode.latents_per_gop
            * (mode.gop_frames - 1)
            / mode.gop_frames,
            "fixed_overlap_requires_packet_repacking": True,
        },
        "theta_sweep_train8": sweep,
        "eval32_clean": final,
        "simpsons": simpsons,
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "simpsons": simpsons}, indent=2))


if __name__ == "__main__":
    main()
