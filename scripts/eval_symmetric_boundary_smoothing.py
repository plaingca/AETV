#!/usr/bin/env python3
"""Evaluate symmetric two-frame receiver smoothing on the fixed runtime cache."""

from pathlib import Path
import torch

from aetv.config import AETV_MODES
from scripts.experiment_gop_boundaries import (
    DEFAULT_CELLS, SequenceCache, decode_cached_sequence, load_model, lpips_metric,
)


def main() -> None:
    mode = AETV_MODES["V8"]
    device = torch.device("cuda")
    dataset = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/eval-runtime-rx.pt", map_location="cpu", weights_only=False)
    model = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    for cell in DEFAULT_CELLS:
        recon = []
        target = []
        with torch.inference_mode():
            for index in range(len(dataset)):
                target.append(dataset[index].to(device))
                recon.append(decode_cached_sequence(model, rx, cell, index, mode, device).squeeze(0))
        t = torch.stack(target)
        base = torch.stack(recon)
        base_error = ((base[:, :, 6] - base[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean().item()
        base_mse = (base - t).square().mean().item()
        best = None
        for alpha in [i / 40 for i in range(41)]:
            q = base.clone()
            f5, f6 = q[:, :, 5].clone(), q[:, :, 6].clone()
            q[:, :, 5] = (1 - alpha) * f5 + alpha * f6
            q[:, :, 6] = (1 - alpha) * f6 + alpha * f5
            boundary = ((q[:, :, 6] - q[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean().item()
            mse = (q - t).square().mean().item()
            if best is None or boundary < best[0]:
                best = boundary, alpha, mse
        reduction = 100 * (base_error - best[0]) / base_error
        psnr = 10 * torch.log10(torch.tensor(1 / best[2])).item()
        q = base.clone()
        f5, f6 = q[:, :, 5].clone(), q[:, :, 6].clone()
        q[:, :, 5] = 0.5 * f5 + 0.5 * f6
        q[:, :, 6] = 0.5 * f6 + 0.5 * f5
        delta = q[:, :, 1:] - q[:, :, :-1]
        within = delta[:, :, [j for j in range(11) if j != 5]].abs().mean().item()
        base_delta = base[:, :, 1:] - base[:, :, :-1]
        base_within = base_delta[:, :, [j for j in range(11) if j != 5]].abs().mean().item()
        lp_base = lpips_metric(base, t, device)
        lp_candidate = lpips_metric(q, t, device)
        print(f"{cell.label}: alpha0={base_error:.6f} alpha0.5_reduction={100 * (base_error - ((q[:, :, 6] - q[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean().item()) / base_error:.1f}% oracle_alpha={best[1]:.2f} oracle_reduction={reduction:.1f}% psnr={psnr:.3f} baseline_psnr={10 * torch.log10(torch.tensor(1 / base_mse)).item():.3f} within_change={100 * (within - base_within) / base_within:.1f}% lpips_change={100 * (lp_candidate - lp_base) / lp_base:.1f}%", flush=True)


if __name__ == "__main__":
    main()
