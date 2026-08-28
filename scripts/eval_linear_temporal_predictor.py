#!/usr/bin/env python3
"""Fit a closed-form receiver-side predictor for the transition frame."""

from pathlib import Path
import torch

from aetv.config import AETV_MODES
from scripts.experiment_gop_boundaries import (
    DEFAULT_CELLS, SequenceCache, decode_cached_sequence, join_gops, load_model,
)


def features(video: torch.Tensor) -> torch.Tensor:
    # Use every decoded frame in the two-GOP window plus a bias. The predictor
    # is linear in RGB values but is allowed to mix time and color channels.
    b, c, t, h, w = video.shape
    return torch.cat((video.permute(0, 2, 1, 3, 4).reshape(b, t * c, h, w),
                      torch.ones(b, 1, h, w, device=video.device)), dim=1)


def main() -> None:
    mode, device = AETV_MODES["V8"], torch.device("cuda")
    train_ds = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_train"))
    eval_ds = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    train_rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/train-runtime-rx.pt", map_location="cpu", weights_only=False)
    eval_rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/eval-runtime-rx.pt", map_location="cpu", weights_only=False)
    model = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            train_x, train_y = [], []
            for i in range(len(train_ds)):
                s = train_ds[i].unsqueeze(0).to(device)
                b = decode_cached_sequence(model, train_rx, cell, i, mode, device)
                train_x.append(features(b).squeeze(0).permute(1, 2, 0).reshape(-1, 37))
                train_y.append(s[:, :, 6].squeeze(0).permute(1, 2, 0).reshape(-1, 3))
            x, y = torch.cat(train_x), torch.cat(train_y)
            xtx = x.T @ x
            ridge = 1e-3 * torch.eye(37, device=device)
            weights = torch.linalg.solve(xtx + ridge, x.T @ y)
            rows = []
            for i in range(len(eval_ds)):
                s = eval_ds[i].unsqueeze(0).to(device)
                b = decode_cached_sequence(model, eval_rx, cell, i, mode, device)
                q = b.clone()
                pred = (features(b).permute(0, 2, 3, 1) @ weights).permute(0, 3, 1, 2).clamp(0, 1)
                q[:, :, 6] = pred
                base_delta = (b[:, :, 6] - b[:, :, 5]) - (s[:, :, 6] - s[:, :, 5])
                cand_delta = (q[:, :, 6] - q[:, :, 5]) - (s[:, :, 6] - s[:, :, 5])
                rows.append((base_delta.abs().mean().item(), cand_delta.abs().mean().item(),
                             (b - s).square().mean().item(), (q - s).square().mean().item()))
            base = sum(r[0] for r in rows) / len(rows); cand = sum(r[1] for r in rows) / len(rows)
            bm = sum(r[2] for r in rows) / len(rows); cm = sum(r[3] for r in rows) / len(rows)
            print(f"{cell.label}: reduction={100*(base-cand)/base:.1f}% baseline_psnr={10*torch.log10(torch.tensor(1/bm)).item():.3f} candidate_psnr={10*torch.log10(torch.tensor(1/cm)).item():.3f}", flush=True)


if __name__ == "__main__": main()
