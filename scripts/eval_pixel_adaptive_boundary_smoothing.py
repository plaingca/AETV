#!/usr/bin/env python3
"""Evaluate per-pixel step-gated symmetric boundary smoothing."""

from pathlib import Path
import torch
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache, decode_cached_sequence, load_model
from aetv.config import AETV_MODES


def main() -> None:
    mode, device = AETV_MODES["V8"], torch.device("cuda")
    ds = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/eval-runtime-rx.pt", map_location="cpu", weights_only=False)
    model = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            t = torch.stack([ds[i].to(device) for i in range(len(ds))])
            base = torch.stack([decode_cached_sequence(model, rx, cell, i, mode, device).squeeze(0) for i in range(len(ds))])
            step = (base[:, :, 6] - base[:, :, 5]).abs()
            base_err = (((base[:, :, 6] - base[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean()).item()
            best = None
            for threshold in torch.linspace(0, float(step.max()), 41):
                q = base.clone(); use = step >= threshold
                f5, f6 = q[:, :, 5].clone(), q[:, :, 6].clone()
                q[:, :, 5] = torch.where(use, 0.5 * f5 + 0.5 * f6, f5)
                q[:, :, 6] = torch.where(use, 0.5 * f6 + 0.5 * f5, f6)
                error = (((q[:, :, 6] - q[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean()).item()
                if best is None or error < best[0]: best = error, float(threshold), int(use.float().mean().item()*100)
            print(f"{cell.label}: baseline={base_err:.6f} threshold={best[1]:.4f} pixels={best[2]}% reduction={100*(base_err-best[0])/base_err:.1f}%",flush=True)


if __name__ == "__main__": main()
