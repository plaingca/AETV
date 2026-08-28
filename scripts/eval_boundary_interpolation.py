#!/usr/bin/env python3
"""Evaluate frame-6 interpolation from the adjacent decoded frames."""

from pathlib import Path
import torch
from aetv.config import AETV_MODES
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache, decode_cached_sequence, load_model


def main() -> None:
    mode, device = AETV_MODES["V8"], torch.device("cuda")
    ds = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/eval-runtime-rx.pt", map_location="cpu", weights_only=False)
    model = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            t = torch.stack([ds[i].to(device) for i in range(len(ds))])
            base = torch.stack([decode_cached_sequence(model, rx, cell, i, mode, device).squeeze(0) for i in range(len(ds))])
            base_err = (((base[:, :, 6] - base[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean()).item()
            best = None
            for alpha in [i / 40 for i in range(41)]:
                q = base.clone()
                prediction = 0.5 * (q[:, :, 5] + q[:, :, 7])
                q[:, :, 6] = (1 - alpha) * q[:, :, 6] + alpha * prediction
                error = (((q[:, :, 6] - q[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean()).item()
                within = (q[:, :, 1:] - q[:, :, :-1]).abs().mean().item()
                if best is None or error < best[0]: best = error, alpha, within, (q - t).square().mean().item()
            print(f"{cell.label}: baseline={base_err:.6f} alpha={best[1]:.2f} reduction={100*(base_err-best[0])/base_err:.1f}% mse={best[3]:.6f}", flush=True)


if __name__ == "__main__": main()
