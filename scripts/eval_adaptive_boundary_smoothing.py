#!/usr/bin/env python3
"""Search a receiver-observable boundary-magnitude gate for smoothing."""

from pathlib import Path
import torch

from aetv.config import AETV_MODES
from scripts.experiment_gop_boundaries import DEFAULT_CELLS, SequenceCache, decode_cached_sequence, load_model


def main() -> None:
    mode, device = AETV_MODES["V8"], torch.device("cuda")
    ds = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/eval-runtime-rx.pt", map_location="cpu", weights_only=False)
    model = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    for cell in DEFAULT_CELLS:
        with torch.inference_mode():
            targets, bases = [], []
            for i in range(len(ds)):
                targets.append(ds[i].to(device))
                bases.append(decode_cached_sequence(model, rx, cell, i, mode, device).squeeze(0))
        t, base = torch.stack(targets), torch.stack(bases)
        step = (base[:, :, 6] - base[:, :, 5]).abs().mean(dim=(1, 2, 3))
        base_err = (((base[:, :, 6] - base[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean()).item()
        best = None
        for threshold in torch.linspace(float(step.min()), float(step.max()), 81):
            q = base.clone()
            use = step >= threshold
            f5, f6 = q[:, :, 5].clone(), q[:, :, 6].clone()
            q[use, :, 5] = 0.5 * f5[use] + 0.5 * f6[use]
            q[use, :, 6] = 0.5 * f6[use] + 0.5 * f5[use]
            error = (((q[:, :, 6] - q[:, :, 5]) - (t[:, :, 6] - t[:, :, 5])).abs().mean()).item()
            if best is None or error < best[0]:
                best = error, float(threshold), int(use.sum())
        print(f"{cell.label}: baseline={base_err:.6f} gate_threshold={best[1]:.5f} used={best[2]}/{len(ds)} reduction={100*(base_err-best[0])/base_err:.1f}%", flush=True)


if __name__ == "__main__":
    main()
