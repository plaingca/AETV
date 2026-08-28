#!/usr/bin/env python3
"""Build fixed multi-GOP runtime RX caches for decoder-state experiments."""

from pathlib import Path
import argparse
import numpy as np
import torch

from aetv.config import AETV_MODES
from aetv.hfchannel import StreamingChannelEmulator
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from scripts.experiment_gop_boundaries import (
    DEFAULT_CELLS, TX_LEVEL, SequenceCache, encode_independent_gops,
    load_model, profile_for_cell, runtime_channel_seed, runtime_retry_seed,
    split_gops,
)


def transmit(values: np.ndarray, mode, cell, seed: int):
    channel = StreamingChannelEmulator(profile_for_cell(cell), seed=seed, fs=mode.geometry.fs)
    demod = StreamingDemodulator(mode.band, continuous=True, mode_name=mode.name)
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
            for result in demod.feed(impaired[start : start + block]):
                for latent, confidence in zip(result.gops_latents, result.gops_weights):
                    received.append(np.asarray(latent, dtype=np.float32))
                    weights.append(np.asarray(confidence, dtype=np.float32))
    if len(received) != values.shape[0]:
        raise RuntimeError(f"recovered {len(received)}/{values.shape[0]} GOPs")
    return np.stack(received), np.stack(weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gops", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    mode = AETV_MODES["V8"]
    data = SequenceCache(args.source, max_frames=args.gops * mode.gop_frames)
    device = torch.device(args.device)
    model = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    result = {"received": {}, "weights": {}, "metadata": {
        "source": str(args.source.resolve()), "gops": args.gops, "seed": args.seed,
        "cells": [cell.label for cell in DEFAULT_CELLS],
    }}
    for cell_index, cell in enumerate(DEFAULT_CELLS):
        result["received"][cell.label] = []
        result["weights"][cell.label] = []
    with torch.inference_mode():
        for index in range(len(data)):
            source = data[index].unsqueeze(0).to(device)
            gops = source.reshape(1, args.gops, 3, mode.gop_frames, mode.height, mode.width)
            encoded = model.encoder(gops.flatten(0, 1))
            encoded = encoded.reshape(args.gops, -1)
            encoded = encoded.float().cpu().numpy()
            for cell_index, cell in enumerate(DEFAULT_CELLS):
                initial = runtime_channel_seed(args.seed, index, cell_index)
                for attempt in range(32):
                    try:
                        rx, conf = transmit(encoded, mode, cell, runtime_retry_seed(initial, attempt))
                        break
                    except RuntimeError:
                        if attempt == 31:
                            raise
                result["received"][cell.label].append(torch.from_numpy(rx))
                result["weights"][cell.label].append(torch.from_numpy(conf))
            print(f"{index + 1}/{len(data)}", flush=True)
    for field in ("received", "weights"):
        for cell in result[field]:
            result[field][cell] = torch.stack(result[field][cell])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
