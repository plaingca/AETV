#!/usr/bin/env python3
"""Train a two-sided learned transition corrector on fixed runtime RX data."""

from pathlib import Path
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aetv.config import AETV_MODES
from scripts.experiment_gop_boundaries import (
    DEFAULT_CELLS, SequenceCache, boundary_losses, decode_independent_gops,
    join_gops, load_model,
)


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, width), width)
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, width), width)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class SymmetricBoundaryCorrector(nn.Module):
    def __init__(self, width=128, blocks=8, scale=1, max_residual=0.5, confidence=False):
        super().__init__()
        self.max_residual = max_residual
        self.scale = scale
        self.confidence = confidence
        self.input = nn.Conv2d(38 if confidence else 36, width, 3, padding=1)
        self.body = nn.Sequential(*(Residual(width) for _ in range(blocks)))
        self.output = nn.Conv2d(width, 6, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, previous, current, confidence=None):
        if previous.shape != current.shape or previous.ndim != 5:
            raise ValueError("expected equal BCTHW GOP tensors")
        _, _, _, height, width = current.shape
        x = torch.cat((previous, current), dim=1).flatten(1, 2)
        if self.confidence:
            if confidence is None or confidence.shape[:2] != (x.shape[0], 2):
                raise ValueError("confidence-conditioned corrector needs B,2 confidence")
            x = torch.cat((x, confidence[:, :, None, None].expand(-1, -1, height, width)), dim=1)
        low = (max(1, height // self.scale), max(1, width // self.scale))
        x = F.interpolate(x, low, mode="bilinear", align_corners=False)
        x = self.body(F.silu(self.input(x)))
        residual = F.interpolate(self.output(x), (height, width), mode="bilinear", align_corners=False)
        residual = self.max_residual * torch.tanh(residual)
        return residual[:, :3], residual[:, 3:]


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return index, self.dataset[index]


def apply(corrector, gops, two_sided=True, confidence=None):
    previous, current = gops[:, 0], gops[:, 1]
    previous_delta, current_delta = corrector(previous, current, confidence)
    out = gops.clone()
    if two_sided:
        out[:, 0, :, -1] = (out[:, 0, :, -1] + previous_delta).clamp(0, 1)
    out[:, 1, :, 0] = (out[:, 1, :, 0] + current_delta).clamp(0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", choices=("train",), default="train")
    ap.add_argument("--checkpoint", type=Path, default=Path("models/v8-hf3k-face-gan.pt"))
    ap.add_argument("--data-dir", type=Path, default=Path("runs/gop-boundary-data"))
    ap.add_argument("--rx-cache", type=Path, default=Path("runs/v8-two-gop-boundary-sweep-lr1e5/train-runtime-rx.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/v8-symmetric-boundary-corrector"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--boundary-weight", type=float, default=16.0)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--within-weight", type=float, default=1.0)
    ap.add_argument("--cell", choices=tuple(c.label for c in DEFAULT_CELLS), default="random")
    ap.add_argument("--one-sided", action="store_true")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--max-residual", type=float, default=0.5)
    ap.add_argument("--confidence", action="store_true")
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    mode, device = AETV_MODES["V8"], torch.device(a.device)
    base = load_model(a.checkpoint, mode, device).eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    ds = SequenceCache(a.data_dir / "v8_192x108_3gop_train")
    rx = torch.load(a.rx_cache, map_location="cpu", weights_only=False)
    corrector = SymmetricBoundaryCorrector(a.width, a.blocks, a.scale, a.max_residual, a.confidence).to(device)
    opt = torch.optim.AdamW(corrector.parameters(), lr=a.lr, weight_decay=1e-4)
    loader = DataLoader(IndexedDataset(ds), batch_size=a.batch, shuffle=True, drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(a.seed))
    cells = list(DEFAULT_CELLS); iterator = iter(loader)
    for step in range(1, a.steps + 1):
        try: sample_indices, source = next(iterator)
        except StopIteration: iterator = iter(loader); sample_indices, source = next(iterator)
        source = source.to(device).float()
        if a.cell == "random":
            indices = torch.randint(0, len(cells), (source.shape[0],))
        else:
            indices = torch.full((source.shape[0],), next(i for i, c in enumerate(cells) if c.label == a.cell), dtype=torch.long)
        base_rows = []
        for row, index in enumerate(indices.tolist()):
            cell = cells[index]
            source_index = int(sample_indices[row])
            received = rx["received"][cell.label][source_index].unsqueeze(0).to(device)
            weights = rx["weights"][cell.label][source_index].unsqueeze(0).to(device)
            base_rows.append(decode_independent_gops(base, received, weights, mode).reshape(1, 2, 3, 6, 108, 192))
        gops = torch.cat(base_rows, dim=0)
        confidence = torch.stack([
            rx["weights"][cells[int(index)].label][int(source_index)].float().mean(dim=1)
            for index, source_index in zip(indices.tolist(), sample_indices.tolist())
        ]).to(device)
        corrected = apply(corrector, gops, not a.one_sided, confidence if a.confidence else None)
        recon = join_gops(corrected.flatten(0, 1), source.shape[0], 2)
        base_recon = join_gops(gops.flatten(0, 1), source.shape[0], 2)
        boundary = boundary_losses(recon, source, 6)
        loss = (F.l1_loss(recon, source) + a.anchor_weight * F.l1_loss(recon, base_recon)
                + a.boundary_weight * boundary["boundary_rgb_delta"]
                + a.within_weight * boundary["within_gop_temporal_error"])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0); opt.step()
        if step == 1 or step % a.log_interval == 0 or step == a.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "boundary": float(boundary["boundary_rgb_delta"].detach())}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "aetv-symmetric-boundary-corrector", "cell": a.cell, "two_sided": not a.one_sided, "confidence": a.confidence, "config": {"width": a.width, "blocks": a.blocks, "scale": a.scale, "max_residual": a.max_residual, "confidence": a.confidence}, "state_dict": corrector.state_dict()}, a.out / "corrector.pt")


if __name__ == "__main__": main()
