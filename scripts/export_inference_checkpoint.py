#!/usr/bin/env python3
"""Strip optimizer state from a training checkpoint for ham distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="models/v7-flex8k.pt")
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    payload = torch.load(src, map_location="cpu", weights_only=False)
    slim = {
        "mode": payload.get("mode", "V7"),
        "stage": payload.get("stage"),
        "step": payload.get("step"),
        "args": payload.get("args", {}),
        "model_state_dict": payload["model_state_dict"],
        "source_run": src.parent.name,
        "source_file": src.name,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    manifest = {
        "path": str(dst),
        "mode": slim["mode"],
        "step": slim["step"],
        "source_run": slim["source_run"],
        "bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
        "model_width": slim["args"].get("model_width"),
        "latent_channels": slim["args"].get("latent_channels"),
    }
    dst.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
