#!/usr/bin/env python3
"""Verify every staged Hub checkpoint against the release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from aetv.config import AETV_MODES
from aetv.models import AETVAutoencoder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    for filename, expected in manifest["files"].items():
        path = args.release_dir / filename
        assert path.stat().st_size == expected["bytes"], filename
        assert sha256_file(path) == expected["sha256"], filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mode_name = payload.get("mode") or payload.get("args", {}).get("mode")
        assert mode_name in AETV_MODES, (filename, mode_name)
        train_args = payload.get("args", {}) or {}
        model = AETVAutoencoder(
            mode=AETV_MODES[mode_name],
            width=int(train_args.get("model_width", 128)),
            latent_channels=int(train_args.get("latent_channels", 3)),
            compact=bool(train_args.get("compact", False)),
            causal=AETV_MODES[mode_name].causal,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        print(f"ok {filename} ({mode_name}, step {payload.get('step')})", flush=True)


if __name__ == "__main__":
    main()
