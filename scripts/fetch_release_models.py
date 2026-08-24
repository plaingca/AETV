#!/usr/bin/env python3
"""Download and checksum-verify the two portable-release checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.codec import RELEASE_CHECKPOINTS, download_default_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    for mode in ("V8", "V7"):
        filename = RELEASE_CHECKPOINTS[mode]["filename"]
        path = download_default_checkpoint(mode, destination=args.output / filename)
        print(f"{mode}: {path}")


if __name__ == "__main__":
    main()
