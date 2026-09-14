#!/usr/bin/env python3
"""Download and checksum-verify the portable ONNX runtime bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.codec import download_runtime_bundle
from aetv.config import RELEASE_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for mode in RELEASE_MODES:
        print(f"{mode}: {download_runtime_bundle(mode, destination=args.output)}")


if __name__ == "__main__":
    main()
