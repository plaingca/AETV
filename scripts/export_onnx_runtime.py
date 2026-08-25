#!/usr/bin/env python3
"""Export native AETV checkpoints and optionally publish runtime graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.export_onnx import export_checkpoint, publish_runtime_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-name",
        help="output stem; valid only when exporting one checkpoint",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default="AETV/AETV")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--create-pr", action="store_true")
    ns = parser.parse_args()
    if ns.runtime_name and len(ns.checkpoint) != 1:
        parser.error("--runtime-name requires exactly one checkpoint")
    manifests = [
        export_checkpoint(checkpoint, ns.output, runtime_name=ns.runtime_name)
        for checkpoint in ns.checkpoint
    ]
    for manifest in manifests:
        print(manifest)
    if ns.push_to_hub:
        print(
            publish_runtime_bundles(
                manifests,
                repo_id=ns.repo_id,
                revision=ns.revision,
                create_pr=ns.create_pr,
            )
        )


if __name__ == "__main__":
    main()
