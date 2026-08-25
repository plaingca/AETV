#!/usr/bin/env python3
"""Console entry point for crash-isolated Windows audio operations."""

from __future__ import annotations

import sys

from aetv.audio_io import _audio_worker_main


if __name__ == "__main__":
    raise SystemExit(_audio_worker_main(sys.argv[1:]))
