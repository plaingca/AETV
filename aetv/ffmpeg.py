"""Resolve the FFmpeg executable used by AETV runtime video operations."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def ffmpeg_executable() -> str:
    """Return an explicit FFmpeg path for source and packaged executions."""
    override = os.environ.get("AETV_FFMPEG", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise RuntimeError(f"AETV_FFMPEG does not exist: {candidate}")
        return str(candidate.resolve())

    if getattr(sys, "frozen", False):
        name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        candidate = Path(sys.executable).resolve().parent / name
        if not candidate.is_file():
            raise RuntimeError(
                f"portable AETV is missing its bundled FFmpeg executable: {candidate}"
            )
        return str(candidate)

    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as error:
        raise RuntimeError("AETV could not resolve its FFmpeg dependency") from error
    if not candidate.is_file():
        raise RuntimeError(f"AETV FFmpeg executable does not exist: {candidate}")
    return str(candidate.resolve())
