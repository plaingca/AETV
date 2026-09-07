"""Persistent prepared clips, invalidated by source, edit, or model changes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from .source import PreparedClip


def clip_cache_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return root / "AETV" / "prepared-clips"
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "aetv" / "prepared-clips"


def _file_signature(path: str | Path) -> tuple[str, int, int]:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return os.path.normcase(str(resolved)), stat.st_size, stat.st_mtime_ns


def prepared_clip_path(codec, source: str, count: int, start_s: float, framing: str) -> Path | None:
    checkpoint = getattr(codec, "checkpoint_path", None)
    if checkpoint is None:
        return None
    try:
        model_files = [_file_signature(checkpoint)]
        if getattr(codec, "backend", "") == "onnxruntime":
            model_files.append(_file_signature(Path(checkpoint).with_name(codec.args["encoder"])))
        identity = {
            # Bump when preprocessing or the cached array contract changes.
            "format": 1,
            "source": _file_signature(source),
            "model": model_files,
            "backend": getattr(codec, "backend", ""),
            "mode": codec.mode.name,
            "gops": count,
            "start_s": float(start_s),
            "framing": framing,
        }
    except (OSError, KeyError, ValueError):
        return None
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return clip_cache_dir() / f"{key}.npz"


def load_prepared_clip(cache: Path | None, source: str, mode, count: int, start_s: float) -> PreparedClip | None:
    if cache is None:
        return None
    try:
        with np.load(cache, allow_pickle=False) as saved:
            latents = saved["latents"]
            preview = saved["preview"]
        if (
            latents.shape != (count, mode.latents_per_gop)
            or latents.dtype != np.float32
            or not np.all(np.isfinite(latents))
            or preview.shape != (min(8, count * mode.gop_frames), mode.height, mode.width, 3)
            or preview.dtype != np.uint8
        ):
            return None
        return PreparedClip(str(source), mode.name, tuple(latents), preview, float(start_s))
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return None


def save_prepared_clip(cache: Path, prepared: PreparedClip) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=".clip-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle, latents=np.asarray(prepared.latents, dtype=np.float32),
                preview=prepared.preview_frames,
            )
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
