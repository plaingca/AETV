"""Webcam and file sources that emit AETV-sized RGB frames."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import AETVModeSpec


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop after covering the target size. Uses OpenCV if present."""
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for webcam and frame resize") from error
    src_h, src_w = frame.shape[:2]
    scale = max(height / src_h, width / src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    return resized[top : top + height, left : left + width]


def iter_webcam(
    mode: AETVModeSpec,
    camera: int = 0,
    duration_s: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield (H, W, 3) uint8 frames from a local camera at the mode frame rate."""
    import cv2

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"could not open webcam index {camera}")
    capture.set(cv2.CAP_PROP_FPS, mode.fps)
    try:
        limit = None if duration_s is None else int(round(duration_s * mode.fps))
        produced = 0
        while limit is None or produced < limit:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("webcam stopped delivering frames")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield resize_frame(rgb, mode.width, mode.height)
            produced += 1
    finally:
        capture.release()


def iter_video_file(
    path: str | Path,
    mode: AETVModeSpec,
    start_s: float = 0.0,
    frames: int | None = None,
) -> np.ndarray:
    """Decode a video file into (T, H, W, 3) uint8 at the mode geometry."""
    if frames is None:
        raise ValueError("iter_video_file requires an exact frame count")
    video_filter = (
        f"fps={mode.fps},"
        f"scale={mode.width}:{mode.height}:force_original_aspect_ratio=increase,"
        f"crop={mode.width}:{mode.height}"
    )
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(path),
        "-vf", video_filter, "-frames:v", str(frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    expected = frames * mode.height * mode.width * 3
    if proc.returncode or len(proc.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg returned {proc.returncode} with {len(proc.stdout)}/{expected} bytes: "
            + proc.stderr.decode("utf-8", "replace")[-2000:]
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).copy().reshape(
        frames, mode.height, mode.width, 3
    )


def collect_gops(
    frames: np.ndarray | Iterator[np.ndarray],
    mode: AETVModeSpec,
) -> np.ndarray:
    """Stack frames into complete GOPs, dropping a trailing partial GOP."""
    if isinstance(frames, np.ndarray):
        stack = frames
    else:
        stack = np.stack(list(frames), axis=0)
    n_gops = stack.shape[0] // mode.gop_frames
    if n_gops < 1:
        raise ValueError(
            f"need at least {mode.gop_frames} frames for one GOP, got {stack.shape[0]}"
        )
    return stack[: n_gops * mode.gop_frames]


def write_mp4(frames: np.ndarray, path: Path, fps: float) -> None:
    """Write (T, H, W, 3) uint8 frames to an H.264 MP4."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3), got {frames.shape}")
    count, height, width, _ = frames.shape
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.run(command, input=frames.tobytes(), stderr=subprocess.PIPE, timeout=600)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])


def write_side_by_side(left: np.ndarray, right: np.ndarray, path: Path, fps: float) -> None:
    if left.shape != right.shape:
        raise ValueError(f"frame stacks differ: {left.shape} vs {right.shape}")
    write_mp4(np.concatenate([left, right], axis=2), path, fps)
