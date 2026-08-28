"""Webcam and file sources that emit AETV-sized RGB frames."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import AETVModeSpec
from .ffmpeg import ffmpeg_executable


class CameraFrameBuffer:
    """One persistent webcam producer shared by preview and transmission.

    Consumers get independent live cursors into a bounded frame history. A
    slow consumer skips frames instead of holding up camera capture or making
    the other consumer stale.
    """

    def __init__(self, history_frames: int = 48):
        self._history = deque(maxlen=max(2, int(history_frames)))
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._key: tuple[int, str] | None = None
        self._sequence = 0
        self._error: Exception | None = None

    def configure(self, mode: AETVModeSpec, camera: int = 0) -> None:
        """Open the requested camera unless that exact producer is running."""
        key = (int(camera), mode.name)
        with self._lifecycle_lock:
            if self._key == key and self._thread is not None and self._thread.is_alive():
                return
            if not self._stop_locked():
                raise RuntimeError(
                    "previous webcam worker did not stop; refusing to open a second camera backend"
                )
            with self._condition:
                self._history.clear()
                self._error = None
                self._key = key
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(mode, int(camera), self._stop),
                daemon=True,
                name=f"aetv-camera-buffer-{camera}",
            )
            self._thread.start()

    def frames(
        self,
        mode: AETVModeSpec,
        camera: int = 0,
        should_stop=None,
        latest: bool = False,
    ) -> Iterator[np.ndarray]:
        """Yield frames without taking ownership of the camera.

        Ordered consumers such as TX read every available frame. A live
        preview can request ``latest`` so a delayed paint skips directly to
        the newest camera picture instead of replaying stale history.
        """
        self.configure(mode, camera)
        with self._condition:
            cursor = self._sequence
        while should_stop is None or not should_stop():
            with self._condition:
                if latest and self._history and self._history[-1][0] > cursor:
                    available = self._history[-1]
                else:
                    available = next(
                        (
                            (sequence, frame)
                            for sequence, frame in self._history
                            if sequence > cursor
                        ),
                        None,
                    )
                if available is None:
                    if self._error is not None:
                        raise self._error
                    self._condition.wait(timeout=0.2)
                    continue
                cursor, frame = available
            yield frame

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._stop_locked():
                with self._condition:
                    self._key = None
                    self._history.clear()

    def _stop_locked(self) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        if thread is not None and thread.is_alive():
            # Do not forget this worker and start a second native capture
            # stack. Some Windows webcam drivers corrupt the process heap if
            # two overlapping Media Foundation/DirectShow lifecycles race.
            return False
        self._thread = None
        return True

    def _run(self, mode: AETVModeSpec, camera: int, stop: threading.Event) -> None:
        try:
            for frame in iter_webcam(mode, camera=camera):
                if stop.is_set():
                    return
                with self._condition:
                    self._sequence += 1
                    self._history.append((self._sequence, frame))
                    self._condition.notify_all()
        except Exception as error:
            if not stop.is_set():
                with self._condition:
                    self._error = error
                    self._condition.notify_all()


def list_cameras(max_index: int = 8) -> list[dict]:
    """Probe local camera indices. Names are best-effort; Windows has no stable API."""
    import cv2
    import sys

    if sys.platform == "win32":
        # Repeated DirectShow VideoCapture open/close calls are not merely slow:
        # several consumer webcam drivers corrupt the process heap during
        # enumeration.  Offer stable index choices and let the persistent
        # preview open only the selected device.
        return [{"index": index, "name": f"Camera {index}"} for index in range(min(4, max_index))]

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    found: list[dict] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index, backend)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            continue
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        size = f"{width}x{height}" if width and height else "camera"
        found.append({"index": index, "name": f"Camera {index} ({size})"})
    return found


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
    """Yield the newest camera frame on a monotonic mode-rate clock.

    Camera backends commonly queue frames internally. Reading only when the
    encoder asks for a frame therefore returns old frames in a burst whenever
    the TX producer was blocked by its rolling buffer. A dedicated reader
    continuously drains the driver; this sampler selects its newest frame at
    uniform wall-clock intervals and never tries to catch up missed ticks.
    """
    import cv2
    import sys

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    condition = threading.Condition()
    stop = threading.Event()
    state: dict = {"frame": None, "sequence": 0, "error": None}
    # Sampling must not stop while the neural encoder is using the yielded
    # GOP. Keep a small, bounded clocked queue so capture for GOP N+1 overlaps
    # encoding GOP N without accumulating stale camera video.
    sampled: queue.Queue[np.ndarray] = queue.Queue(
        maxsize=max(2, int(mode.gop_frames) * 2)
    )

    def drain_camera() -> None:
        capture = cv2.VideoCapture(camera, backend)
        if not capture.isOpened():
            with condition:
                state["error"] = RuntimeError(f"could not open webcam index {camera}")
                condition.notify_all()
            capture.release()
            return
        # Ask for a native rate comfortably above AETV's sampling rate and
        # minimize backend buffering where the property is honored.
        capture.set(cv2.CAP_PROP_FPS, max(30.0, float(mode.fps)))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    with condition:
                        state["error"] = RuntimeError("webcam stopped delivering frames")
                        condition.notify_all()
                    return
                with condition:
                    state["frame"] = frame
                    state["sequence"] += 1
                    condition.notify_all()
        finally:
            capture.release()

    reader = threading.Thread(
        target=drain_camera, daemon=True, name=f"aetv-camera-{camera}"
    )
    reader.start()

    def sample_camera() -> None:
        try:
            with condition:
                ready_until = time.monotonic() + 10.0
                while state["frame"] is None and state["error"] is None:
                    remaining = ready_until - time.monotonic()
                    if remaining <= 0:
                        state["error"] = RuntimeError(
                            f"webcam index {camera} did not deliver a frame"
                        )
                        condition.notify_all()
                        return
                    condition.wait(remaining)
                if state["error"] is not None:
                    return

            period = 1.0 / mode.fps
            next_frame_at = time.monotonic()
            while not stop.is_set():
                delay = next_frame_at - time.monotonic()
                if delay > 0 and stop.wait(delay):
                    return
                with condition:
                    if state["error"] is not None:
                        return
                    frame = state["frame"].copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                value = resize_frame(rgb, mode.width, mode.height)
                while not stop.is_set():
                    try:
                        sampled.put_nowait(value)
                        break
                    except queue.Full:
                        # The consumer fell behind by more than two GOPs.
                        # Discard the oldest picture and stay live.
                        try:
                            sampled.get_nowait()
                        except queue.Empty:
                            pass
                next_frame_at += period
                next_frame_at = max(next_frame_at, time.monotonic())
        except Exception as error:
            with condition:
                state["error"] = error
                condition.notify_all()

    sampler = threading.Thread(
        target=sample_camera, daemon=True, name=f"aetv-camera-sampler-{camera}"
    )
    sampler.start()
    try:
        limit = None if duration_s is None else int(round(duration_s * mode.fps))
        produced = 0
        while limit is None or produced < limit:
            try:
                frame = sampled.get(timeout=0.2)
            except queue.Empty:
                with condition:
                    error = state["error"]
                if error is not None:
                    raise error
                if not sampler.is_alive():
                    raise RuntimeError(f"webcam index {camera} sampler stopped")
                continue
            yield frame
            produced += 1
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        reader.join(timeout=2.0)
        # VideoCapture is owned and released by drain_camera. Releasing it
        # here from another thread while read() is active can corrupt native
        # Media Foundation/DirectShow state. A stuck reader is a daemon and a
        # later CameraFrameBuffer.configure() will refuse to overlap it.


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
        # The duration control describes the transmission length, not a
        # maximum imposed by the selected clip. Repeat a short file so ffmpeg
        # can always supply the requested whole GOPs.
        ffmpeg_executable(), "-v", "error", "-stream_loop", "-1",
        "-ss", f"{start_s:.3f}", "-i", str(path),
        "-vf", video_filter, "-frames:v", str(frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
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
        ffmpeg_executable(),
        "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.run(
        command,
        input=frames.tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])


def write_video_smoke_test(path: Path) -> None:
    """Exercise the same FFmpeg path used by the receive Save video button."""
    height, width = 32, 48
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    frames = np.empty((3, height, width, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, :, :, 0] = x + index * 20
        frames[index, :, :, 1] = y + index * 30
        frames[index, :, :, 2] = 128
    write_mp4(frames, path, fps=6.0)
    payload = path.read_bytes()
    if len(payload) < 32 or b"ftyp" not in payload[:32]:
        raise RuntimeError(f"FFmpeg smoke test did not create a valid MP4: {path}")


def write_side_by_side(left: np.ndarray, right: np.ndarray, path: Path, fps: float) -> None:
    if left.shape != right.shape:
        raise ValueError(f"frame stacks differ: {left.shape} vs {right.shape}")
    write_mp4(np.concatenate([left, right], axis=2), path, fps)
