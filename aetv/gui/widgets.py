"""Shared GUI widgets: letterboxed video, PTT lamp, log, eliding labels."""

from __future__ import annotations

from collections import deque
import math
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QWidget,
)

CANVAS = QColor("#202024")
CANVAS_TEXT = QColor("#888888")


def blend_gop_boundary(
    previous: np.ndarray | None,
    frames: np.ndarray,
    transition_frames: int = 4,
) -> np.ndarray:
    """Fade an independently decoded GOP onto the previous final frame.

    Rather than duplicating or inserting pictures, apply the first-frame
    reconstruction offset to the beginning of the new GOP and taper it to
    zero.  Each new frame keeps its own motion, the stream stays at its native
    frame rate, and the one-second codec boundary becomes a short transition.
    """
    values = np.asarray(frames)
    if (
        previous is None
        or values.ndim != 4
        or values.shape[0] == 0
        or previous.shape != values.shape[1:]
        or transition_frames <= 0
    ):
        return values
    count = min(int(transition_frames), len(values))
    output = values.copy()
    correction = previous.astype(np.float32) - values[0].astype(np.float32)
    # Leave 20% of the new reconstruction visible on its first frame, then
    # smoothly remove the concealment over about one third of a second at 12 fps.
    weights = np.linspace(0.8, 0.0, count, endpoint=True, dtype=np.float32)
    for index, weight in enumerate(weights):
        blended = values[index].astype(np.float32) + weight * correction
        output[index] = np.clip(np.rint(blended), 0, 255).astype(values.dtype)
    return output


class ElidingLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full = text
        self._apply()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply()

    def _apply(self) -> None:
        metrics = self.fontMetrics()
        super().setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, max(1, self.width())))


class VideoView(QWidget):
    """Letterboxed RGB preview. Never rescales with a painter stretch of a tiny pixmap."""

    def __init__(self, placeholder: str = "No video", parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._pixmap = QPixmap()
        self._frames = deque(maxlen=120)
        self._last_enqueued_frame: np.ndarray | None = None
        self._fps = 12.0
        self._playout_deadline: float | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance_frame)
        self.setMinimumSize(240, 135)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def clear(self) -> None:
        self._timer.stop()
        self._playout_deadline = None
        self._frames.clear()
        self._last_enqueued_frame = None
        self._pixmap = QPixmap()
        self.update()

    def set_rgb(self, frames: np.ndarray, fps: float | None = None) -> None:
        if frames is None or frames.size == 0:
            self.clear()
            return
        if fps is not None and fps > 0:
            self._fps = float(fps)
        if frames.ndim == 4:
            # A GOP is a new presentation window. Do not let delayed GUI
            # paints turn into seconds of stale camera or receive playback.
            self._frames.clear()
            self._frames.extend(np.ascontiguousarray(frame).copy() for frame in frames)
            if not self._timer.isActive():
                self._start_playout()
            return
        # A live single-frame preview always supersedes queued GOP playback.
        self._timer.stop()
        self._playout_deadline = None
        self._frames.clear()
        self._show_frame(frames)

    def enqueue_rgb(
        self,
        frames: np.ndarray,
        *,
        fps: float,
        prebuffer_frames: int = 24,
        boundary_blend_frames: int = 0,
        max_queue_frames: int | None = None,
    ) -> None:
        """Queue decoded frames for clocked, jitter-buffered receive playout.

        Boundary concealment is opt-in. Cross-fading independent GOP renders
        hides a luminance jump but also creates the visible multi-frame smearing
        that makes motion look as though it leaked across codec boundaries.
        """
        if frames is None or frames.size == 0:
            return
        if fps > 0:
            self._fps = float(fps)
        values = frames if frames.ndim == 4 else frames[np.newaxis, ...]
        values = blend_gop_boundary(
            self._last_enqueued_frame,
            values,
            transition_frames=boundary_blend_frames,
        )
        self._last_enqueued_frame = np.ascontiguousarray(values[-1]).copy()
        incoming = [np.ascontiguousarray(frame).copy() for frame in values]
        if max_queue_frames is not None:
            queue_limit = max(len(incoming), int(max_queue_frames))
            stale = max(0, len(self._frames) + len(incoming) - queue_limit)
            for _ in range(min(stale, len(self._frames))):
                self._frames.popleft()
        self._frames.extend(incoming)
        if self._timer.isActive():
            return
        # After startup or an underrun, hold the last picture until enough
        # decoded material is available to ride through the GUI's batched GOP
        # delivery. Playback itself remains locked to the advertised frame rate.
        if len(self._frames) < max(1, int(prebuffer_frames)):
            return
        self._start_playout()

    def _start_playout(self) -> None:
        self._playout_deadline = time.monotonic()
        self._advance_frame()

    def _advance_frame(self) -> None:
        if not self._frames:
            self._timer.stop()
            self._playout_deadline = None
            return
        self._show_frame(self._frames.popleft())
        now = time.monotonic()
        period = 1.0 / max(self._fps, 1e-6)
        deadline = (
            self._playout_deadline + period
            if self._playout_deadline is not None
            else now + period
        )
        # Never burst frames to catch up after a delayed GUI paint. The next
        # enqueue trims stale backlog, while this clock keeps visible cadence
        # smooth and alternates integer timer delays to represent 6/12 fps
        # without long-term rounding drift.
        if deadline <= now:
            deadline = now + period
        self._playout_deadline = deadline
        self._timer.start(max(1, math.ceil(1000.0 * (deadline - now))))

    def _show_frame(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(frame)
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), CANVAS)
        if self._pixmap.isNull():
            painter.setPen(CANVAS_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        target = self.rect().adjusted(1, 1, -1, -1)
        scaled = self._pixmap.scaled(
            target.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


class PttLamp(QLabel):
    def __init__(self, parent=None):
        super().__init__("PTT idle", parent)
        self.set_keyed(False)

    def set_keyed(self, on: bool) -> None:
        if on:
            self.setText("  PTT  ")
            self.setStyleSheet(
                "QLabel { background: #b3261e; color: #ffffff; font-weight: 700; "
                "padding: 2px 10px; border-radius: 3px; }"
            )
        else:
            self.setText("PTT idle")
            self.setStyleSheet(
                "QLabel { background: transparent; color: palette(mid); padding: 2px 10px; }"
            )


class LogPane(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setPlaceholderText("Station log")
        metrics = self.fontMetrics()
        self.setMinimumHeight(metrics.lineSpacing() * 6 + 8)

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class HLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
