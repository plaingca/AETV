"""Shared GUI widgets: letterboxed video, PTT lamp, log, eliding labels."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
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
        self.setMinimumSize(240, 135)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def clear(self) -> None:
        self._pixmap = QPixmap()
        self.update()

    def set_rgb(self, frames: np.ndarray) -> None:
        if frames is None or frames.size == 0:
            self.clear()
            return
        frame = frames[-1] if frames.ndim == 4 else frames
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
