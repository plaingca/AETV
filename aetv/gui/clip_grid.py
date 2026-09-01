"""Streamdeck-style prepared clip cells for the transmit pane."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QMenu, QVBoxLayout, QWidget


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mpeg", ".mpg"}


class ClipCell(QFrame):
    fileDropped = Signal(int, str)
    activated = Signal(int)
    hovered = Signal(int)
    editRequested = Signal(int)
    removeRequested = Signal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.path = ""
        self.ready = False
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(105, 72)
        self.setToolTip("Drop a video here, or click to choose one")
        self.picture = QLabel("+")
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumHeight(42)
        self.name = QLabel(f"Clip {index + 1}")
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self.picture, 1)
        layout.addWidget(self.name)
        self._restyle()

    def set_path(self, path: str) -> None:
        self.path = path
        self.ready = False
        self.picture.setPixmap(QPixmap())
        self.picture.setText("Preparing…" if path else "+")
        self.name.setText(Path(path).stem if path else f"Clip {self.index + 1}")
        self.setToolTip(path or "Drop a video here, or click to choose one")
        self._restyle()

    def set_progress(self, progress: float) -> None:
        self.picture.setText(f"Encoding {round(progress * 100)}%")

    def set_ready(self, preview: np.ndarray) -> None:
        self.ready = True
        frame = np.ascontiguousarray(preview[0] if preview.ndim == 4 else preview)
        height, width = frame.shape[:2]
        image = QImage(
            frame.data, width, height, width * 3, QImage.Format.Format_RGB888
        ).copy()
        self.picture.setText("")
        self.picture.setPixmap(
            QPixmap.fromImage(image).scaled(
                150,
                58,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._restyle()

    def set_error(self, message: str) -> None:
        self.ready = False
        self.picture.setText("Failed")
        self.setToolTip(f"{self.path}\n{message}")
        self._restyle(error=True)

    def clear(self) -> None:
        self.set_path("")

    def _restyle(self, error: bool = False) -> None:
        if error:
            border = "#b3261e"
        elif self.ready:
            border = "#3d8b55"
        else:
            border = "palette(mid)"
        self.setStyleSheet(
            f"ClipCell {{ border: 1px solid {border}; border-radius: 4px; }}"
        )

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in VIDEO_SUFFIXES:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.fileDropped.emit(self.index, urls[0].toLocalFile())
            event.acceptProposedAction()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.index)
        super().mouseReleaseEvent(event)

    def enterEvent(self, event) -> None:
        if self.path:
            self.hovered.emit(self.index)
        super().enterEvent(event)

    def contextMenuEvent(self, event) -> None:
        if not self.path:
            return
        menu = QMenu(self)
        edit = menu.addAction("Edit clip…")
        remove = menu.addAction("Remove clip")
        chosen = menu.exec(event.globalPos())
        if chosen is edit:
            self.editRequested.emit(self.index)
        elif chosen is remove:
            self.removeRequested.emit(self.index)


class ClipGrid(QWidget):
    fileChosen = Signal(int, str)
    activated = Signal(int)
    hovered = Signal(int)
    editRequested = Signal(int)
    removeRequested = Signal(int)

    def __init__(self, count: int = 8, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.cells: list[ClipCell] = []
        for index in range(count):
            cell = ClipCell(index)
            cell.fileDropped.connect(self.fileChosen)
            cell.activated.connect(self.activated)
            cell.hovered.connect(self.hovered)
            cell.editRequested.connect(self.editRequested)
            cell.removeRequested.connect(self.removeRequested)
            layout.addWidget(cell, index // 4, index % 4)
            self.cells.append(cell)
