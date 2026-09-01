"""Non-destructive trim and resize editor for transmit clip-bank entries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPolygon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
)

from aetv.source import ClipEdit, frame_for_output
from aetv.gui.widgets import VideoView


class TrimSlider(QSlider):
    """Preview playhead plus draggable in/out pips on one timeline."""

    trimChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(0, 1000)
        self._in_value = 0
        self._out_value = 1000
        self._minimum_gap = 1
        self._dragging: str | None = None
        self.setMinimumHeight(34)
        self.setToolTip(
            "Drag the IN and OUT pips to trim; drag the round playhead to preview"
        )

    def set_trim(self, start: int, end: int, minimum_gap: int = 1) -> None:
        self._minimum_gap = max(1, int(minimum_gap))
        self._in_value = max(self.minimum(), min(int(start), self.maximum()))
        self._out_value = max(
            self._in_value + self._minimum_gap,
            min(int(end), self.maximum()),
        )
        if self._out_value > self.maximum():
            self._out_value = self.maximum()
            self._in_value = max(self.minimum(), self._out_value - self._minimum_gap)
        self.update()

    def trim(self) -> tuple[int, int]:
        return self._in_value, self._out_value

    def _option(self) -> QStyleOptionSlider:
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return option

    def _groove_y(self) -> int:
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            self._option(),
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        return groove.center().y()

    def _value_to_x(self, value: int) -> int:
        option = self._option()
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_min = groove.x()
        slider_max = groove.right() - handle.width() + 1
        position = QStyle.sliderPositionFromValue(
            self.minimum(), self.maximum(), int(value), max(1, slider_max - slider_min)
        )
        return slider_min + position + handle.width() // 2

    def _x_to_value(self, x: int) -> int:
        option = self._option()
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_min = groove.x() + handle.width() // 2
        span = max(1, groove.width() - handle.width())
        position = max(0, min(int(x) - slider_min, span))
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), position, span
        )

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        start_x = self._value_to_x(self._in_value)
        end_x = self._value_to_x(self._out_value)
        y = self._groove_y()
        painter.setPen(QColor("#48a9d6"))
        painter.setBrush(QColor("#48a9d6"))
        painter.drawRoundedRect(start_x, y - 2, max(2, end_x - start_x), 4, 2, 2)
        self._draw_pip(painter, start_x, y, "IN", QColor("#63d17f"))
        self._draw_pip(painter, end_x, y, "OUT", QColor("#ffb54a"))

    def _draw_pip(
        self, painter: QPainter, x: int, y: int, label: str, color: QColor
    ) -> None:
        painter.setPen(color)
        painter.setBrush(color)
        painter.drawLine(x, y - 8, x, y + 8)
        painter.drawPolygon(
            QPolygon([QPoint(x - 5, y - 10), QPoint(x + 5, y - 10), QPoint(x, y - 4)])
        )
        metrics = painter.fontMetrics()
        painter.drawText(x - metrics.horizontalAdvance(label) // 2, y + 19, label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().toPoint().x()
            distances = {
                "in": abs(x - self._value_to_x(self._in_value)),
                "out": abs(x - self._value_to_x(self._out_value)),
            }
            nearest = min(distances, key=distances.get)
            if distances[nearest] <= 12:
                self._dragging = nearest
                self._move_pip(x)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None:
            self._move_pip(event.position().toPoint().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging is not None:
            self._move_pip(event.position().toPoint().x())
            self._dragging = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _move_pip(self, x: int) -> None:
        value = self._x_to_value(x)
        if self._dragging == "in":
            self._in_value = min(value, self._out_value - self._minimum_gap)
        elif self._dragging == "out":
            self._out_value = max(value, self._in_value + self._minimum_gap)
        self.trimChanged.emit(self._in_value, self._out_value)
        self.update()


class ClipEditorDialog(QDialog):
    """Preview a file and choose its in/out points and output framing."""

    def __init__(
        self,
        path: str,
        mode,
        initial: ClipEdit | None = None,
        default_duration_s: float = 10.0,
        parent=None,
    ):
        super().__init__(parent)
        self.path = str(path)
        self.mode = mode
        self._capture = None
        self.duration_s, self.source_size = self._open_video()
        self.setWindowTitle(f"Edit clip — {Path(path).name}")
        self.resize(720, 560)

        self.preview = VideoView("Unable to preview clip")
        self.preview.setMinimumSize(480, 270)
        self.timeline = TrimSlider()
        self.timeline.valueChanged.connect(self._show_timeline_frame)
        self.timeline.trimChanged.connect(self._timeline_trim_changed)

        self.in_point = self._time_spin()
        self.out_point = self._time_spin()
        minimum_segment = min(1.0, self.duration_s)
        self.in_point.setRange(0.0, max(0.0, self.duration_s - minimum_segment))
        self.out_point.setRange(minimum_segment, self.duration_s)
        default_end = min(self.duration_s, max(1.0, float(default_duration_s)))
        if initial is not None:
            start = min(initial.start_s, max(0.0, self.duration_s - minimum_segment))
            end = min(self.duration_s, max(start + minimum_segment, initial.end_s))
            framing = initial.framing
        else:
            start, end, framing = 0.0, default_end, "crop"
        self.in_point.setValue(start)
        self.out_point.setValue(end)
        self.in_point.valueChanged.connect(self._validate_range)
        self.out_point.valueChanged.connect(self._validate_range)

        self.framing = QComboBox()
        self.framing.addItem("Crop to fill", "crop")
        self.framing.addItem("Fit with black bars", "fit")
        self.framing.addItem("Stretch to output", "stretch")
        self.framing.setCurrentIndex(max(0, self.framing.findData(framing)))
        self.framing.currentIndexChanged.connect(lambda _index: self._show_timeline_frame())

        self.duration_label = QLabel()
        self._validate_range()
        self._sync_timeline_trim()
        source_w, source_h = self.source_size
        info = QLabel(
            f"Source: {source_w}×{source_h} · Output: {mode.width}×{mode.height} at {mode.fps:g} fps"
        )
        info.setStyleSheet("color: palette(mid);")

        form = QFormLayout()
        form.addRow("In point", self.in_point)
        form.addRow("Out point", self.out_point)
        form.addRow("Resize", self.framing)
        form.addRow("Prepared length", self.duration_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.timeline)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._seek_to(start)

    def _time_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setSuffix(" s")
        return spin

    def _open_video(self) -> tuple[float, tuple[int, int]]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for the clip editor") from error
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open video: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("video duration or dimensions could not be read")
        self._capture = capture
        return frame_count / fps, (width, height)

    def _show_timeline_frame(self, _value: int | None = None) -> None:
        seconds = self.duration_s * self.timeline.value() / 1000.0
        self._seek_to(seconds)

    def _timeline_trim_changed(self, start: int, end: int) -> None:
        previous_start = self.in_point.value()
        start_s = self.duration_s * start / 1000.0
        end_s = self.duration_s * end / 1000.0
        self.in_point.blockSignals(True)
        self.out_point.blockSignals(True)
        self.in_point.setValue(start_s)
        self.out_point.setValue(end_s)
        self.in_point.blockSignals(False)
        self.out_point.blockSignals(False)
        self._validate_range()
        self._seek_to(start_s if abs(start_s - previous_start) > 0.005 else end_s)

    def _sync_timeline_trim(self) -> None:
        if self.duration_s <= 0:
            return
        start = round(1000.0 * self.in_point.value() / self.duration_s)
        end = round(1000.0 * self.out_point.value() / self.duration_s)
        minimum_gap = max(1, round(1000.0 * min(1.0, self.duration_s) / self.duration_s))
        self.timeline.set_trim(start, end, minimum_gap)

    def _seek_to(self, seconds: float) -> None:
        if self._capture is None:
            return
        import cv2

        self._capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
        ok, frame = self._capture.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = frame_for_output(
            np.ascontiguousarray(rgb),
            self.mode.width,
            self.mode.height,
            str(self.framing.currentData() or "crop"),
        )
        self.preview.set_rgb(output)

    def _validate_range(self, _value: float | None = None) -> None:
        start = self.in_point.value()
        end = self.out_point.value()
        minimum_segment = min(1.0, self.duration_s)
        if end - start < minimum_segment:
            sender = self.sender()
            if sender is self.in_point:
                self.out_point.setValue(min(self.duration_s, start + minimum_segment))
            else:
                self.in_point.setValue(max(0.0, end - minimum_segment))
            start, end = self.in_point.value(), self.out_point.value()
        whole_gops = max(1, int(end - start))
        self.duration_label.setText(
            f"{end - start:.2f} s selected · {whole_gops} whole 1-second GOP{'s' if whole_gops != 1 else ''}"
        )
        self._sync_timeline_trim()

    def edit(self) -> ClipEdit:
        start = self.in_point.value()
        # Radio frames are prepared in whole one-second GOPs. Keep the chosen
        # in point and clamp the out point to the last complete GOP.
        gops = max(1, int(self.out_point.value() - start))
        end = min(self.duration_s, start + gops)
        return ClipEdit(
            path=self.path,
            start_s=start,
            end_s=end,
            framing=str(self.framing.currentData() or "crop"),
        )

    def done(self, result: int) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        super().done(result)
