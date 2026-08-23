"""Scrolling spectrum of the receive ring buffer.

The backing image is kept at the widget's device-pixel size so the
painter never rescales it. Downscaling a tall history into a short
strip makes rows crawl; one pixel per tick does not.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from aetv.config import AETV_MODES

MIN_DBFS = -140.0
MAX_DBFS = 3.0
MIN_DISPLAY_RANGE_DB = 32.0


def _colormap() -> np.ndarray:
    stops = (
        (0.00, (0, 0, 0)),
        (0.25, (0, 0, 140)),
        (0.50, (0, 170, 90)),
        (0.75, (245, 235, 40)),
        (1.00, (255, 255, 255)),
    )
    table = np.zeros((256, 3), dtype=np.uint8)
    for index in range(256):
        x = index / 255.0
        k = 0
        while k + 2 < len(stops) and x > stops[k + 1][0]:
            k += 1
        lo, hi = stops[k], stops[k + 1]
        t = (x - lo[0]) / (hi[0] - lo[0])
        table[index] = np.rint(np.array(lo[1]) + t * (np.array(hi[1]) - np.array(lo[1]))).astype(np.uint8)
    return table


COLORMAP = _colormap()


def reduce_to_width(row: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros(0, dtype=np.float32)
    if row.size == width:
        return row.astype(np.float32, copy=False)
    if row.size > width:
        # Peak-hold so a one-bin carrier is not sampled away.
        edges = np.linspace(0, row.size, width + 1).astype(int)
        return np.array([row[edges[i] : edges[i + 1]].max() if edges[i + 1] > edges[i] else 0.0 for i in range(width)], dtype=np.float32)
    x = np.linspace(0, row.size - 1, width)
    return np.interp(x, np.arange(row.size), row).astype(np.float32)


def spectrum_dbfs(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a Hann-windowed spectrum whose tone amplitudes are in dBFS.

    Dividing by the window's coherent gain keeps the FFT size from changing
    the displayed level. A full-scale sinusoid therefore lands near 0 dBFS
    instead of gaining roughly 54 dB merely because a 1024-point FFT is used.
    """
    values = np.asarray(samples, dtype=np.float32)
    window = np.hanning(values.size)
    coherent_gain = max(float(window.sum()) / 2.0, 1e-12)
    amplitudes = np.abs(np.fft.rfft(values * window)) / coherent_gain
    dbfs = 20.0 * np.log10(np.maximum(amplitudes, 1e-12))
    return dbfs.astype(np.float32), amplitudes.astype(np.float32)


def automatic_levels(values: np.ndarray) -> tuple[float, float]:
    """Choose a robust noise floor and highlight level for a spectrum row."""
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -100.0, -60.0
    floor = float(np.percentile(finite, 35.0))
    highlight = float(np.percentile(finite, 99.5))
    floor = float(np.clip(floor, MIN_DBFS, MAX_DBFS - MIN_DISPLAY_RANGE_DB))
    ceiling = float(np.clip(max(highlight, floor + MIN_DISPLAY_RANGE_DB), floor + MIN_DISPLAY_RANGE_DB, MAX_DBFS))
    return floor, ceiling


class Waterfall(QWidget):
    def __init__(self, parent=None, fps: int = 20):
        super().__init__(parent)
        self._ring = None
        self._fs = 24000
        self._band_lo = 800.0
        self._band_hi = 9200.0
        self._image = QImage()
        self._peak = 0.0
        self._clipping = False
        self._clip_latched = False
        self._display_floor: float | None = None
        self._display_ceiling: float | None = None
        self.setMinimumHeight(90)
        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        timer = QTimer(self)
        timer.timeout.connect(self.tick)
        timer.start(max(1, 1000 // max(1, fps)))

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        return QSize(640, 160)

    def set_ring(self, ring) -> None:
        self._ring = ring
        self._reset_scaling()
        if ring is not None:
            self._fs = ring.fs

    def set_mode(self, mode_name: str) -> None:
        mode = AETV_MODES[mode_name]
        self._fs = mode.geometry.fs
        self._band_lo, self._band_hi = mode.geometry.tx_bandpass
        self._reset_scaling()

    def _reset_scaling(self) -> None:
        self._display_floor = None
        self._display_ceiling = None

    def clear(self) -> None:
        if not self._image.isNull():
            self._image.fill(Qt.GlobalColor.black)
        self._peak = 0.0
        self._clipping = False
        self._reset_scaling()
        self.update()

    def clip_latched(self) -> bool:
        return self._clip_latched

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._clip_latched = False
            self.update()

    def resizeEvent(self, event) -> None:
        self._ensure_image()
        super().resizeEvent(event)

    def _ensure_image(self) -> None:
        ratio = max(1.0, float(self.devicePixelRatio()))
        width = max(1, int(round(self.width() * ratio)))
        height = max(1, int(round(self.height() * ratio)))
        if self._image.width() != width or self._image.height() != height:
            self._image = QImage(width, height, QImage.Format.Format_RGB32)
            self._image.fill(Qt.GlobalColor.black)
            self._image.setDevicePixelRatio(ratio)

    def tick(self) -> None:
        self._ensure_image()
        ring = self._ring
        if ring is None or self._image.isNull():
            self.update()
            return
        nfft = 1024
        samples = ring.tail(nfft)
        if samples.size < 32:
            self.update()
            return
        mag, _ = spectrum_dbfs(samples)
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self._fs)
        usable = mag[freqs <= self._fs / 2]
        width = self._image.width()
        meter_w = max(8, int(round(12 * self.devicePixelRatio())))
        row = reduce_to_width(usable.astype(np.float32), max(1, width - meter_w))
        # Scale from the useful modem passband. The robust percentiles keep
        # Flex AGC noise dark while allowing carriers to reach the highlights.
        in_band = mag[(freqs >= self._band_lo) & (freqs <= self._band_hi)]
        target_floor, target_ceiling = automatic_levels(in_band if in_band.size else usable)
        if self._display_floor is None or self._display_ceiling is None:
            self._display_floor, self._display_ceiling = target_floor, target_ceiling
        else:
            # Gentle tracking prevents brightness pumping without leaving an
            # AGC level change stuck as a white screen.
            alpha = 0.18
            self._display_floor += alpha * (target_floor - self._display_floor)
            self._display_ceiling += alpha * (target_ceiling - self._display_ceiling)
        span = max(self._display_ceiling - self._display_floor, MIN_DISPLAY_RANGE_DB)
        norm = np.clip((row - self._display_floor) / span, 0.0, 1.0)
        norm = np.power(norm, 1.35)  # reserve yellow/white for actual signals
        pixel_freqs = np.linspace(0.0, self._fs / 2.0, row.size)
        outside = (pixel_freqs < self._band_lo) | (pixel_freqs > self._band_hi)
        norm[outside] *= 0.45
        colors = COLORMAP[np.rint(norm * 255).astype(np.uint8)]
        shifted = QImage(self._image.size(), QImage.Format.Format_RGB32)
        shifted.setDevicePixelRatio(self._image.devicePixelRatio())
        shifted.fill(Qt.GlobalColor.black)
        painter = QPainter(shifted)
        painter.drawImage(0, 1, self._image)
        painter.end()
        self._image = shifted
        for x, rgb in enumerate(colors):
            self._image.setPixel(x, 0, QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])).rgb())
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        self._peak = 0.85 * self._peak + 0.15 * peak
        self._clipping = peak >= 0.99
        if self._clipping:
            self._clip_latched = True
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self._image.isNull():
            painter.drawImage(0, 0, self._image)
        self._draw_band_markers(painter)
        self._draw_meter(painter)
        if self._ring is None:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 140))
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Start receiving to see the waterfall")

    def _draw_band_markers(self, painter: QPainter) -> None:
        width = self.width()
        meter_w = 12
        span = self._fs / 2
        if span <= 0:
            return
        for hz, label in ((self._band_lo, f"{self._band_lo:.0f}"), (self._band_hi, f"{self._band_hi:.0f}")):
            x = int((hz / span) * (width - meter_w))
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine))
            painter.drawLine(x, 0, x, self.height())
            painter.setPen(QColor(220, 220, 220, 180))
            painter.drawText(x + 3, 12, f"{label} Hz")

    def _draw_meter(self, painter: QPainter) -> None:
        w = 10
        rect_x = self.width() - w - 2
        h = self.height() - 4
        y0 = 2
        painter.fillRect(rect_x, y0, w, h, QColor("#111111"))
        level = min(1.0, self._peak)
        fill = int(h * level)
        if self._clip_latched:
            color = QColor(255, 60, 60)
        elif level > 0.85:
            color = QColor(255, 190, 60)
        else:
            color = QColor(90, 220, 120)
        painter.fillRect(rect_x, y0 + h - fill, w, fill, color)
        if self._clip_latched:
            painter.setPen(QColor(255, 60, 60))
            painter.drawText(rect_x - 28, 12, "CLIP")
