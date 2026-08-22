"""Transmit pane: webcam / file source, preview, send with CAT PTT."""

from __future__ import annotations

import math
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aetv.audio_io import AudioUnavailable, list_audio_devices
from aetv.config import AETV_MODES
from aetv.source import iter_webcam, list_cameras
from aetv.station import TxEngine, TxPhase, TxState
from aetv.gui.widgets import ElidingLabel, VideoView


class TransmitPanel(QWidget):
    transmitStarted = Signal()
    transmitFinished = Signal()
    logMessage = Signal(str)
    pttChanged = Signal(bool)
    _stateArrived = Signal(object)
    _errorArrived = Signal(str)
    _previewArrived = Signal(object)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self.engine = TxEngine(
            station,
            on_state=self._stateArrived.emit,
            on_error=self._errorArrived.emit,
            on_preview=self._previewArrived.emit,
        )
        self._thread: threading.Thread | None = None
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._stateArrived.connect(self._apply_state, Qt.ConnectionType.QueuedConnection)
        self._errorArrived.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._previewArrived.connect(self._show_preview, Qt.ConnectionType.QueuedConnection)
        self._file_path = ""
        self._build()
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._kick_preview)

    def control_strip(self) -> QWidget:
        return self._strip

    def picture_area(self) -> QWidget:
        return self.preview

    def transmitting(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def sync_from_config(self) -> None:
        settings = self.station.settings
        self.mode.setCurrentIndex(max(0, self.mode.findData(settings.mode)))
        self.gops.setValue(settings.gops)
        level = max(0.05, min(1.0, settings.tx_level))
        self.level_db.setValue(20.0 * math.log10(level))
        self._fill_cameras()
        self._fill_outputs()

    def send(self) -> None:
        if self.transmitting():
            return
        self._apply_panel_settings()
        problems = self.station.settings.validate()
        if self.file_radio.isChecked() and not self._file_path:
            problems.append("choose a video file to send")
        if self.station.codec is None:
            problems.append("checkpoint is still loading")
        if problems:
            self.status.setText(problems[0])
            return
        self._stop_preview()
        source = "webcam" if self.cam_radio.isChecked() else self._file_path
        self.send_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.transmitStarted.emit()
        self._thread = threading.Thread(target=self._run_send, args=(source,), daemon=True, name="aetv-tx")
        self._thread.start()

    def cancel(self) -> None:
        self.engine.cancel()
        self.status.setText("cancelling…")

    def _run_send(self, source: str) -> None:
        self.engine.transmit(source)

    def _build(self) -> None:
        self.preview = VideoView("Choose a webcam or a video file")
        self.cam_radio = QRadioButton("Webcam")
        self.file_radio = QRadioButton("Video file")
        self.cam_radio.setChecked(True)
        self.camera = QComboBox()
        refresh_cam = QPushButton("Refresh")
        refresh_cam.clicked.connect(self._fill_cameras)
        self.file_button = QPushButton("File…")
        self.file_button.clicked.connect(self._choose_file)
        self.file_label = ElidingLabel("No file selected")
        self.mode = QComboBox()
        for name, spec in AETV_MODES.items():
            self.mode.addItem(f"{name} — {spec.width}×{spec.height} @ {spec.fps:g} fps", name)
        self.gops = QSpinBox()
        self.gops.setRange(1, 300)
        self.gops.setSuffix(" s")
        self.level_db = QDoubleSpinBox()
        self.level_db.setRange(-24.0, 0.0)
        self.level_db.setSingleStep(0.5)
        self.level_db.setSuffix(" dB FS")
        self.output = QComboBox()
        self.send_button = QPushButton("Send")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.send_button.clicked.connect(self.send)
        self.cancel_button.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = ElidingLabel("Ready")

        self.cam_radio.toggled.connect(self._on_source_toggled)

        self._strip = QWidget()
        strip = QVBoxLayout(self._strip)
        strip.setContentsMargins(0, 0, 0, 0)
        src = QHBoxLayout()
        src.addWidget(self.cam_radio)
        src.addWidget(self.camera, 1)
        src.addWidget(refresh_cam)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_radio)
        file_row.addWidget(self.file_button)
        file_row.addWidget(self.file_label, 1)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addWidget(self.mode)
        mode_row.addWidget(QLabel("Length"))
        mode_row.addWidget(self.gops)
        mode_row.addWidget(QLabel("Level"))
        mode_row.addWidget(self.level_db)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("To radio"))
        out_row.addWidget(self.output, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        strip.addLayout(src)
        strip.addLayout(file_row)
        strip.addLayout(mode_row)
        strip.addLayout(out_row)
        strip.addWidget(self.progress)
        strip.addLayout(buttons)
        strip.addWidget(self.status)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self._strip, 0)
        self.sync_from_config()
        self._preview_timer.start()

    def _on_source_toggled(self, _on: bool) -> None:
        if self.cam_radio.isChecked() and not self.transmitting():
            self._preview_timer.start()
        else:
            self._stop_preview()

    def _kick_preview(self) -> None:
        if self.transmitting() or not self.cam_radio.isChecked():
            return
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        self._preview_thread = threading.Thread(target=self._preview_once, daemon=True, name="webcam-preview")
        self._preview_thread.start()

    def _preview_once(self) -> None:
        try:
            mode = AETV_MODES[self.station.settings.mode]
            camera = int(self.camera.currentData() or 0)
            frame = next(iter_webcam(mode, camera=camera, duration_s=1.0 / mode.fps))
            if not self._preview_stop.is_set():
                self._previewArrived.emit(frame)
        except Exception:
            pass

    def _stop_preview(self) -> None:
        self._preview_stop.set()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Video to send", "", "Video (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)"
        )
        if not path:
            return
        self._file_path = path
        self.file_label.setText(path)
        self.file_radio.setChecked(True)

    def _fill_cameras(self) -> None:
        current = self.station.settings.camera_index
        self.camera.clear()
        try:
            cameras = list_cameras()
        except Exception:
            cameras = []
        if not cameras:
            self.camera.addItem("Camera 0", 0)
        for item in cameras:
            self.camera.addItem(item["name"], item["index"])
        index = self.camera.findData(current)
        if index >= 0:
            self.camera.setCurrentIndex(index)

    def _fill_outputs(self) -> None:
        current = self.station.settings.audio_output
        self.output.clear()
        self.output.addItem("System default", "")
        try:
            for item in list_audio_devices("output"):
                self.output.addItem(item.label(), item.name)
        except AudioUnavailable:
            return
        index = self.output.findData(current)
        if index >= 0:
            self.output.setCurrentIndex(index)

    def _apply_panel_settings(self) -> None:
        settings = self.station.settings
        settings.mode = self.mode.currentData()
        settings.gops = int(self.gops.value())
        settings.tx_level = float(10 ** (self.level_db.value() / 20.0))
        settings.camera_index = int(self.camera.currentData() or 0)
        settings.audio_output = self.output.currentData() or ""

    def _apply_state(self, state: TxState) -> None:
        self.status.setText(state.message or state.phase.value)
        self.progress.setValue(int(round(state.progress * 100)))
        keyed = state.phase in {TxPhase.KEYING, TxPhase.SENDING, TxPhase.UNKEYING}
        self.pttChanged.emit(keyed)
        if state.phase in {TxPhase.DONE, TxPhase.CANCELLED, TxPhase.FAILED, TxPhase.IDLE}:
            self.send_button.setEnabled(self.station.codec is not None)
            self.cancel_button.setEnabled(False)
            if state.phase != TxPhase.IDLE:
                self.transmitFinished.emit()
            if self.cam_radio.isChecked():
                self._preview_timer.start()

    def _on_error(self, message: str) -> None:
        self.status.setText(message)
        self.logMessage.emit(message)

    def _show_preview(self, frames) -> None:
        self.preview.set_rgb(frames)
