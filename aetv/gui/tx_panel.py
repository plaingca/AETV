"""Transmit pane: webcam / file source, preview, send with CAT PTT."""

from __future__ import annotations

import math
import threading

from PySide6.QtCore import Qt, Signal
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
    _previewStopped = Signal()
    _workerFinished = Signal()
    _camerasArrived = Signal(object)
    _outputsArrived = Signal(object)

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
        self._start_gate = threading.Event()
        self._cancel_requested = threading.Event()
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._preview_restart_pending = False
        self._camera_load_active = False
        self._output_load_active = False
        self._stateArrived.connect(self._apply_state, Qt.ConnectionType.QueuedConnection)
        self._errorArrived.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._previewArrived.connect(self._show_preview, Qt.ConnectionType.QueuedConnection)
        self._previewStopped.connect(self._on_preview_stopped, Qt.ConnectionType.QueuedConnection)
        self._workerFinished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self._camerasArrived.connect(self._apply_cameras, Qt.ConnectionType.QueuedConnection)
        self._outputsArrived.connect(self._apply_outputs, Qt.ConnectionType.QueuedConnection)
        self._file_path = ""
        self._build()

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
        # Some camera backends take seconds to return from read().  Ask the
        # preview to stop here, but wait for it only in the TX worker.
        self._preview_stop.set()
        source = "webcam" if self.cam_radio.isChecked() else self._file_path
        self.send_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("preparing transmit…")
        self._start_gate.clear()
        self._cancel_requested.clear()
        self.transmitStarted.emit()
        self._thread = threading.Thread(target=self._run_send, args=(source,), daemon=True, name="aetv-tx")
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._start_gate.set()
        self.engine.cancel()
        self.status.setText("cancelling…")

    def _run_send(self, source: str) -> None:
        try:
            preview = self._preview_thread
            if preview is not None and preview.is_alive():
                preview.join(timeout=3.0)
            self._start_gate.wait()
            if self._cancel_requested.is_set():
                self._stateArrived.emit(
                    TxState(TxPhase.CANCELLED, 0.0, "cancelled")
                )
                return
            self.engine.transmit(source)
        finally:
            # Terminal state signals can reach the event loop before this
            # thread returns.  Clear the worker first, then let the UI restart
            # the camera knowing transmitting() is definitively false.
            self._thread = None
            self._workerFinished.emit()

    def _on_worker_finished(self) -> None:
        if self.cam_radio.isChecked():
            self._start_preview()

    def allow_transmit(self) -> None:
        """Release the TX worker after half-duplex receive has shut down."""
        self._start_gate.set()

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
        self.camera.currentIndexChanged.connect(self._restart_preview)
        self.mode.currentIndexChanged.connect(self._restart_preview)
        self._start_preview()

    def _on_source_toggled(self, _on: bool) -> None:
        if self.cam_radio.isChecked() and not self.transmitting():
            self._start_preview()
        else:
            self._stop_preview()

    def _restart_preview(self, _index: int) -> None:
        if not hasattr(self, "cam_radio"):
            return
        self._preview_restart_pending = True
        self._preview_stop.set()
        if self._preview_thread is None or not self._preview_thread.is_alive():
            self._on_preview_stopped()

    def _start_preview(self) -> None:
        if self.transmitting() or not self.cam_radio.isChecked():
            return
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        mode_name = self.mode.currentData() or self.station.settings.mode
        camera = int(self.camera.currentData() or 0)
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            args=(mode_name, camera),
            daemon=True,
            name="webcam-preview",
        )
        self._preview_thread.start()

    def _preview_loop(self, mode_name: str, camera: int) -> None:
        try:
            for frame in iter_webcam(AETV_MODES[mode_name], camera=camera):
                if self._preview_stop.is_set():
                    break
                self._previewArrived.emit(frame)
        except Exception as error:
            if not self._preview_stop.is_set():
                self._errorArrived.emit(f"Webcam preview unavailable: {error}")
        finally:
            self._previewStopped.emit()

    def _on_preview_stopped(self) -> None:
        if not self._preview_restart_pending:
            return
        self._preview_restart_pending = False
        self._start_preview()

    def _stop_preview(self) -> None:
        self._preview_stop.set()
        thread = self._preview_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def stop_preview(self) -> None:
        """Release the preview camera before the application closes."""
        self._stop_preview()

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
        if self._camera_load_active:
            return
        self._camera_load_active = True
        self.camera.clear()
        self.camera.addItem("Loading cameras…", self.station.settings.camera_index)
        threading.Thread(
            target=self._load_cameras, daemon=True, name="aetv-camera-list"
        ).start()

    def _load_cameras(self) -> None:
        try:
            cameras = list_cameras()
        except Exception:
            cameras = []
        self._camerasArrived.emit(cameras)

    def _apply_cameras(self, cameras) -> None:
        self._camera_load_active = False
        current = self.station.settings.camera_index
        self.camera.blockSignals(True)
        self.camera.clear()
        if not cameras:
            self.camera.addItem("Camera 0", 0)
        for item in cameras:
            self.camera.addItem(item["name"], item["index"])
        index = self.camera.findData(current)
        if index >= 0:
            self.camera.setCurrentIndex(index)
        self.camera.blockSignals(False)

    def _fill_outputs(self) -> None:
        if self._output_load_active:
            return
        self._output_load_active = True
        self.output.clear()
        self.output.addItem("Loading devices…", self.station.settings.audio_output)
        threading.Thread(
            target=self._load_outputs, daemon=True, name="aetv-audio-list"
        ).start()

    def _load_outputs(self) -> None:
        try:
            outputs = list_audio_devices("output")
        except AudioUnavailable:
            outputs = []
        self._outputsArrived.emit(outputs)

    def _apply_outputs(self, outputs) -> None:
        self._output_load_active = False
        current = self.station.settings.audio_output
        self.output.clear()
        self.output.addItem("System default", "")
        for item in outputs:
            self.output.addItem(item.label(), item.name)
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

    def _on_error(self, message: str) -> None:
        self.status.setText(message)
        self.logMessage.emit(message)

    def _show_preview(self, frames) -> None:
        self.preview.set_rgb(frames)
