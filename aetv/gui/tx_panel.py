"""Transmit pane: webcam, desktop, and prepared clip sources."""

from __future__ import annotations

import math
import threading
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QRubberBand,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aetv.audio_io import AudioUnavailable, list_audio_devices
from aetv.config import AETV_MODES, RELEASE_MODES, RELEASE_MODE_LABELS
from aetv.hfchannel import CHANNEL_PROFILES
from aetv.source import (
    CameraFrameBuffer,
    ClipEdit,
    PreparedClip,
    ScreenCaptureSpec,
    iter_screen_capture,
    list_cameras,
    list_windows,
)
from aetv.station import TxEngine, TxPhase, TxState
from aetv.gui.clip_editor import ClipEditorDialog
from aetv.gui.clip_grid import ClipGrid
from aetv.gui.widgets import ElidingLabel, VideoView


class _RegionSelector(QDialog):
    """A translucent virtual-desktop overlay used to drag a capture region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.origin = QPoint()
        self.selection: QRect | None = None
        self.band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(QGuiApplication.primaryScreen().virtualGeometry())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 105))
        painter.setPen(QColor("white"))
        painter.drawText(
            self.rect().adjusted(20, 20, -20, -20),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "Drag around the area to capture · Esc cancels",
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            self.band.setGeometry(QRect(self.origin, self.origin))
            self.band.show()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.band.setGeometry(QRect(self.origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        local = self.band.geometry().normalized()
        if local.width() < 2 or local.height() < 2:
            self.reject()
            return
        top_left = self.mapToGlobal(local.topLeft())
        self.selection = QRect(top_left, local.size())
        self.accept()


class TransmitPanel(QWidget):
    transmitStarted = Signal()
    transmitFinished = Signal()
    modeRequested = Signal(str)
    logMessage = Signal(str)
    pttChanged = Signal(bool)
    _stateArrived = Signal(object)
    _errorArrived = Signal(str)
    _previewArrived = Signal(object)
    _cameraPreviewArrived = Signal(object)
    _previewStopped = Signal()
    _workerFinished = Signal()
    _camerasArrived = Signal(object)
    _outputsArrived = Signal(object)
    _clipProgress = Signal(int, float, int)
    _clipReady = Signal(int, object, int)
    _clipFailed = Signal(int, str, int)
    _inputsArrived = Signal(object)
    loopbackVideo = Signal(object, object)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self._camera_frames = CameraFrameBuffer()
        self._live_source_lock = threading.Lock()
        self._active_live_source: str | ScreenCaptureSpec = "webcam"
        self.engine = TxEngine(
            station,
            on_state=self._stateArrived.emit,
            on_error=self._errorArrived.emit,
            on_preview=self._previewArrived.emit,
            camera_frames=self._camera_frames.frames,
            on_loopback=lambda video, state: self.loopbackVideo.emit(video, state),
            live_source=self._live_source_for_tx,
        )
        self._thread: threading.Thread | None = None
        self._start_gate = threading.Event()
        self._cancel_requested = threading.Event()
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._preview_restart_pending = False
        self._webcam_tx_active = False
        self._camera_preview_queued = threading.Event()
        self._camera_load_active = False
        self._output_load_active = False
        self._prepared_clips: dict[int, PreparedClip] = {}
        self._clip_edits: dict[int, ClipEdit] = {}
        self._selected_clip: int | None = None
        self._clip_generation = 0
        self._stateArrived.connect(self._apply_state, Qt.ConnectionType.QueuedConnection)
        self._errorArrived.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._previewArrived.connect(self._show_preview, Qt.ConnectionType.QueuedConnection)
        self._cameraPreviewArrived.connect(
            self._show_camera_preview, Qt.ConnectionType.QueuedConnection
        )
        self._previewStopped.connect(self._on_preview_stopped, Qt.ConnectionType.QueuedConnection)
        self._workerFinished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self._camerasArrived.connect(self._apply_cameras, Qt.ConnectionType.QueuedConnection)
        self._outputsArrived.connect(self._apply_outputs, Qt.ConnectionType.QueuedConnection)
        self._clipProgress.connect(self._apply_clip_progress, Qt.ConnectionType.QueuedConnection)
        self._clipReady.connect(self._apply_clip_ready, Qt.ConnectionType.QueuedConnection)
        self._clipFailed.connect(self._apply_clip_failed, Qt.ConnectionType.QueuedConnection)
        self._inputsArrived.connect(self._apply_inputs, Qt.ConnectionType.QueuedConnection)
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
        previous = self.mode.blockSignals(True)
        selected = "V8_AV" if settings.waveform_mode == "analog_av" else settings.mode
        self.mode.setCurrentIndex(max(0, self.mode.findData(selected)))
        self.mode.blockSignals(previous)
        self.gops.setValue(settings.gops)
        level = max(0.05, min(1.0, settings.tx_level))
        self.level_db.setValue(20.0 * math.log10(level))
        profile = settings.tx_channel_profile
        if profile in self.channel_keys:
            button = self.channel_buttons.button(self.channel_keys.index(profile))
            if button is not None:
                button.setChecked(True)
        self._sync_channel_route()
        self.av_power.setValue(int(round(settings.av_video_power * 100)))
        self.mic_mix.setValue(int(round(settings.av_microphone_mix * 100)))
        self._sync_av_controls()
        self._fill_cameras()
        self._fill_outputs()

    def send(self) -> None:
        if self.transmitting():
            return
        self._apply_panel_settings()
        problems = self.station.settings.validate(
            radio_tx=not self.emulating(),
            receive=False,
        )
        if self.file_radio.isChecked() and not self._file_path:
            problems.append("choose a video file to send")
        codec = self.station.codec
        if codec is None:
            problems.append("checkpoint is still loading")
        elif codec.mode.name != self.station.settings.mode:
            problems.append(f"{self.station.settings.mode} checkpoint is still loading")
        if problems:
            self.status.setText(problems[0])
            return
        if self.cam_radio.isChecked():
            source = "webcam"
        elif self.screen_radio.isChecked():
            source = self.screen_target.currentData()
            if not isinstance(source, ScreenCaptureSpec):
                self.status.setText("choose a screen capture target")
                return
        elif self._selected_clip is not None and self._selected_clip in self._prepared_clips:
            source = self._prepared_clips[self._selected_clip]
        else:
            source = self._file_path
        if source == "webcam" or isinstance(source, ScreenCaptureSpec):
            with self._live_source_lock:
                self._active_live_source = source
        if source == "webcam" or isinstance(source, ScreenCaptureSpec):
            # Hand the camera from the Qt preview subscriber to TX before the
            # encoder/CUDA worker starts.  Leaving queued QImage paints active
            # during this transition has caused native Qt/Media Foundation
            # heap corruption on Windows, which bypasses Python exceptions.
            self._webcam_tx_active = True
            self._preview_restart_pending = False
            if not self._stop_preview_blocking():
                self._webcam_tx_active = False
                self.status.setText("Live preview did not stop; transmit was not started")
                return
            self._camera_preview_queued.clear()
        self.send_button.setEnabled(False)
        self.mode.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.mode.setEnabled(False)
        self.microphone.setEnabled(False)
        self.output.setEnabled(False)
        for button in self.channel_buttons.buttons():
            button.setEnabled(False)
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

    def _run_send(self, source: str | ScreenCaptureSpec | PreparedClip) -> None:
        try:
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
        self._webcam_tx_active = False
        if self.cam_radio.isChecked() or self.screen_radio.isChecked():
            self._start_preview()

    def allow_transmit(self) -> None:
        """Release the TX worker after half-duplex receive has shut down."""
        self._start_gate.set()

    def _build(self) -> None:
        self.preview = VideoView("Choose a webcam or a video file")
        self.cam_radio = QRadioButton("Webcam")
        self.screen_radio = QRadioButton("Screen")
        self.file_radio = QRadioButton("Video file")
        self.cam_radio.setChecked(True)
        self.camera = QComboBox()
        refresh_cam = QPushButton("Refresh")
        refresh_cam.clicked.connect(self._fill_cameras)
        self.file_button = QPushButton("File…")
        self.file_button.clicked.connect(self._choose_file)
        self.file_label = ElidingLabel("No file selected")
        self.screen_target = QComboBox()
        refresh_screen = QPushButton("Refresh")
        refresh_screen.clicked.connect(self._fill_screen_targets)
        region_button = QPushButton("Region…")
        region_button.clicked.connect(self._choose_region)
        self.mode = QComboBox()
        for name in RELEASE_MODES:
            self.mode.addItem(RELEASE_MODE_LABELS[name], name)
        v8 = AETV_MODES["V8"]
        self.mode.addItem(
            f"V8 A/V — {v8.width}×{v8.height} @ {v8.fps:g} fps + analog audio",
            "V8_AV",
        )
        self.gops = QSpinBox()
        self.gops.setRange(1, 300)
        self.gops.setSuffix(" s")
        self.level_db = QDoubleSpinBox()
        self.level_db.setRange(-24.0, 0.0)
        self.level_db.setSingleStep(0.5)
        self.level_db.setSuffix(" dB FS")
        self.output = QComboBox()
        self.microphone = QComboBox()
        self.av_power = QSlider(Qt.Orientation.Horizontal)
        self.av_power.setRange(0, 100)
        self.av_power_label = QLabel()
        self.mic_mix = QSlider(Qt.Orientation.Horizontal)
        self.mic_mix.setRange(0, 100)
        self.mic_mix_label = QLabel()
        self.send_button = QPushButton("Send")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.send_button.clicked.connect(self.send)
        self.cancel_button.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = ElidingLabel("Ready")

        self.cam_radio.toggled.connect(self._on_source_toggled)
        self.screen_radio.toggled.connect(self._on_source_toggled)
        self.file_radio.toggled.connect(self._on_source_toggled)

        self._strip = QWidget()
        strip = QVBoxLayout(self._strip)
        strip.setContentsMargins(0, 0, 0, 0)
        src = QHBoxLayout()
        src.addWidget(self.cam_radio)
        src.addWidget(self.camera, 1)
        src.addWidget(refresh_cam)
        screen_row = QHBoxLayout()
        screen_row.addWidget(self.screen_radio)
        screen_row.addWidget(self.screen_target, 1)
        screen_row.addWidget(region_button)
        screen_row.addWidget(refresh_screen)
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
        self.av_row = QHBoxLayout()
        self.av_row.addWidget(QLabel("Microphone"))
        self.av_row.addWidget(self.microphone, 1)
        self.av_row.addWidget(QLabel("Audio / video power"))
        self.av_row.addWidget(self.av_power, 1)
        self.av_row.addWidget(self.av_power_label)
        self.mix_row = QHBoxLayout()
        self.mix_row.addWidget(QLabel("Clip audio"))
        self.mix_row.addWidget(self.mic_mix, 1)
        self.mix_row.addWidget(QLabel("Microphone"))
        self.mix_row.addWidget(self.mic_mix_label)
        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Route"))
        self.channel_keys = ["radio", *CHANNEL_PROFILES]
        self.channel_buttons = QButtonGroup(self)
        self.channel_buttons.setExclusive(True)
        for index, key in enumerate(self.channel_keys):
            label = "Radio" if key == "radio" else CHANNEL_PROFILES[key].label
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(
                "Transmit to the configured radio/audio output"
                if key == "radio"
                else CHANNEL_PROFILES[key].description + " (local loopback; no PTT)"
            )
            self.channel_buttons.addButton(button, index)
            channel_row.addWidget(button)
        self.channel_buttons.button(0).setChecked(True)
        self.channel_buttons.idClicked.connect(self._sync_channel_route)
        buttons = QHBoxLayout()
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        strip.addLayout(src)
        strip.addLayout(screen_row)
        strip.addLayout(file_row)
        strip.addLayout(mode_row)
        strip.addLayout(out_row)
        strip.addLayout(self.av_row)
        strip.addLayout(self.mix_row)
        strip.addLayout(channel_row)
        strip.addWidget(self.progress)
        strip.addLayout(buttons)
        strip.addWidget(self.status)

        self.clip_grid = ClipGrid()
        self.clip_grid.fileChosen.connect(self._set_clip)
        self.clip_grid.activated.connect(self._activate_clip)
        self.clip_grid.hovered.connect(self._preview_clip)
        self.clip_grid.editRequested.connect(self._edit_clip)
        self.clip_grid.removeRequested.connect(self._remove_clip)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addWidget(QLabel("Prepared clips — drop files; hover to preview; click to transmit"))
        layout.addWidget(self.clip_grid, 0)
        layout.addWidget(self._strip, 0)
        self._fill_screen_targets()
        stored_bank = self.station.settings.clip_bank
        if stored_bank:
            for index, value in enumerate(stored_bank[: len(self.clip_grid.cells)]):
                if not value or not value.get("path"):
                    continue
                edit = ClipEdit.from_dict(value)
                self._clip_edits[index] = edit
                self._show_clip_edit(index, edit)
        else:
            for index, path in enumerate(
                (self.station.settings.clip_paths or [])[: len(self.clip_grid.cells)]
            ):
                if path:
                    edit = ClipEdit(str(path), 0.0, float(self.gops.value()), "crop")
                    self._clip_edits[index] = edit
                    self._show_clip_edit(index, edit)
        self.sync_from_config()
        self.camera.currentIndexChanged.connect(self._restart_preview)
        self.screen_target.currentIndexChanged.connect(self._on_screen_target_changed)
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        self.av_power.valueChanged.connect(self._on_av_power_changed)
        self.mic_mix.valueChanged.connect(self._on_mic_mix_changed)
        self._start_preview()

    def _on_source_toggled(self, _on: bool) -> None:
        if _on:
            self._update_active_live_source()
        if (
            (self.cam_radio.isChecked() or self.screen_radio.isChecked())
            and not self.transmitting()
            and not self._webcam_tx_active
        ):
            self._restart_preview(0)
        else:
            self._stop_preview()

    def _update_active_live_source(self) -> None:
        selected: str | ScreenCaptureSpec | None = None
        if self.cam_radio.isChecked():
            selected = "webcam"
        elif self.screen_radio.isChecked():
            candidate = self.screen_target.currentData()
            if isinstance(candidate, ScreenCaptureSpec):
                selected = candidate
        if selected is not None:
            with self._live_source_lock:
                self._active_live_source = selected

    def _live_source_for_tx(self) -> str | ScreenCaptureSpec:
        with self._live_source_lock:
            return self._active_live_source

    def _on_screen_target_changed(self, index: int) -> None:
        if self.screen_radio.isChecked():
            self._update_active_live_source()
        self._restart_preview(index)

    def _restart_preview(self, _index: int) -> None:
        if not hasattr(self, "cam_radio"):
            return
        self._preview_restart_pending = True
        self._preview_stop.set()
        if self._preview_thread is None or not self._preview_thread.is_alive():
            self._on_preview_stopped()

    def _start_preview(self) -> None:
        live_source = self.cam_radio.isChecked() or self.screen_radio.isChecked()
        if self.transmitting() or self._webcam_tx_active or not live_source:
            return
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        mode_name = self._selected_mode_name()
        camera = int(self.camera.currentData() or 0)
        screen = self.screen_target.currentData() if self.screen_radio.isChecked() else None
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            args=(mode_name, camera, screen),
            daemon=True,
            name="webcam-preview",
        )
        self._preview_thread.start()

    def _preview_loop(
        self, mode_name: str, camera: int, screen: ScreenCaptureSpec | None
    ) -> None:
        try:
            if screen is not None:
                frames = iter_screen_capture(AETV_MODES[mode_name], screen)
            else:
                frames = self._camera_frames.frames(
                    AETV_MODES[mode_name],
                    camera=camera,
                    should_stop=self._preview_stop.is_set,
                    latest=True,
                )
            for frame in frames:
                if self._preview_stop.is_set():
                    break
                # At most one camera paint may wait in Qt's event queue. The
                # next delivery always comes from the newest buffered frame.
                if not self._camera_preview_queued.is_set():
                    self._camera_preview_queued.set()
                    self._cameraPreviewArrived.emit(frame)
        except Exception as error:
            if not self._preview_stop.is_set():
                label = "Screen" if screen is not None else "Webcam"
                self._errorArrived.emit(f"{label} preview unavailable: {error}")
        finally:
            self._previewStopped.emit()

    def _on_preview_stopped(self) -> None:
        if not self._preview_restart_pending:
            return
        self._preview_restart_pending = False
        self._start_preview()

    def _stop_preview(self) -> None:
        self._preview_stop.set()

    def _stop_preview_blocking(self, timeout: float = 1.0) -> bool:
        """Stop only the preview consumer, leaving its camera producer live."""
        self._stop_preview()
        thread = self._preview_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return thread is None or not thread.is_alive()

    def stop_preview(self) -> None:
        """Release the preview camera before the application closes."""
        self._stop_preview_blocking(timeout=2.0)
        self._camera_frames.close()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Video to send", "", "Video (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)"
        )
        if not path:
            return
        self._file_path = path
        self._selected_clip = None
        self.file_label.setText(path)
        self.file_radio.setChecked(True)

    def _set_clip(self, index: int, path: str) -> None:
        if not path:
            return
        self._open_clip_editor(index, path, None)

    def _edit_clip(self, index: int) -> None:
        edit = self._clip_edits.get(index)
        if edit is not None:
            self._open_clip_editor(index, edit.path, edit)

    def _open_clip_editor(
        self, index: int, path: str, initial: ClipEdit | None
    ) -> None:
        mode_name = self._selected_mode_name()
        try:
            dialog = ClipEditorDialog(
                path,
                AETV_MODES[mode_name],
                initial=initial,
                default_duration_s=float(self.gops.value()),
                parent=self,
            )
        except Exception as error:
            self.status.setText(f"Clip editor could not open the video: {error}")
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        edit = dialog.edit()
        self._clip_edits[index] = edit
        self._show_clip_edit(index, edit)
        self._prepared_clips.pop(index, None)
        self._selected_clip = None
        self._save_clip_paths()
        self._queue_clip_preparation()

    def _show_clip_edit(self, index: int, edit: ClipEdit) -> None:
        cell = self.clip_grid.cells[index]
        cell.set_path(edit.path)
        seconds = max(1, int(edit.duration_s))
        cell.name.setText(f"{Path(edit.path).stem} · {seconds}s")

    def _activate_clip(self, index: int) -> None:
        cell = self.clip_grid.cells[index]
        if not cell.path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Choose clip {index + 1}",
                "",
                "Video (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)",
            )
            if path:
                self._set_clip(index, path)
            return
        prepared = self._prepared_clips.get(index)
        if prepared is None:
            self.status.setText(f"{cell.name.text()} is still being prepared")
            return
        self._selected_clip = index
        self._file_path = prepared.path
        self.file_label.setText(f"Prepared: {prepared.path}")
        self.file_radio.setChecked(True)
        self.send()

    def _preview_clip(self, index: int) -> None:
        prepared = self._prepared_clips.get(index)
        if prepared is not None and not self.transmitting():
            self.preview.set_rgb(prepared.preview_frames, fps=2.0)

    def _remove_clip(self, index: int) -> None:
        self.clip_grid.cells[index].clear()
        self._prepared_clips.pop(index, None)
        self._clip_edits.pop(index, None)
        if self._selected_clip == index:
            self._selected_clip = None
        self._save_clip_paths()

    def _save_clip_paths(self) -> None:
        self.station.settings.clip_paths = [cell.path for cell in self.clip_grid.cells]
        self.station.settings.clip_bank = [
            self._clip_edits[index].to_dict() if index in self._clip_edits else {}
            for index in range(len(self.clip_grid.cells))
        ]

    def model_ready(self) -> None:
        """Refresh prepared clips after the requested codec becomes available."""
        self._queue_clip_preparation()

    def _queue_clip_preparation(self) -> None:
        self._clip_generation += 1
        generation = self._clip_generation
        self._selected_clip = None
        self._prepared_clips.clear()
        codec = self.station.codec
        mode_name = self._selected_mode_name()
        if codec is None or codec.mode.name != mode_name:
            return
        edits = list(self._clip_edits.items())
        if not edits:
            return
        for index, _edit in edits:
            self.clip_grid.cells[index].set_progress(0.0)
        threading.Thread(
            target=self._prepare_clip_batch,
            args=(edits, mode_name, generation),
            daemon=True,
            name="aetv-clip-preparer",
        ).start()

    def _prepare_clip_batch(
        self,
        edits: list[tuple[int, ClipEdit]],
        mode_name: str,
        generation: int,
    ) -> None:
        for index, edit in edits:
            if generation != self._clip_generation:
                return
            try:
                n_gops = max(1, int(edit.duration_s))
                prepared = self.engine.prepare_clip(
                    edit.path,
                    mode_name,
                    n_gops,
                    on_progress=lambda progress, slot=index: self._clipProgress.emit(
                        slot, progress, generation
                    ),
                    start_s=edit.start_s,
                    framing=edit.framing,
                )
            except Exception as error:
                self._clipFailed.emit(index, str(error), generation)
            else:
                self._clipReady.emit(index, prepared, generation)

    def _apply_clip_progress(self, index: int, progress: float, generation: int) -> None:
        if generation == self._clip_generation:
            self.clip_grid.cells[index].set_progress(progress)

    def _apply_clip_ready(self, index: int, prepared: PreparedClip, generation: int) -> None:
        if generation != self._clip_generation:
            return
        self._prepared_clips[index] = prepared
        self.clip_grid.cells[index].set_ready(prepared.preview_frames)
        self.status.setText(f"{self.clip_grid.cells[index].name.text()} ready")

    def _apply_clip_failed(self, index: int, message: str, generation: int) -> None:
        if generation == self._clip_generation:
            self.clip_grid.cells[index].set_error(message)

    def _fill_screen_targets(self) -> None:
        current = self.screen_target.currentData() if hasattr(self, "screen_target") else None
        self.screen_target.blockSignals(True)
        self.screen_target.clear()
        for index, screen in enumerate(QGuiApplication.screens()):
            rect = screen.geometry()
            spec = ScreenCaptureSpec(
                f"Monitor {index + 1}: {screen.name()}",
                (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()),
            )
            self.screen_target.addItem(spec.name, spec)
        for window in list_windows():
            self.screen_target.addItem(f"Window: {window.name}", window)
        if isinstance(current, ScreenCaptureSpec):
            match = next(
                (
                    index
                    for index in range(self.screen_target.count())
                    if self.screen_target.itemData(index) == current
                ),
                -1,
            )
            if match >= 0:
                self.screen_target.setCurrentIndex(match)
        self.screen_target.blockSignals(False)

    def _choose_region(self) -> None:
        selector = _RegionSelector(self)
        if selector.exec() != QDialog.DialogCode.Accepted or selector.selection is None:
            return
        rect = selector.selection
        x, y, width, height = rect.x(), rect.y(), rect.width(), rect.height()
        spec = ScreenCaptureSpec(
            f"Region {x},{y} {width}×{height}",
            (x, y, x + width, y + height),
        )
        self.screen_target.addItem(spec.name, spec)
        self.screen_target.setCurrentIndex(self.screen_target.count() - 1)
        self.screen_radio.setChecked(True)

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
        try:
            inputs = list_audio_devices("input")
        except AudioUnavailable:
            inputs = []
        self._inputsArrived.emit(inputs)

    def _apply_outputs(self, outputs) -> None:
        self._output_load_active = False
        current = self.output.currentData()
        self.output.clear()
        self.output.addItem("System default", "")
        for item in outputs:
            self.output.addItem(item.label(), item.selection_value())
        index = self.output.findData(current)
        if index < 0 and current not in {None, ""}:
            index = next(
                (i for i, item in enumerate(outputs, start=1) if item.name == str(current)),
                -1,
            )
        self.output.setCurrentIndex(max(0, index))

    def _apply_inputs(self, inputs) -> None:
        current = self.station.settings.microphone_input
        self.microphone.clear()
        self.microphone.addItem("System default", "")
        for item in inputs:
            self.microphone.addItem(item.label(), item.selection_value())
        index = self.microphone.findData(current)
        if index < 0 and current not in {None, ""}:
            index = next(
                (i for i, item in enumerate(inputs, start=1) if item.name == str(current)),
                -1,
            )
        self.microphone.setCurrentIndex(max(0, index))

    def _apply_panel_settings(self) -> None:
        settings = self.station.settings
        settings.mode = self._selected_mode_name()
        settings.waveform_mode = (
            "analog_av" if self.mode.currentData() == "V8_AV" else "video"
        )
        settings.gops = int(self.gops.value())
        settings.tx_level = float(10 ** (self.level_db.value() / 20.0))
        settings.camera_index = int(self.camera.currentData() or 0)
        settings.audio_output = self.output.currentData() or ""
        settings.microphone_input = self.microphone.currentData() or ""
        settings.av_video_power = self.av_power.value() / 100.0
        settings.av_microphone_mix = self.mic_mix.value() / 100.0
        settings.tx_channel_profile = self.selected_channel_profile()

    def selected_channel_profile(self) -> str:
        index = self.channel_buttons.checkedId()
        return self.channel_keys[index] if 0 <= index < len(self.channel_keys) else "radio"

    def emulating(self) -> bool:
        return self.selected_channel_profile() != "radio"

    def _sync_channel_route(self, _button_id: int | None = None) -> None:
        profile = self.selected_channel_profile()
        self.station.settings.tx_channel_profile = profile
        testing = profile != "radio"
        self.output.setEnabled(not testing)
        self.send_button.setText("Run loopback" if testing else "Send")

    def _selected_mode_name(self) -> str:
        return "V8" if self.mode.currentData() == "V8_AV" else (
            self.mode.currentData() or self.station.settings.mode
        )

    def _on_mode_changed(self, index: int) -> None:
        selected = self.mode.currentData()
        base_mode = "V8" if selected == "V8_AV" else (
            selected or self.station.settings.mode
        )
        waveform_mode = (
            "analog_av" if selected == "V8_AV" else "video"
        )
        self.station.settings.waveform_mode = waveform_mode
        if hasattr(self, "_sync_av_controls"):
            self._sync_av_controls()
        self._restart_preview(index)
        if hasattr(self, "_selected_clip"):
            self._selected_clip = None
        if hasattr(self, "_prepared_clips"):
            self._prepared_clips.clear()
        if base_mode != self.station.settings.mode:
            self.send_button.setEnabled(False)
            self.status.setText(f"loading {base_mode} model…")
        self.modeRequested.emit(base_mode)

    def _sync_av_controls(self) -> None:
        visible = self.mode.currentData() == "V8_AV"
        for index in range(self.av_row.count()):
            widget = self.av_row.itemAt(index).widget()
            if widget is not None:
                widget.setVisible(visible)
        for index in range(self.mix_row.count()):
            widget = self.mix_row.itemAt(index).widget()
            if widget is not None:
                widget.setVisible(visible)
        self._on_av_power_changed(self.av_power.value())
        self._on_mic_mix_changed(self.mic_mix.value())

    def _on_av_power_changed(self, value: int) -> None:
        self.station.settings.av_video_power = value / 100.0
        self.av_power_label.setText(f"{100 - value}% / {value}%")

    def _on_mic_mix_changed(self, value: int) -> None:
        self.station.settings.av_microphone_mix = value / 100.0
        self.mic_mix_label.setText(f"{value}% mic")

    def _apply_state(self, state: TxState) -> None:
        self.status.setText(state.message or state.phase.value)
        self.progress.setValue(int(round(state.progress * 100)))
        keyed = state.phase in {TxPhase.KEYING, TxPhase.SENDING, TxPhase.UNKEYING}
        self.pttChanged.emit(keyed)
        if state.phase in {TxPhase.DONE, TxPhase.CANCELLED, TxPhase.FAILED, TxPhase.IDLE}:
            codec = self.station.codec
            self.send_button.setEnabled(
                codec is not None and codec.mode.name == self.station.settings.mode
            )
            self.mode.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.mode.setEnabled(True)
            self.microphone.setEnabled(True)
            for button in self.channel_buttons.buttons():
                button.setEnabled(True)
            self._sync_channel_route()
            if state.phase != TxPhase.IDLE:
                self.transmitFinished.emit()

    def _on_error(self, message: str) -> None:
        self.status.setText(message)
        self.logMessage.emit(message)

    def _show_preview(self, frames) -> None:
        # The high-rate camera subscriber is stopped while TX owns the webcam.
        # During a local loopback, show each newly captured GOP so the source
        # preview remains live while the decoded result plays in the RX pane.
        # Keep radio TX unchanged: painting there has previously destabilized
        # some Windows webcam/CUDA driver combinations.
        if (
            self.transmitting()
            and (self.cam_radio.isChecked() or self.screen_radio.isChecked())
            and not self.emulating()
        ):
            return
        self.preview.set_rgb(frames)

    def _show_camera_preview(self, frame) -> None:
        self._camera_preview_queued.clear()
        if self._webcam_tx_active or self.transmitting():
            return
        self.preview.set_rgb(frame)
