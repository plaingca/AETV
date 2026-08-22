"""Receive pane: decoded video, source picker, KiwiSDR list, start/stop."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aetv.audio_io import AudioUnavailable, list_audio_devices
from aetv.kiwi import KiwiReceiver, find_receivers
from aetv.station import RxEngine, RxState
from aetv.gui.widgets import ElidingLabel, VideoView


class _KiwiListThread(QThread):
    finished_list = Signal(object, str)

    def __init__(self, lat: float, lon: float, max_km: float, parent=None):
        super().__init__(parent)
        self._lat = lat
        self._lon = lon
        self._max_km = max_km

    def run(self) -> None:
        try:
            receivers = find_receivers(self._lat, self._lon, max_km=self._max_km)
            self.finished_list.emit(receivers, "")
        except Exception as error:
            self.finished_list.emit([], str(error))


class ReceivePanel(QWidget):
    statusChanged = Signal(str)
    listeningChanged = Signal(bool)
    logMessage = Signal(str)
    _stateArrived = Signal(object)
    _errorArrived = Signal(str)
    _videoArrived = Signal(object, object)
    _ringArrived = Signal(object)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self.engine = RxEngine(
            station,
            on_state=self._stateArrived.emit,
            on_error=self._errorArrived.emit,
            on_video=lambda video, state: self._videoArrived.emit(video, state),
            on_ring=self._ringArrived.emit,
        )
        self._waterfall = None
        self._kiwi_thread: _KiwiListThread | None = None
        self._receivers: list[KiwiReceiver] = []
        self._stateArrived.connect(self._apply_state, Qt.ConnectionType.QueuedConnection)
        self._errorArrived.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._videoArrived.connect(self._show_video, Qt.ConnectionType.QueuedConnection)
        self._ringArrived.connect(self._apply_ring, Qt.ConnectionType.QueuedConnection)
        self._build()

    def attach_waterfall(self, waterfall) -> None:
        self._waterfall = waterfall

    def control_strip(self) -> QWidget:
        return self._strip

    def picture_area(self) -> QWidget:
        return self.preview

    def listening(self) -> bool:
        return self.engine.listening

    def sync_from_config(self) -> None:
        settings = self.station.settings
        self.source.setCurrentIndex(max(0, self.source.findData(settings.rx_source)))
        self._fill_inputs()
        self.kiwi_host.setText(settings.kiwi_host)
        self._sync_source_visibility()

    def start(self) -> bool:
        self._apply_panel_settings()
        problems = self.station.settings.validate()
        if self.station.settings.rx_source == "kiwi" and not self.station.settings.kiwi_host:
            problems.append("pick a KiwiSDR or type a host:port")
        if problems:
            self.status.setText(problems[0])
            self.statusChanged.emit(problems[0])
            return False
        try:
            self.engine.start()
        except Exception as error:
            self.status.setText(str(error))
            self.statusChanged.emit(str(error))
            self.logMessage.emit(str(error))
            return False
        self._set_listening(True)
        return True

    def stop(self) -> None:
        self.engine.stop()
        self.progress.setValue(0)
        self._set_listening(False)
        if self._waterfall is not None:
            self._waterfall.set_ring(None)

    def save_current(self) -> None:
        try:
            path = self.engine.save_current()
        except Exception as error:
            self.status.setText(str(error))
            return
        if path is None:
            self.status.setText("nothing to save yet")
            return
        self.logMessage.emit(f"saved {path}")
        self.status.setText(f"saved {path.name}")

    def _build(self) -> None:
        self.preview = VideoView("Start receiving")
        self.status = ElidingLabel("Stopped")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.source = QComboBox()
        self.source.addItem("Soundcard", "soundcard")
        self.source.addItem("KiwiSDR", "kiwi")
        self.source.currentIndexChanged.connect(self._sync_source_visibility)
        self.input_device = QComboBox()
        self.kiwi_host = QLineEdit()
        self.kiwi_host.setPlaceholderText("host:8073")
        self.kiwi_list = QComboBox()
        self.kiwi_list.setMinimumWidth(180)
        self.kiwi_list.activated.connect(self._apply_kiwi_choice)
        refresh_audio = QPushButton("Refresh")
        refresh_audio.clicked.connect(self._fill_inputs)
        self.find_button = QPushButton("Find Kiwis")
        self.find_button.clicked.connect(self._refresh_kiwis)
        self.start_button = QPushButton("Start receiving")
        self.stop_button = QPushButton("Stop")
        self.save_button = QPushButton("Save video…")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self.save_current)

        self._strip = QWidget()
        strip = QVBoxLayout(self._strip)
        strip.setContentsMargins(0, 0, 0, 0)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source"))
        row1.addWidget(self.source)
        row1.addWidget(self.input_device, 1)
        row1.addWidget(refresh_audio)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Kiwi"))
        row2.addWidget(self.kiwi_host, 1)
        row2.addWidget(self.kiwi_list, 1)
        row2.addWidget(self.find_button)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        strip.addLayout(row1)
        strip.addLayout(row2)
        strip.addWidget(self.status)
        strip.addWidget(self.progress)
        strip.addLayout(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self._strip, 0)
        self.sync_from_config()

    def _sync_source_visibility(self) -> None:
        kiwi = self.source.currentData() == "kiwi"
        self.input_device.setVisible(not kiwi)
        self.kiwi_host.setVisible(kiwi)
        self.kiwi_list.setVisible(kiwi)
        self.find_button.setVisible(kiwi)

    def _fill_inputs(self) -> None:
        current = self.station.settings.audio_input
        self.input_device.clear()
        self.input_device.addItem("System default", "")
        try:
            for item in list_audio_devices("input"):
                self.input_device.addItem(item.label(), item.name)
        except AudioUnavailable:
            return
        index = self.input_device.findData(current)
        if index >= 0:
            self.input_device.setCurrentIndex(index)

    def _apply_panel_settings(self) -> None:
        settings = self.station.settings
        settings.rx_source = self.source.currentData()
        settings.audio_input = self.input_device.currentData() or ""
        settings.kiwi_host = self.kiwi_host.text().strip()

    def _refresh_kiwis(self) -> None:
        if self._kiwi_thread is not None and self._kiwi_thread.isRunning():
            return
        settings = self.station.settings
        self.find_button.setEnabled(False)
        self.status.setText("searching public KiwiSDR list…")
        self._kiwi_thread = _KiwiListThread(settings.kiwi_lat, settings.kiwi_lon, settings.kiwi_max_km, self)
        self._kiwi_thread.finished_list.connect(self._on_kiwi_list)
        self._kiwi_thread.start()

    def _on_kiwi_list(self, receivers, error: str) -> None:
        self.find_button.setEnabled(True)
        if error:
            self.status.setText(error)
            self.logMessage.emit(error)
            return
        self._receivers = list(receivers)
        self.kiwi_list.clear()
        usable = [item for item in self._receivers if item.usable]
        shown = usable or self._receivers[:40]
        if not shown:
            self.kiwi_list.addItem("no reachable receivers", "")
            self.status.setText("no reachable KiwiSDRs in range")
            return
        for item in shown:
            mark = "API" if item.usable else "busy"
            self.kiwi_list.addItem(f"{mark}  {item.label()}", item.host)
        self.status.setText(f"{len(usable)} usable of {len(self._receivers)} reachable")
        if usable:
            self.kiwi_host.setText(usable[0].host)
            self.station.settings.kiwi_host = usable[0].host

    def _apply_kiwi_choice(self, index: int) -> None:
        host = self.kiwi_list.itemData(index)
        if host:
            self.kiwi_host.setText(str(host))
            self.station.settings.kiwi_host = str(host)

    def _set_listening(self, on: bool) -> None:
        self.start_button.setEnabled(not on)
        self.stop_button.setEnabled(on)
        self.listeningChanged.emit(on)

    def _apply_state(self, state: RxState) -> None:
        self.status.setText(state.message)
        self.statusChanged.emit(state.message)
        self.progress.setValue(min(100, state.gops * 10) if state.listening else 0)

    def _on_error(self, message: str) -> None:
        self.status.setText(message)
        self.logMessage.emit(message)

    def _show_video(self, video, state: RxState) -> None:
        self.preview.set_rgb(video)
        self.status.setText(state.message)
        self.statusChanged.emit(state.message)
        self.progress.setValue(min(100, max(5, state.gops * 8)))

    def _apply_ring(self, ring) -> None:
        if self._waterfall is None:
            return
        self._waterfall.set_mode(self.station.settings.mode)
        self._waterfall.set_ring(ring)
        if ring is None:
            self._waterfall.clear()
