"""Settings dialog: station, audio, rig, KiwiSDR, folders."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from aetv.audio_io import AudioUnavailable, list_audio_devices
from aetv.cat import CatConfig, list_serial_ports, open_ptt
from aetv.config import AETV_MODES
from aetv.settings import StationSettings, normalize_callsign
from aetv.source import list_cameras


class _CatTestThread(QThread):
    finished_ok = Signal(bool, str)

    def __init__(self, config: CatConfig, key: bool, parent=None):
        super().__init__(parent)
        self._config = config
        self._key = key

    def run(self) -> None:
        try:
            ptt = open_ptt(self._config)
            try:
                text = ptt.describe()
                if self._key:
                    ptt.set_ptt(True)
                    self.msleep(250)
                    ptt.set_ptt(False)
                    text += " — PTT pulsed"
                self.finished_ok.emit(True, text)
            finally:
                if hasattr(ptt, "close"):
                    ptt.close()
        except Exception as error:
            self.finished_ok.emit(False, str(error))


class SettingsDialog(QDialog):
    def __init__(self, settings: StationSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AETV settings")
        self._settings = settings
        self._cat_thread: _CatTestThread | None = None
        tabs = QTabWidget()
        tabs.addTab(self._station_tab(), "Station")
        tabs.addTab(self._audio_tab(), "Audio")
        tabs.addTab(self._rig_tab(), "Rig")
        tabs.addTab(self._kiwi_tab(), "KiwiSDR")
        tabs.addTab(self._folders_tab(), "Folders")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        self.resize(520, 480)

    def apply_to(self, settings: StationSettings) -> None:
        settings.callsign = normalize_callsign(self.callsign.text()) or "N0CALL"
        settings.mode = self.mode.currentData()
        settings.checkpoint = self.checkpoint.text().strip()
        settings.torch_device = self.torch_device.currentText().strip()
        settings.gops = int(self.gops.value())
        settings.camera_index = int(self.camera.currentData() or 0)
        settings.tx_level = float(10 ** (self.level_db.value() / 20.0))
        settings.audio_input = self.audio_input.currentData() or ""
        settings.audio_output = self.audio_output.currentData() or ""
        settings.rx_source = self.rx_source.currentData()
        settings.cat_backend = self.cat_backend.currentData()
        settings.rigctld_host = self.rigctld_host.text().strip()
        settings.rigctld_port = int(self.rigctld_port.value())
        settings.flex_host = self.flex_host.text().strip()
        settings.flex_power = int(self.flex_power.value())
        settings.freq_mhz = self.freq_mhz.value() if self.freq_mhz.value() > 0 else None
        settings.require_mode = self.require_mode.text().strip() or "DIGU"
        settings.serial_port = self.serial_port.currentText().strip()
        settings.ptt_lead_s = float(self.ptt_lead.value())
        settings.ptt_tail_s = float(self.ptt_tail.value())
        settings.audio_only = self.audio_only.isChecked()
        settings.kiwi_host = self.kiwi_host.text().strip()
        settings.kiwi_user = self.kiwi_user.text().strip()
        settings.kiwi_password = self.kiwi_password.text()
        settings.kiwi_dial_mhz = float(self.kiwi_dial.value())
        settings.kiwi_lat = float(self.kiwi_lat.value())
        settings.kiwi_lon = float(self.kiwi_lon.value())
        settings.kiwi_max_km = float(self.kiwi_max_km.value())
        settings.receive_dir = self.receive_dir.text().strip()
        settings.autosave = self.autosave.isChecked()
        settings.buffer_seconds = float(self.buffer_s.value())
        settings.decode_every_s = float(self.decode_every.value())

    def _station_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.callsign = QLineEdit(self._settings.callsign)
        self.callsign.setMaxLength(8)
        self.mode = QComboBox()
        for name, spec in AETV_MODES.items():
            self.mode.addItem(f"{name} — {spec.description}", name)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self._settings.mode)))
        self.checkpoint = QLineEdit(self._settings.checkpoint)
        self.checkpoint.setPlaceholderText("models/v7-flex8k.pt")
        browse = QPushButton("File…")
        browse.clicked.connect(self._browse_checkpoint)
        ck = QWidget()
        ck_row = QHBoxLayout(ck)
        ck_row.setContentsMargins(0, 0, 0, 0)
        ck_row.addWidget(self.checkpoint, 1)
        ck_row.addWidget(browse)
        self.torch_device = QComboBox()
        self.torch_device.setEditable(True)
        self.torch_device.addItems(["", "cuda", "cpu"])
        if self._settings.torch_device:
            self.torch_device.setCurrentText(self._settings.torch_device)
        self.gops = QSpinBox()
        self.gops.setRange(1, 300)
        self.gops.setValue(self._settings.gops)
        self.gops.setSuffix(" s")
        self.camera = QComboBox()
        self._fill_cameras()
        self.level_db = QDoubleSpinBox()
        self.level_db.setRange(-24.0, 0.0)
        self.level_db.setSingleStep(0.5)
        self.level_db.setSuffix(" dB")
        level = max(0.05, min(1.0, self._settings.tx_level))
        self.level_db.setValue(20.0 * __import__("math").log10(level))
        form.addRow("Callsign", self.callsign)
        form.addRow("Mode", self.mode)
        form.addRow("Checkpoint", ck)
        form.addRow("Torch device", self.torch_device)
        form.addRow("Transmit length", self.gops)
        form.addRow("Webcam", self.camera)
        form.addRow("TX peak", self.level_db)
        note = QLabel("The beacon carries this callsign. Identify with your own.")
        note.setWordWrap(True)
        form.addRow("", note)
        return page

    def _audio_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.audio_input = QComboBox()
        self.audio_output = QComboBox()
        self._fill_audio()
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self._fill_audio)
        self.rx_source = QComboBox()
        self.rx_source.addItem("Soundcard", "soundcard")
        self.rx_source.addItem("KiwiSDR (remote IQ)", "kiwi")
        self.rx_source.setCurrentIndex(max(0, self.rx_source.findData(self._settings.rx_source)))
        self.buffer_s = QDoubleSpinBox()
        self.buffer_s.setRange(20.0, 300.0)
        self.buffer_s.setValue(self._settings.buffer_seconds)
        self.buffer_s.setSuffix(" s")
        self.decode_every = QDoubleSpinBox()
        self.decode_every.setRange(0.5, 10.0)
        self.decode_every.setValue(self._settings.decode_every_s)
        self.decode_every.setSuffix(" s")
        form.addRow("Receive source", self.rx_source)
        form.addRow("Input", self.audio_input)
        form.addRow("Output (to radio)", self.audio_output)
        form.addRow("", refresh)
        form.addRow("Receive buffer", self.buffer_s)
        form.addRow("Decode every", self.decode_every)
        return page

    def _rig_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.cat_backend = QComboBox()
        for value, label in (
            ("none", "None (VOX / manual PTT)"),
            ("rigctld", "Hamlib rigctld (TCP)"),
            ("flex", "FlexRadio 6000 (SmartSDR)"),
            ("rts", "Serial RTS"),
            ("dtr", "Serial DTR"),
        ):
            self.cat_backend.addItem(label, value)
        self.cat_backend.setCurrentIndex(max(0, self.cat_backend.findData(self._settings.cat_backend)))
        self.rigctld_host = QLineEdit(self._settings.rigctld_host)
        self.rigctld_port = QSpinBox()
        self.rigctld_port.setRange(1, 65535)
        self.rigctld_port.setValue(self._settings.rigctld_port)
        self.flex_host = QLineEdit(self._settings.flex_host)
        self.flex_host.setPlaceholderText("192.168.88.239")
        self.flex_power = QSpinBox()
        self.flex_power.setRange(1, 100)
        self.flex_power.setValue(self._settings.flex_power)
        self.flex_power.setSuffix(" W")
        self.freq_mhz = QDoubleSpinBox()
        self.freq_mhz.setDecimals(6)
        self.freq_mhz.setRange(0.0, 30.0)
        self.freq_mhz.setValue(self._settings.freq_mhz or 0.0)
        self.freq_mhz.setSpecialValueText("do not check")
        self.require_mode = QLineEdit(self._settings.require_mode)
        self.serial_port = QComboBox()
        self.serial_port.setEditable(True)
        self.serial_port.addItems(list_serial_ports())
        if self._settings.serial_port:
            self.serial_port.setCurrentText(self._settings.serial_port)
        self.ptt_lead = QDoubleSpinBox()
        self.ptt_lead.setRange(0.0, 2.0)
        self.ptt_lead.setSingleStep(0.05)
        self.ptt_lead.setSuffix(" s")
        self.ptt_lead.setValue(self._settings.ptt_lead_s)
        self.ptt_tail = QDoubleSpinBox()
        self.ptt_tail.setRange(0.0, 2.0)
        self.ptt_tail.setSingleStep(0.05)
        self.ptt_tail.setSuffix(" s")
        self.ptt_tail.setValue(self._settings.ptt_tail_s)
        self.audio_only = QCheckBox("Play DAX / soundcard without keying (audio-only)")
        self.audio_only.setChecked(self._settings.audio_only)
        test_cat = QPushButton("Test CAT")
        test_ptt = QPushButton("Test PTT")
        test_cat.clicked.connect(lambda: self._test_cat(False))
        test_ptt.clicked.connect(lambda: self._test_cat(True))
        row = QWidget()
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.addWidget(test_cat)
        row_l.addWidget(test_ptt)
        row_l.addStretch(1)
        form.addRow("PTT method", self.cat_backend)
        form.addRow("rigctld host", self.rigctld_host)
        form.addRow("rigctld port", self.rigctld_port)
        form.addRow("Flex host", self.flex_host)
        form.addRow("Flex power", self.flex_power)
        form.addRow("Check frequency (MHz)", self.freq_mhz)
        form.addRow("Require mode", self.require_mode)
        form.addRow("Serial port", self.serial_port)
        form.addRow("PTT lead", self.ptt_lead)
        form.addRow("PTT tail", self.ptt_tail)
        form.addRow(self.audio_only)
        form.addRow(row)
        note = QLabel(
            "Frequency and mode are checked, never set. Tune the radio first. "
            "AETV V7 needs a wide DIGU slice, about 800–9200 Hz."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _kiwi_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.kiwi_host = QLineEdit(self._settings.kiwi_host)
        self.kiwi_host.setPlaceholderText("host:8073")
        self.kiwi_user = QLineEdit(self._settings.kiwi_user or self._settings.callsign)
        self.kiwi_password = QLineEdit(self._settings.kiwi_password)
        self.kiwi_password.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.kiwi_dial = QDoubleSpinBox()
        self.kiwi_dial.setDecimals(6)
        self.kiwi_dial.setRange(0.1, 30.0)
        self.kiwi_dial.setValue(self._settings.kiwi_dial_mhz)
        self.kiwi_dial.setSuffix(" MHz")
        self.kiwi_lat = QDoubleSpinBox()
        self.kiwi_lat.setDecimals(3)
        self.kiwi_lat.setRange(-90.0, 90.0)
        self.kiwi_lat.setValue(self._settings.kiwi_lat)
        self.kiwi_lon = QDoubleSpinBox()
        self.kiwi_lon.setDecimals(3)
        self.kiwi_lon.setRange(-180.0, 180.0)
        self.kiwi_lon.setValue(self._settings.kiwi_lon)
        self.kiwi_max_km = QDoubleSpinBox()
        self.kiwi_max_km.setRange(50.0, 20000.0)
        self.kiwi_max_km.setValue(self._settings.kiwi_max_km)
        self.kiwi_max_km.setSuffix(" km")
        form.addRow("Host", self.kiwi_host)
        form.addRow("Ident / user", self.kiwi_user)
        form.addRow("Password", self.kiwi_password)
        form.addRow("USB dial", self.kiwi_dial)
        form.addRow("Your latitude", self.kiwi_lat)
        form.addRow("Your longitude", self.kiwi_lon)
        form.addRow("Search radius", self.kiwi_max_km)
        note = QLabel(
            "The receive pane can refresh the public Kiwi list and pick a "
            "receiver that still has an API channel. V7 is received as IQ "
            "centred on dial + 5 kHz so the 8 kHz waveform fits the 12 kHz stream."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _folders_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.receive_dir = QLineEdit(self._settings.receive_dir)
        browse = QPushButton("Folder…")
        browse.clicked.connect(self._browse_receive)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.receive_dir, 1)
        layout.addWidget(browse)
        self.autosave = QCheckBox("Autosave each decoded reception")
        self.autosave.setChecked(self._settings.autosave)
        form.addRow("Received video", row)
        form.addRow(self.autosave)
        return page

    def _fill_audio(self) -> None:
        for combo, kind, current in (
            (self.audio_input, "input", self._settings.audio_input),
            (self.audio_output, "output", self._settings.audio_output),
        ):
            combo.clear()
            combo.addItem("System default", "")
            try:
                devices = list_audio_devices(kind)
            except AudioUnavailable as error:
                combo.addItem(str(error), "")
                continue
            for item in devices:
                combo.addItem(item.label(), item.name)
            index = combo.findData(current)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _fill_cameras(self) -> None:
        self.camera.clear()
        try:
            cameras = list_cameras()
        except Exception:
            cameras = []
        if not cameras:
            self.camera.addItem("Camera 0", 0)
        for item in cameras:
            self.camera.addItem(item["name"], item["index"])
        index = self.camera.findData(self._settings.camera_index)
        if index >= 0:
            self.camera.setCurrentIndex(index)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "AETV checkpoint", "", "PyTorch (*.pt);;All files (*)")
        if path:
            self.checkpoint.setText(path)

    def _browse_receive(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Received video folder")
        if path:
            self.receive_dir.setText(path)

    def pending_cat(self) -> CatConfig:
        backend = self.cat_backend.currentData()
        return CatConfig(
            backend=backend,
            rigctld_host=self.rigctld_host.text().strip(),
            rigctld_port=int(self.rigctld_port.value()),
            flex_host=self.flex_host.text().strip(),
            flex_power=int(self.flex_power.value()),
            freq_mhz=self.freq_mhz.value() if self.freq_mhz.value() > 0 else None,
            require_mode=self.require_mode.text().strip() or None,
            serial_port=self.serial_port.currentText().strip(),
            serial_line=backend if backend in {"rts", "dtr"} else "rts",
        )

    def _test_cat(self, key: bool) -> None:
        if self._cat_thread is not None and self._cat_thread.isRunning():
            return
        self._cat_thread = _CatTestThread(self.pending_cat(), key, self)
        self._cat_thread.finished_ok.connect(self._on_cat_test)
        self._cat_thread.start()

    def _on_cat_test(self, ok: bool, message: str) -> None:
        if ok:
            QMessageBox.information(self, "CAT", message)
        else:
            QMessageBox.warning(self, "CAT", message)

