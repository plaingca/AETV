"""Settings dialog: station, audio, rig, KiwiSDR, folders."""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, QTimer, Signal
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
from aetv.cat import CatConfig, list_hamlib_models, list_serial_ports, open_ptt
from aetv.config import AETV_MODES
from aetv.flex import FlexRadioInfo, discover_radios, with_probed_path_mtu
from aetv.kiwi import normalize_kiwi_host
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


class _FlexDiscoveryThread(QThread):
    finished_list = Signal(object, str)

    def __init__(self, configured_host: str = "", parent=None):
        super().__init__(parent)
        self._configured_host = configured_host.strip()

    def run(self) -> None:
        try:
            radios = discover_radios(2.0)
            if self._configured_host and not any(
                radio.ip == self._configured_host for radio in radios
            ):
                # Routed VPNs often do not carry discovery broadcasts. Keep a
                # manually configured radio in the list and probe it directly.
                radios.append(
                    FlexRadioInfo(
                        ip=self._configured_host,
                        nickname="Configured FlexRadio",
                    )
                )
            measured = []
            for radio in radios:
                try:
                    radio = with_probed_path_mtu(radio, timeout=1.5)
                except OSError:
                    pass
                measured.append(radio)
            self.finished_list.emit(measured, "")
        except Exception as error:
            self.finished_list.emit([], str(error))


class _DeviceInventorySignals(QObject):
    finished = Signal(object)


class _DeviceInventoryTask(QRunnable):
    """Enumerate hardware without stalling construction of the dialog."""

    def __init__(self):
        super().__init__()
        self.signals = _DeviceInventorySignals()

    def run(self) -> None:
        inventory = {"cameras": [], "audio": {}, "hamlib": [], "serial": []}
        try:
            inventory["cameras"] = list_cameras()
        except Exception:
            pass
        for kind in ("input", "output"):
            try:
                inventory["audio"][kind] = list_audio_devices(kind)
            except Exception as error:
                inventory["audio"][kind] = error
        try:
            inventory["hamlib"] = list_hamlib_models()
        except Exception:
            pass
        try:
            inventory["serial"] = list_serial_ports()
        except Exception:
            pass
        try:
            self.signals.finished.emit(inventory)
        except RuntimeError:
            # The dialog/application may have closed while enumeration was in
            # flight.  In that case there is no UI left to receive the result.
            pass


class SettingsDialog(QDialog):
    def __init__(self, settings: StationSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AETV settings")
        self._settings = settings
        self._cat_thread: _CatTestThread | None = None
        self._flex_thread: _FlexDiscoveryThread | None = None
        self._inventory_task: _DeviceInventoryTask | None = None
        tabs = QTabWidget()
        tabs.addTab(self._station_tab(), "Station")
        tabs.addTab(self._audio_tab(), "Audio")
        tabs.addTab(self._rig_tab(), "Rig")
        tabs.addTab(self._kiwi_tab(), "Remote receiver")
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
        QTimer.singleShot(0, self._load_device_inventory)
        if self._settings.cat_backend == "flex" and self._settings.flex_host:
            QTimer.singleShot(0, self._discover_flex)

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
        settings.hamlib_model = int(self.hamlib_model.currentData() or 0)
        settings.hamlib_device = self.hamlib_device.currentText().strip()
        settings.hamlib_baud = int(self.hamlib_baud.currentData() or 0)
        settings.flex_host = self.flex_radio.currentData() or self.flex_radio.currentText().strip()
        settings.flex_power = int(self.flex_power.value())
        settings.flex_native_audio = self.flex_native_audio.isChecked()
        if settings.cat_backend == "flex" and settings.flex_native_audio:
            # A native Flex station should not silently keep listening to the
            # system soundcard just because that was the old default.
            settings.rx_source = "flex"
        settings.freq_mhz = self.freq_mhz.value() if self.freq_mhz.value() > 0 else None
        settings.require_mode = self.require_mode.text().strip() or "DIGU"
        settings.serial_port = self.serial_port.currentText().strip()
        settings.ptt_lead_s = float(self.ptt_lead.value())
        settings.ptt_tail_s = float(self.ptt_tail.value())
        settings.audio_only = self.audio_only.isChecked()
        kiwi_text = self.kiwi_host.text().strip()
        try:
            settings.kiwi_host = normalize_kiwi_host(kiwi_text)
        except ValueError:
            settings.kiwi_host = kiwi_text
        settings.kiwi_user = self.kiwi_user.text().strip()
        settings.kiwi_password = self.kiwi_password.text()
        settings.kiwi_dial_mhz = float(self.kiwi_dial.value())
        settings.kiwi_lat = float(self.kiwi_lat.value())
        settings.kiwi_lon = float(self.kiwi_lon.value())
        settings.kiwi_max_km = float(self.kiwi_max_km.value())
        settings.receive_dir = self.receive_dir.text().strip()
        settings.autosave = self.autosave.isChecked()
        settings.debug_capture = self.debug_capture.isChecked()
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
        self.checkpoint.setPlaceholderText("models/v7-flex8k-severe.pt")
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
        self.camera.addItem("Loading cameras…", self._settings.camera_index)
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
        self.audio_input.addItem("Loading devices…", self._settings.audio_input)
        self.audio_output.addItem("Loading devices…", self._settings.audio_output)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self._load_device_inventory)
        self.rx_source = QComboBox()
        self.rx_source.addItem("Soundcard", "soundcard")
        self.rx_source.addItem("FlexRadio network audio (automatic)", "flex")
        self.rx_source.addItem("Public KiwiSDR (remote receive only)", "kiwi")
        self.rx_source.setCurrentIndex(max(0, self.rx_source.findData(self._settings.rx_source)))
        self.buffer_s = QDoubleSpinBox()
        self.buffer_s.setRange(20.0, 300.0)
        self.buffer_s.setValue(self._settings.buffer_seconds)
        self.buffer_s.setSuffix(" s")
        self.decode_every = QDoubleSpinBox()
        self.decode_every.setRange(0.05, 2.0)
        self.decode_every.setSingleStep(0.05)
        self.decode_every.setValue(self._settings.decode_every_s)
        self.decode_every.setSuffix(" s")
        form.addRow("Receive source", self.rx_source)
        form.addRow("Input", self.audio_input)
        form.addRow("Output (to radio)", self.audio_output)
        form.addRow("", refresh)
        form.addRow("Receive buffer", self.buffer_s)
        form.addRow("RX poll interval", self.decode_every)
        return page

    def _rig_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.cat_backend = QComboBox()
        for value, label in (
            ("none", "None (VOX / manual PTT)"),
            ("hamlib", "Hamlib — connect directly to my radio"),
            ("flex", "FlexRadio — discover on my network"),
            ("rts", "Serial RTS"),
            ("dtr", "Serial DTR"),
        ):
            self.cat_backend.addItem(label, value)
        self.cat_backend.setCurrentIndex(max(0, self.cat_backend.findData(self._settings.cat_backend)))
        self.hamlib_model = QComboBox()
        self.hamlib_model.setEditable(True)
        self.hamlib_model.addItem("Loading radio models…", self._settings.hamlib_model)
        self.hamlib_device = QComboBox()
        self.hamlib_device.setEditable(True)
        self.hamlib_device.setCurrentText(self._settings.hamlib_device)
        self.hamlib_device.setPlaceholderText("COM3 or network address")
        self.hamlib_baud = QComboBox()
        for baud in (0, 4800, 9600, 19200, 38400, 57600, 115200):
            self.hamlib_baud.addItem("Radio default" if baud == 0 else str(baud), baud)
        baud_index = self.hamlib_baud.findData(self._settings.hamlib_baud)
        self.hamlib_baud.setCurrentIndex(max(0, baud_index))
        self.flex_radio = QComboBox()
        self.flex_radio.setEditable(True)
        self.flex_radio.setPlaceholderText("Radio is discovered automatically")
        if self._settings.flex_host:
            self.flex_radio.addItem(self._settings.flex_host, self._settings.flex_host)
        discover = QPushButton("Discover radios")
        discover.clicked.connect(self._discover_flex)
        flex_row = QWidget()
        flex_layout = QHBoxLayout(flex_row)
        flex_layout.setContentsMargins(0, 0, 0, 0)
        flex_layout.addWidget(self.flex_radio, 1)
        flex_layout.addWidget(discover)
        self.flex_mtu = QLabel("Not measured")
        self.flex_power = QSpinBox()
        self.flex_power.setRange(1, 100)
        self.flex_power.setValue(self._settings.flex_power)
        self.flex_power.setSuffix(" W")
        self.flex_native_audio = QCheckBox("Use direct VITA-49 network audio (no DAX device)")
        self.flex_native_audio.setChecked(self._settings.flex_native_audio)
        self.freq_mhz = QDoubleSpinBox()
        self.freq_mhz.setDecimals(6)
        self.freq_mhz.setRange(0.0, 60.0)
        self.freq_mhz.setValue(self._settings.freq_mhz or 0.0)
        self.freq_mhz.setSpecialValueText("use current slice")
        self.require_mode = QLineEdit(self._settings.require_mode)
        self.serial_port = QComboBox()
        self.serial_port.setEditable(True)
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
        self.audio_only = QCheckBox("Send audio without keying (audio-only test)")
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
        form.addRow("Radio connection", self.cat_backend)
        form.addRow("Hamlib radio", self.hamlib_model)
        form.addRow("Radio device", self.hamlib_device)
        form.addRow("Serial speed", self.hamlib_baud)
        form.addRow("FlexRadio", flex_row)
        form.addRow("Current path MTU", self.flex_mtu)
        form.addRow("Flex power", self.flex_power)
        form.addRow(self.flex_native_audio)
        form.addRow("Operating frequency (MHz)", self.freq_mhz)
        form.addRow("Require mode", self.require_mode)
        form.addRow("Serial port", self.serial_port)
        form.addRow("PTT lead", self.ptt_lead)
        form.addRow("PTT tail", self.ptt_tail)
        form.addRow(self.audio_only)
        form.addRow(row)
        note = QLabel(
            "Hamlib runs inside AETV; rigctld is not needed. FlexRadio control, "
            "PTT, receive and transmit audio use the radio's native network APIs."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.cat_backend.currentIndexChanged.connect(self._sync_rig_visibility)
        self._rig_form = form
        self._rig_fields = {
            "hamlib": [self.hamlib_model, self.hamlib_device, self.hamlib_baud],
            "flex": [flex_row, self.flex_mtu, self.flex_power, self.flex_native_audio],
            "serial": [self.serial_port],
        }
        self._sync_rig_visibility()
        self._flex_radios = {}
        self.flex_radio.currentIndexChanged.connect(self._update_flex_mtu)
        return page

    def _kiwi_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.kiwi_host = QLineEdit(self._settings.kiwi_host)
        self.kiwi_host.setPlaceholderText("Paste http://host:8073/ or host:port")
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
        form.addRow("AETV TX dial", self.kiwi_dial)
        form.addRow("Your latitude", self.kiwi_lat)
        form.addRow("Your longitude", self.kiwi_lon)
        form.addRow("Search radius", self.kiwi_max_km)
        note = QLabel(
            "The receive pane can refresh the public Kiwi list and pick a "
            "receiver that still has an API channel. Enter the same suppressed-"
            "carrier dial frequency used by the transmitter; AETV automatically "
            "centres Kiwi IQ 5 kHz higher so the full waveform fits."
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
        self.debug_capture = QCheckBox("Save TX waveform, Kiwi IQ, and modem debug logs")
        self.debug_capture.setChecked(self._settings.debug_capture)
        form.addRow("Received video", row)
        form.addRow(self.autosave)
        form.addRow(self.debug_capture)
        return page

    def _load_device_inventory(self) -> None:
        if self._inventory_task is not None:
            return
        self._inventory_task = _DeviceInventoryTask()
        self._inventory_task.signals.finished.connect(self._apply_device_inventory)
        QThreadPool.globalInstance().start(self._inventory_task)

    def _apply_device_inventory(self, inventory: dict) -> None:
        self._inventory_task = None
        for combo, kind, current in (
            (self.audio_input, "input", self._settings.audio_input),
            (self.audio_output, "output", self._settings.audio_output),
        ):
            combo.clear()
            combo.addItem("System default", "")
            devices = inventory["audio"].get(kind, [])
            if isinstance(devices, Exception):
                combo.addItem(str(devices), "")
                devices = []
            for item in devices:
                combo.addItem(item.label(), item.name)
            index = combo.findData(current)
            if index >= 0:
                combo.setCurrentIndex(index)

        self.camera.clear()
        cameras = inventory["cameras"]
        if not cameras:
            self.camera.addItem("Camera 0", 0)
        for item in cameras:
            self.camera.addItem(item["name"], item["index"])
        index = self.camera.findData(self._settings.camera_index)
        if index >= 0:
            self.camera.setCurrentIndex(index)

        self.hamlib_model.clear()
        self.hamlib_model.addItem("Choose your radio…", 0)
        for item in inventory["hamlib"]:
            self.hamlib_model.addItem(item.label, item.model_id)
        model_index = self.hamlib_model.findData(self._settings.hamlib_model)
        if model_index >= 0:
            self.hamlib_model.setCurrentIndex(model_index)

        for combo, current in (
            (self.hamlib_device, self._settings.hamlib_device),
            (self.serial_port, self._settings.serial_port),
        ):
            combo.clear()
            combo.addItems(inventory["serial"])
            combo.setCurrentText(current)

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
            hamlib_model=int(self.hamlib_model.currentData() or 0),
            hamlib_device=self.hamlib_device.currentText().strip(),
            hamlib_baud=int(self.hamlib_baud.currentData() or 0),
            flex_host=self.flex_radio.currentData() or self.flex_radio.currentText().strip(),
            flex_power=int(self.flex_power.value()),
            freq_mhz=self.freq_mhz.value() if self.freq_mhz.value() > 0 else None,
            require_mode=self.require_mode.text().strip() or None,
            serial_port=self.serial_port.currentText().strip(),
            serial_line=backend if backend in {"rts", "dtr"} else "rts",
        )

    def _sync_rig_visibility(self) -> None:
        backend = self.cat_backend.currentData()
        for group, widgets in self._rig_fields.items():
            visible = (
                backend == group
                or (group == "serial" and backend in {"rts", "dtr"})
            )
            for widget in widgets:
                widget.setVisible(visible)
                label = self._rig_form.labelForField(widget)
                if label is not None:
                    label.setVisible(visible)

    def _discover_flex(self) -> None:
        if self._flex_thread is not None and self._flex_thread.isRunning():
            return
        self.flex_radio.clear()
        self.flex_radio.addItem("Listening for radios…", "")
        configured = self.flex_radio.currentData() or self.flex_radio.currentText().strip()
        if not configured:
            configured = self._settings.flex_host
        self.flex_mtu.setText("Measuring…")
        self._flex_thread = _FlexDiscoveryThread(configured, self)
        self._flex_thread.finished_list.connect(self._on_flex_discovery)
        self._flex_thread.start()

    def _on_flex_discovery(self, radios, error: str) -> None:
        self.flex_radio.clear()
        self._flex_radios = {radio.ip: radio for radio in radios}
        if error:
            self.flex_radio.addItem(f"Discovery failed: {error}", "")
            self.flex_mtu.setText("Unavailable")
            return
        if not radios:
            self.flex_radio.addItem("No radios found — type an IP address", "")
            self.flex_mtu.setText("Unavailable")
            return
        for radio in radios:
            self.flex_radio.addItem(radio.label, radio.ip)
        current = self.flex_radio.findData(self._settings.flex_host)
        if current >= 0:
            self.flex_radio.setCurrentIndex(current)
        self._update_flex_mtu()

    def _update_flex_mtu(self) -> None:
        host = self.flex_radio.currentData() or self.flex_radio.currentText().strip()
        radio = self._flex_radios.get(host)
        if radio is not None and radio.path_mtu:
            udp_payload = radio.path_mtu - 20 - 8
            self.flex_mtu.setText(
                f"{radio.path_mtu} bytes ({udp_payload} bytes available to UDP)"
            )
        else:
            self.flex_mtu.setText("Not measured — click Discover radios")

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

