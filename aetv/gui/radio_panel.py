"""Visible HF and direct-RF tuning and gain controls."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)


class RadioPanel(QWidget):
    applyRequested = Signal(object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setObjectName("radioControls")
        self._settings = settings
        self._rx_source = settings.rx_source
        self._frequency_kind = None
        self._frequencies = {}
        grid = QGridLayout(self)
        self.backend = QComboBox()
        self.backend.addItem("Audio / CAT", "audio")
        self.backend.addItem("PlutoSDR", "pluto")
        self.backend.addItem("HackRF (experimental)", "hackrf")
        self.frequency = QDoubleSpinBox()
        self.frequency.setObjectName("rfFrequency")
        self.frequency.setRange(1.1, 6000)
        self.frequency.setDecimals(6)
        self.frequency.setSingleStep(0.001)
        self.frequency.setSuffix(" MHz")
        self.frequency.setMinimumWidth(200)
        self.frequency.setStyleSheet("font-size: 22px; font-family: monospace;")
        self.uri = QLineEdit()
        self.uri.setPlaceholderText("ip:192.168.2.1 or usb:…")
        self.serial = QLineEdit()
        self.serial.setPlaceholderText("RTL serial, e.g. 1001")
        self.tx_gain, self.tx_value = self._slider(-359, 0, 4)
        self.rx_gain, self.rx_value = self._slider(0, 496, 10)
        self.tx_gain.setObjectName("plutoTxGain")
        self.rx_gain.setObjectName("sdrRxGain")
        self.rx_label = QLabel("RX gain")
        self.auto_correct = QCheckBox("Auto frequency correction (AC16)")
        self.correction = QDoubleSpinBox()
        self.correction.setRange(-25000, 25000)
        self.correction.setDecimals(0)
        self.correction.setSingleStep(50)
        self.correction.setSuffix(" Hz")
        self.correction.setToolTip(
            "Manual correction to the received signal offset; used when auto correction is disabled"
        )
        self.apply = QPushButton("Apply tuning")
        self.apply.setObjectName("applyRadioSettings")
        self.apply.clicked.connect(self._apply)
        grid.addWidget(QLabel("Transmit via"), 0, 0)
        grid.addWidget(self.backend, 0, 1)
        self.frequency_label = QLabel("Operating frequency")
        # Keep routing and tuning on separate rows. Native Windows styles
        # and larger fonts can otherwise force the window wider than 1280 px.
        grid.addWidget(self.frequency_label, 1, 0)
        grid.addWidget(self.frequency, 1, 1, 1, 2)
        self.advanced_toggle = QPushButton("Radio details ▸")
        self.advanced_toggle.setCheckable(True)
        grid.addWidget(self.apply, 1, 3)
        grid.addWidget(self.advanced_toggle, 0, 3)
        grid.setColumnStretch(2, 1)
        self.advanced = QWidget()
        details = QGridLayout(self.advanced)
        details.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.advanced, 2, 0, 1, 4)
        grid = details
        self.pluto_address_label = QLabel("Pluto address")
        grid.addWidget(self.pluto_address_label, 0, 0)
        grid.addWidget(self.uri, 0, 1, 1, 3)
        self.tx_label = QLabel("Pluto TX gain")
        grid.addWidget(self.tx_label, 1, 0)
        grid.addWidget(self.tx_gain, 1, 1)
        grid.addWidget(self.tx_value, 1, 2)
        grid.addWidget(self.rx_label, 1, 3)
        grid.addWidget(self.rx_gain, 1, 4)
        grid.addWidget(self.rx_value, 1, 5)
        self.serial_label = QLabel("RTL serial")
        grid.addWidget(self.serial_label, 2, 0)
        grid.addWidget(self.serial, 2, 1)
        grid.addWidget(self.auto_correct, 2, 2, 1, 2)
        grid.addWidget(self.correction, 2, 4)
        self.hackrf_serial = QLineEdit()
        self.hackrf_serial.setPlaceholderText("Empty = first HackRF; optional hex serial")
        self.hackrf_tx_gain, hackrf_tx_value = self._slider(0, 47, 1)
        self.hackrf_lna, hackrf_lna_value = self._slider(0, 5, 1 / 8)
        self.hackrf_vga, hackrf_vga_value = self._slider(0, 31, 1 / 2)
        self.hackrf_widgets = [
            QLabel("HackRF serial"), self.hackrf_serial,
            QLabel("HackRF TX gain"), self.hackrf_tx_gain, hackrf_tx_value,
            QLabel("HackRF RX LNA"), self.hackrf_lna, hackrf_lna_value,
            QLabel("HackRF RX VGA"), self.hackrf_vga, hackrf_vga_value,
            QLabel("HackRF: half duplex · RF amplifier and antenna bias power off · hardware testing pending"),
        ]
        for widget, row, col, span in zip(self.hackrf_widgets,
                [3]*5 + [4]*6 + [5], [0,1,2,3,5,0,1,2,3,4,5,0],
                [1,1,1,2,1] + [1]*6 + [6]):
            grid.addWidget(widget, row, col, 1, span)
        self.audio_help = QLabel(
            "Audio/CAT setup: open File → Settings → Rig to choose the connection. "
            "Select the radio's audio input and output in the panes below."
        )
        self.audio_help.setWordWrap(True)
        grid.addWidget(self.audio_help, 6, 0, 1, 6)
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self._toggle_details)
        self.auto_correct.toggled.connect(self._visibility)
        self.backend.currentIndexChanged.connect(self._visibility)
        self.sync(settings)

    def _toggle_details(self, expanded):
        self.advanced.setVisible(expanded)
        self.advanced_toggle.setText("Radio details ▾" if expanded else "Radio details ▸")

    def _slider(self, low, high, scale):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        label = QLabel(f"{slider.value() / scale:.2f} dB")
        slider.valueChanged.connect(
            lambda value: label.setText(f"{value / scale:.2f} dB")
        )
        return slider, label

    def sync(self, settings):
        self._settings = settings
        self._frequency_kind = None
        self._frequencies = {"sdr": settings.sdr_frequency_mhz, "hf": settings.freq_mhz or 0.0}
        self.backend.blockSignals(True)
        self.backend.setCurrentIndex(max(0, self.backend.findData(settings.tx_backend)))
        self.backend.blockSignals(False)
        self.uri.setText(settings.pluto_uri)
        self.serial.setText(settings.rtl_serial)
        self.tx_gain.setValue(round(settings.pluto_tx_gain * 4))
        self.hackrf_serial.setText(settings.hackrf_serial)
        self.hackrf_tx_gain.setValue(settings.hackrf_tx_gain)
        self.hackrf_lna.setValue(settings.hackrf_rx_lna_gain // 8)
        self.hackrf_vga.setValue(settings.hackrf_rx_vga_gain // 2)
        self.auto_correct.setChecked(settings.sdr_auto_correct)
        self.correction.setValue(settings.sdr_rx_correction_hz)
        self.set_receive_source(settings.rx_source)

    def set_receive_source(self, source):
        self._rx_source = source
        pluto = source == "pluto"
        self.rx_gain.setMaximum(700 if pluto else 496)
        self.rx_gain.setValue(
            round(
                (self._settings.pluto_rx_gain if pluto else self._settings.rtl_rx_gain)
                * 10
            )
        )
        self.rx_value.setText(f"{self.rx_gain.value() / 10:.2f} dB")
        self.rx_label.setText("Pluto RX gain" if pluto else "RTL RX gain")
        self._visibility()

    def _visibility(self):
        direct_rx = self._rx_source in {"pluto", "rtlsdr", "hackrf"}
        direct_tx = self.backend.currentData() in {"pluto", "hackrf"}
        self.audio_help.setVisible(not direct_rx and not direct_tx)
        hackrf_tx = self.backend.currentData() == "hackrf"
        hackrf_rx = self._rx_source == "hackrf"
        for widget in self.hackrf_widgets:
            widget.setVisible(hackrf_tx or hackrf_rx)
        self.hackrf_tx_gain.setEnabled(hackrf_tx)
        self.hackrf_lna.setEnabled(hackrf_rx)
        self.hackrf_vga.setEnabled(hackrf_rx)
        pluto_tx = self.backend.currentData() == "pluto"
        for widget in (self.tx_label, self.tx_gain, self.tx_value):
            widget.setVisible(pluto_tx)
        for widget in (self.pluto_address_label, self.uri):
            widget.setVisible(pluto_tx or self._rx_source == "pluto")
        for widget in (self.serial_label, self.serial):
            widget.setVisible(self._rx_source == "rtlsdr")
        for widget in (self.rx_label, self.rx_gain, self.rx_value):
            widget.setVisible(self._rx_source in {"pluto", "rtlsdr"})
        self.auto_correct.setVisible(direct_rx)
        self.correction.setVisible(direct_rx)
        self.tx_gain.setEnabled(pluto_tx)
        self.rx_gain.setEnabled(self._rx_source in {"pluto", "rtlsdr"})
        self.serial.setEnabled(self._rx_source == "rtlsdr")
        self.uri.setEnabled(self.backend.currentData() == "pluto" or self._rx_source == "pluto")
        kind = "sdr" if direct_tx or direct_rx else "hf"
        if kind != self._frequency_kind:
            if self._frequency_kind is not None:
                self._frequencies[self._frequency_kind] = self.frequency.value()
            self._frequency_kind = kind
            self.frequency.setRange(*((1.1, 6000) if kind == "sdr" else (0, 60)))
            self.frequency.setSpecialValueText("" if kind == "sdr" else "Use current rig frequency")
            self.frequency.setValue(self._frequencies[kind])
        self.frequency_label.setText("RF center" if kind == "sdr" else "HF operating frequency")
        tunable = self._rx_source == "flex" or (
            not self._settings.audio_only and self._settings.cat_backend in {"flex", "hamlib", "rigctld"}
        )
        self.frequency.setEnabled(kind == "sdr" or tunable)
        self.frequency.setToolTip(
            "RF center frequency for SDR transmit and receive."
            if kind == "sdr" else
            "HF dial frequency used when starting transmit or receive. Apply restarts active receive; "
            "zero keeps the current rig frequency."
            if tunable else
            "Select Flex, Hamlib, or rigctld in Rig settings to tune the HF radio. "
            "VOX and RTS/DTR control PTT only; tune those radios manually."
        )
        self.auto_correct.setEnabled(direct_rx)
        self.correction.setEnabled(direct_rx and not self.auto_correct.isChecked())

    def _apply(self):
        values = dict(
            tx_backend=self.backend.currentData(),
            pluto_uri=self.uri.text().strip(),
            rtl_serial=self.serial.text().strip(),
            pluto_tx_gain=self.tx_gain.value() / 4,
            hackrf_serial=self.hackrf_serial.text().strip(),
            hackrf_tx_gain=self.hackrf_tx_gain.value(),
            hackrf_rx_lna_gain=self.hackrf_lna.value() * 8,
            hackrf_rx_vga_gain=self.hackrf_vga.value() * 2,
            sdr_auto_correct=self.auto_correct.isChecked(),
            sdr_rx_correction_hz=self.correction.value(),
        )
        if self._frequency_kind == "sdr":
            values["sdr_frequency_mhz"] = self.frequency.value()
        elif self.frequency.isEnabled():
            values["freq_mhz"] = self.frequency.value() or None
        if self._rx_source in {"pluto", "rtlsdr"}:
            values["pluto_rx_gain" if self._rx_source == "pluto" else "rtl_rx_gain"] = (
                self.rx_gain.value() / 10
            )
        self.applyRequested.emit(values)
