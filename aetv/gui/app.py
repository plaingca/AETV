"""AETV ham-station window: receive | transmit, waterfall, CAT, KiwiSDR."""

from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from aetv.settings import StationSettings, load_settings, save_settings
from aetv.station import Station
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.settings_dialog import SettingsDialog
from aetv.gui.tx_panel import TransmitPanel
from aetv.gui.waterfall import Waterfall
from aetv.gui.widgets import LogPane, PttLamp


class _CodecThread(QThread):
    loaded = Signal(str)
    failed = Signal(str)

    def __init__(self, station: Station, parent=None):
        super().__init__(parent)
        self._station = station

    def run(self) -> None:
        try:
            codec = self._station.load_codec()
            device = str(codec.device)
            self.loaded.emit(f"{codec.mode.name} on {device}")
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self, settings: StationSettings | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AETV — Autoencoder Television")
        self.settings = settings or load_settings()
        self.station = Station(self.settings)
        self._codec_thread: _CodecThread | None = None
        self._build()
        self._load_codec()

    def _build(self) -> None:
        self.rx = ReceivePanel(self.station)
        self.tx = TransmitPanel(self.station)
        self.waterfall = Waterfall()
        self.waterfall.set_mode(self.settings.mode)
        self.rx.attach_waterfall(self.waterfall)

        panes = QSplitter(Qt.Orientation.Horizontal)
        panes.addWidget(self._wrap("Receive", self.rx))
        panes.addWidget(self._wrap("Transmit", self.tx))
        panes.setChildrenCollapsible(False)
        panes.setStretchFactor(0, 1)
        panes.setStretchFactor(1, 1)

        stack = QSplitter(Qt.Orientation.Vertical)
        stack.addWidget(self.waterfall)
        stack.addWidget(panes)
        stack.setStretchFactor(0, 0)
        stack.setStretchFactor(1, 1)
        stack.setSizes([170, 520])
        self.setCentralWidget(stack)

        self.log = LogPane()
        dock = QDockWidget("Log", self)
        dock.setObjectName("log")
        dock.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._log_dock = dock

        self.ptt_lamp = PttLamp()
        self.station_label = QLabel()
        self.rig_label = QLabel("CAT off")
        self.model_label = QLabel("Loading model…")
        self.rx_status = QLabel()
        bar = QStatusBar()
        bar.addWidget(self.ptt_lamp)
        bar.addWidget(self.station_label)
        bar.addWidget(self.rig_label)
        bar.addPermanentWidget(self.rx_status, 1)
        bar.addPermanentWidget(self.model_label)
        self.setStatusBar(bar)
        self._refresh_station_label()

        self.rx.statusChanged.connect(self.rx_status.setText)
        self.rx.logMessage.connect(self._log)
        self.tx.logMessage.connect(self._log)
        self.tx.pttChanged.connect(self.ptt_lamp.set_keyed)
        self.tx.transmitStarted.connect(self._on_tx_started)
        self.tx.transmitFinished.connect(self._on_tx_finished)

        self._build_menu()
        self.resize(1280, 800)
        self._log(
            "Identify with your own callsign. Confirm you are authorized "
            "for the frequency, bandwidth, and power before keying."
        )

    def _wrap(self, title: str, widget: QWidget) -> QWidget:
        from PySide6.QtWidgets import QGroupBox, QVBoxLayout

        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        settings_action = QAction("&Settings…", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.open_settings)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        receive_menu = self.menuBar().addMenu("&Receive")
        start = QAction("Start receiving", self)
        start.triggered.connect(self.rx.start)
        stop = QAction("Stop", self)
        stop.triggered.connect(self.rx.stop)
        receive_menu.addAction(start)
        receive_menu.addAction(stop)

        transmit_menu = self.menuBar().addMenu("&Transmit")
        send = QAction("&Send", self)
        send.setShortcut(QKeySequence("Ctrl+Return"))
        send.triggered.connect(self.tx.send)
        cancel = QAction("Cancel", self)
        cancel.setShortcut(QKeySequence("Escape"))
        cancel.triggered.connect(self.tx.cancel)
        transmit_menu.addAction(send)
        transmit_menu.addAction(cancel)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About AETV", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _load_codec(self) -> None:
        self.model_label.setText("Loading model…")
        self.tx.send_button.setEnabled(False)
        self.rx.start_button.setEnabled(False)
        self._codec_thread = _CodecThread(self.station, self)
        self._codec_thread.loaded.connect(self._on_model_loaded)
        self._codec_thread.failed.connect(self._on_model_failed)
        self._codec_thread.start()

    def _on_model_loaded(self, text: str) -> None:
        self.model_label.setText(text)
        self.tx.send_button.setEnabled(True)
        self.rx.start_button.setEnabled(True)
        self.waterfall.set_mode(self.settings.mode)
        self._log(f"codec ready: {text}")

    def _on_model_failed(self, message: str) -> None:
        self.model_label.setText("No checkpoint")
        self._log(message)
        QMessageBox.warning(
            self,
            "AETV checkpoint",
            message + "\n\nCopy the published Flex-8k weights to models/v7-flex8k.pt.",
        )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        dialog.apply_to(self.settings)
        save_settings(self.settings)
        self.station.settings = self.settings
        self.rx.sync_from_config()
        self.tx.sync_from_config()
        self.waterfall.set_mode(self.settings.mode)
        self._refresh_station_label()
        self._log("settings saved")
        if self.station.codec is None or self.station.codec.mode.name != self.settings.mode:
            self._load_codec()

    def _refresh_station_label(self) -> None:
        backend = self.settings.cat_backend if not self.settings.audio_only else "audio-only"
        self.station_label.setText(f"  {self.settings.callsign}  {self.settings.mode}  ")
        if backend == "none":
            self.rig_label.setText("CAT off")
        elif backend == "flex":
            self.rig_label.setText(f"Flex {self.settings.flex_host or '?'}")
        elif backend == "rigctld":
            self.rig_label.setText(f"rigctld {self.settings.rigctld_host}:{self.settings.rigctld_port}")
        else:
            self.rig_label.setText(f"{backend} {self.settings.serial_port}")

    def _on_tx_started(self) -> None:
        if self.rx.listening():
            self.rx.stop()
            self._resume_rx = True
            self._log("receive paused for half-duplex transmit")
        else:
            self._resume_rx = False

    def _on_tx_finished(self) -> None:
        self.ptt_lamp.set_keyed(False)
        if getattr(self, "_resume_rx", False):
            self.rx.start()
            self._resume_rx = False

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append_line(f"{stamp}  {message}")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About AETV",
            "AETV — Autoencoder Television\n"
            "Analog video over HF OFDM for amateur radio.\n\n"
            "Published mode: V7 Flex-8k, 256×144 @ 12 fps, 24 kHz DAX.\n"
            "Identify every transmission. This software does not replace "
            "a license or a band-plan check.\n\n"
            "Artistic License 2.0",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.tx.transmitting():
            answer = QMessageBox.question(
                self,
                "AETV",
                "A transmission is in progress. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.tx.cancel()
            if self.tx._thread is not None:
                self.tx._thread.join(timeout=8.0)
        if self.rx.listening():
            self.rx.stop()
        save_settings(self.settings)
        event.accept()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    app = QApplication(args)
    app.setApplicationName("AETV")
    app.setApplicationDisplayName("AETV")
    app.setOrganizationName("AETV")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
