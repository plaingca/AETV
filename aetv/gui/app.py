"""AETV ham-station window: receive | transmit, waterfall, CAT, KiwiSDR."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QKeySequence
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

from aetv.codec import AETVCodec
from aetv.settings import StationSettings, load_settings, save_settings
from aetv.config import RELEASE_MODES
from aetv.hfchannel import CHANNEL_PROFILES
from aetv.station import Station
from aetv.source import write_video_smoke_test
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.model_manager import ModelInventoryThread, ModelManagerDialog
from aetv.gui.settings_dialog import SettingsDialog
from aetv.gui.tx_panel import TransmitPanel
from aetv.gui.waterfall import Waterfall
from aetv.gui.widgets import LogPane, PttLamp


APP_ICON = Path(__file__).resolve().parent.parent / "assets" / "aetv-logo.png"


class _CodecThread(QThread):
    loaded = Signal(object, str)
    failed = Signal(str)

    def __init__(self, config: tuple[str, str, str], parent=None):
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        try:
            # GUI downloads are an explicit Model Manager action. Keeping
            # network access out of codec construction makes download timing
            # visible to the operator.
            mode, checkpoint, device = self.config
            codec = AETVCodec(
                checkpoint=checkpoint or None,
                device=device or None,
                mode=mode,
                allow_download=False,
            )
            device = str(codec.device)
            self.loaded.emit(
                codec, f"{codec.mode.name} on {device} ({codec.backend})"
            )
        except Exception as error:
            self.failed.emit(str(error))


def _codec_config(settings: StationSettings) -> tuple[str, str, str]:
    return settings.mode, settings.checkpoint, settings.torch_device


def _rx_runtime_config(settings: StationSettings) -> tuple:
    """Settings captured by an active receive engine rather than read live."""
    return (
        settings.rx_source,
        settings.audio_input,
        settings.flex_host,
        settings.flex_power,
        settings.freq_mhz,
        settings.require_mode,
        settings.kiwi_host,
        settings.kiwi_dial_mhz,
        settings.kiwi_user,
        settings.kiwi_password,
        settings.callsign,
        settings.buffer_seconds,
        settings.decode_every_s,
        settings.debug_capture,
    )


def _smoke_codec_result(window) -> int | None:
    """Return only after Qt has delivered a definitive codec outcome."""
    if window._model_inventory_thread is not None:
        return None
    if window.station.codec is not None:
        return 0
    if window._last_codec_error:
        return 1
    return None


class MainWindow(QMainWindow):
    def __init__(self, settings: StationSettings | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AETV — Autoencoder Television")
        if APP_ICON.is_file():
            self.setWindowIcon(QIcon(str(APP_ICON)))
        self.settings = settings or load_settings()
        self.station = Station(self.settings)
        self._codec_thread: _CodecThread | None = None
        self._tx_waiting_for_rx = False
        self._tx_finished_while_stopping_rx = False
        self._resume_rx = False
        self._reload_codec_after_rx_stop = False
        self._resume_rx_after_codec_reload = False
        self._restart_rx_after_settings_stop = False
        self._emulation_active = False
        self._path_planner = None
        self._ft8_calibration = None
        self._model_inventory_thread: ModelInventoryThread | None = None
        self._last_codec_error = ""
        self._smoke_test = False
        self._build()
        QTimer.singleShot(0, self._begin_startup_model_check)

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
        self.station.set_logger(self._log)
        dock = QDockWidget("Log", self)
        dock.setObjectName("log")
        dock.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._log_dock = dock

        self.ptt_lamp = PttLamp()
        self.station_label = QLabel()
        self.rig_label = QLabel("CAT off")
        self.model_label = QLabel("Checking installed models…")
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
        self.rx.stopFinished.connect(self._on_rx_stopped_for_tx)
        self.rx.stopFinished.connect(self._on_rx_stopped_for_codec_reload)
        self.rx.stopFinished.connect(self._on_rx_stopped_for_settings_restart)
        self.rx.logMessage.connect(self._log)
        self.rx.pathPlannerRequested.connect(self.open_path_planner)
        self.tx.logMessage.connect(self._log)
        self.tx.pttChanged.connect(self.ptt_lamp.set_keyed)
        self.tx.pttChanged.connect(self.rx.on_local_ptt_changed)
        self.tx.transmitStarted.connect(self._on_tx_started)
        self.tx.transmitFinished.connect(self._on_tx_finished)
        self.tx.modeRequested.connect(self._on_tx_mode_requested)
        self.tx.loopbackVideo.connect(self.rx.show_emulated)

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
        models_action = QAction("&Model Manager…", self)
        models_action.triggered.connect(self.open_model_manager)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(settings_action)
        file_menu.addAction(models_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        receive_menu = self.menuBar().addMenu("&Receive")
        start = QAction("Start receiving", self)
        start.triggered.connect(self.rx.start)
        stop = QAction("Stop", self)
        stop.triggered.connect(self.rx.stop)
        receive_menu.addAction(start)
        receive_menu.addAction(stop)
        receive_menu.addSeparator()
        planner = QAction("Kiwi path planner…", self)
        planner.triggered.connect(self.open_path_planner)
        receive_menu.addAction(planner)
        ft8_calibration = QAction("FT8 propagation calibration…", self)
        ft8_calibration.triggered.connect(self.open_ft8_calibration)
        receive_menu.addAction(ft8_calibration)

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
        config = _codec_config(self.settings)
        self._last_codec_error = ""
        self.model_label.setText(f"Loading {config[0]} model…")
        self.tx.send_button.setEnabled(False)
        self.rx.start_button.setEnabled(False)
        current = self._codec_thread
        if current is not None and current.isRunning():
            return
        thread = _CodecThread(config, self)
        thread.loaded.connect(
            lambda codec, text, worker=thread: self._on_model_loaded(
                worker, codec, text
            )
        )
        thread.failed.connect(
            lambda message, worker=thread: self._on_model_failed(worker, message)
        )
        thread.finished.connect(
            lambda worker=thread: self._on_codec_thread_finished(worker)
        )
        self._codec_thread = thread
        thread.start()

    def _on_model_loaded(self, thread: _CodecThread, codec, text: str) -> None:
        if thread.config != _codec_config(self.settings):
            self._log(
                f"discarding stale codec load ({text}); loading {self.settings.mode}"
            )
            return
        self._last_codec_error = ""
        with self.station.codec_lock:
            self.station.codec = codec
        self.model_label.setText(text)
        self.model_label.setToolTip(
            f"Model: {self.station.codec.checkpoint_path}\n"
            "Install or inspect release models with File > Model Manager."
        )
        self.tx.send_button.setEnabled(True)
        self.tx.model_ready()
        self.rx.start_button.setEnabled(True)
        self.waterfall.set_mode(self.settings.mode)
        self._log(f"codec ready: {text}")
        if self._resume_rx_after_codec_reload:
            self._resume_rx_after_codec_reload = False
            self._log(f"restarting receive with {self.settings.mode}")
            self.rx.start()

    def _on_codec_thread_finished(self, thread: _CodecThread) -> None:
        if thread.config != _codec_config(self.settings):
            self._load_codec()

    def _on_model_failed(self, thread: _CodecThread, message: str) -> None:
        if thread.config != _codec_config(self.settings):
            self._log(f"discarding stale codec error: {message}")
            return
        self._last_codec_error = message
        self._resume_rx_after_codec_reload = False
        self.model_label.setText("No model — open Model Manager")
        self.tx.mode.setEnabled(True)
        self._log(message)
        if self._smoke_test:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("AETV model")
        box.setText(message)
        box.setInformativeText(
            "Use Model Manager to install a checksum-verified release model, "
            "or choose a local runtime model in Settings."
        )
        manage = box.addButton("Open Model Manager…", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is manage:
            self.open_model_manager()

    def _begin_startup_model_check(self) -> None:
        explicit_model = self.settings.checkpoint or os.environ.get(
            "AETV_RUNTIME_MODEL"
        ) or os.environ.get("AETV_CHECKPOINT")
        if explicit_model:
            # An explicit local selection is validated by the codec loader and
            # must never be replaced by a network download behind the scenes.
            self._load_codec()
            return
        self.model_label.setText("Checking installed models…")
        self.tx.send_button.setEnabled(False)
        self.rx.start_button.setEnabled(False)
        self._model_inventory_thread = ModelInventoryThread(self)
        self._model_inventory_thread.complete.connect(self._startup_model_check_complete)
        self._model_inventory_thread.start()

    def _startup_model_check_complete(self, statuses: dict, error_message: str) -> None:
        self._model_inventory_thread = None
        if error_message:
            self._log(f"model inventory failed: {error_message}")
        if any(status.installed for status in statuses.values()):
            self._activate_available_model(statuses)
            return
        self._show_first_run_model_manager(statuses)

    def _show_first_run_model_manager(self, statuses: dict) -> None:
        self.model_label.setText("No model installed")
        dialog = ModelManagerDialog(
            self.settings.mode,
            self,
            statuses=statuses,
            first_run=True,
        )
        if dialog.exec() == dialog.DialogCode.Accepted:
            self._activate_available_model(dialog.statuses)
        else:
            self.model_label.setText("No model — File > Model Manager…")
            self._log("model setup skipped; Send and Receive remain disabled")

    def open_model_manager(self) -> None:
        dialog = ModelManagerDialog(self.settings.mode, self)
        dialog.exec()
        self._activate_available_model(dialog.statuses)

    def _activate_available_model(self, statuses: dict) -> None:
        available = [
            mode
            for mode in RELEASE_MODES
            if statuses.get(mode) is not None and statuses[mode].installed
        ]
        if not available:
            if self.station.codec is None:
                self.model_label.setText("No model — File > Model Manager…")
            return
        previous_mode = self.settings.mode
        if self.settings.mode not in available:
            self.settings.mode = available[0]
            self._log(
                f"{previous_mode} has no installed model; using {self.settings.mode}"
            )
        codec_matches = (
            self.station.codec is not None
            and self.station.codec.mode.name == self.settings.mode
        )
        if not codec_matches or previous_mode != self.settings.mode:
            # A failed explicit selection must not mask a newly installed
            # release model, and checkpoints cannot be reused across modes.
            self.settings.checkpoint = ""
        save_settings(self.settings)
        self.station.settings = self.settings
        self.rx.sync_from_config()
        self.tx.sync_from_config()
        self.waterfall.set_mode(self.settings.mode)
        self._refresh_station_label()
        if not codec_matches:
            self._load_codec()

    def open_settings(self) -> None:
        if self.tx.transmitting():
            message = "settings cannot be changed during a transmission"
            self.tx.status.setText(message)
            self._log(message)
            return
        previous_codec_config = _codec_config(self.settings)
        previous_rx_config = _rx_runtime_config(self.settings)
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
        codec_config = _codec_config(self.settings)
        codec_changed = (
            self.station.codec is None
            or self.station.codec.mode.name != self.settings.mode
            or codec_config != previous_codec_config
        )
        rx_changed = _rx_runtime_config(self.settings) != previous_rx_config
        if self.rx.listening() and (codec_changed or rx_changed):
            if codec_changed:
                self._reload_codec_after_rx_stop = True
                self._resume_rx_after_codec_reload = True
                self._log("receive stopping to apply the new mode/checkpoint")
            else:
                self._restart_rx_after_settings_stop = True
                self._log("receive restarting to apply the new source settings")
            self.rx.stop()
        elif codec_changed:
            self._load_codec()

    def _on_rx_stopped_for_codec_reload(self) -> None:
        if not self._reload_codec_after_rx_stop:
            return
        self._reload_codec_after_rx_stop = False
        self._load_codec()

    def _on_rx_stopped_for_settings_restart(self) -> None:
        if not self._restart_rx_after_settings_stop:
            return
        self._restart_rx_after_settings_stop = False
        self.rx.start()

    def _on_tx_mode_requested(self, mode: str) -> None:
        if mode == self.settings.mode or self.tx.transmitting():
            return
        previous = self.settings.mode
        self.settings.mode = mode
        # A local runtime bundle belongs to one geometry and cannot be reused
        # after selecting the other release mode.
        self.settings.checkpoint = ""
        save_settings(self.settings)
        self.station.settings = self.settings
        self.waterfall.set_mode(mode)
        self._refresh_station_label()
        self._log(f"mode changed from {previous} to {mode}; loading model")
        if self.rx.listening():
            self._reload_codec_after_rx_stop = True
            self._resume_rx_after_codec_reload = True
            self.rx.stop()
        else:
            self._load_codec()

    def open_path_planner(self) -> None:
        from aetv.gui.path_planner import PathPlannerDialog

        if self._path_planner is None:
            self._path_planner = PathPlannerDialog(self.station, self.rx, self)
            self._path_planner.finished.connect(
                lambda _result: setattr(self, "_path_planner", None)
            )
        self._path_planner.show()
        self._path_planner.raise_()
        self._path_planner.activateWindow()

    def open_ft8_calibration(self) -> None:
        from aetv.gui.ft8_calibration import Ft8CalibrationDialog

        if self._ft8_calibration is None:
            self._ft8_calibration = Ft8CalibrationDialog(self.station, self)
            self._ft8_calibration.calibrationImported.connect(
                self._on_ft8_calibration_imported
            )
            self._ft8_calibration.finished.connect(
                lambda _result: setattr(self, "_ft8_calibration", None)
            )
        self._ft8_calibration.show()
        self._ft8_calibration.raise_()
        self._ft8_calibration.activateWindow()

    def _on_ft8_calibration_imported(self, count: int) -> None:
        self._log(f"FT8 propagation calibration updated with {count} observations")
        self.rx.refresh_propagation_calibration()
        if self._path_planner is not None:
            self._path_planner.refresh()

    def _refresh_station_label(self) -> None:
        backend = self.settings.cat_backend if not self.settings.audio_only else "audio-only"
        self.station_label.setText(f"  {self.settings.callsign}  {self.settings.mode}  ")
        if backend == "none":
            self.rig_label.setText("CAT off")
        elif backend == "flex":
            route = "VITA" if self.settings.flex_native_audio else "audio device"
            self.rig_label.setText(f"Flex {self.settings.flex_host or '?'} · {route}")
        elif backend == "hamlib":
            self.rig_label.setText(f"Hamlib {self.settings.hamlib_device or '?'}")
        else:
            self.rig_label.setText(f"{backend} {self.settings.serial_port}")

    def _on_tx_started(self) -> None:
        self._tx_finished_while_stopping_rx = False
        emulating = self.tx.emulating()
        self._emulation_active = emulating
        if emulating:
            key = self.tx.selected_channel_profile()
            self.rx.prepare_emulator(CHANNEL_PROFILES[key].label)
        if self.rx.listening() and (emulating or self.settings.rx_source != "kiwi"):
            self.rx.stop()
            self._resume_rx = True
            self._tx_waiting_for_rx = True
            self._log(
                "receive paused for channel loopback"
                if emulating else "receive paused for half-duplex transmit"
            )
        else:
            self._resume_rx = False
            self._tx_waiting_for_rx = False
            if self.rx.listening() and self.settings.rx_source == "kiwi":
                self._log("Kiwi receive remains live for full-duplex monitoring")
            self.tx.allow_transmit()

    def _on_rx_stopped_for_tx(self) -> None:
        if not self._tx_waiting_for_rx:
            return
        self._tx_waiting_for_rx = False
        self.tx.allow_transmit()
        if self._tx_finished_while_stopping_rx and self._resume_rx:
            self._resume_or_hold_receive()

    def _on_tx_finished(self) -> None:
        self.ptt_lamp.set_keyed(False)
        if getattr(self, "_resume_rx", False):
            if self._tx_waiting_for_rx:
                self._tx_finished_while_stopping_rx = True
                return
            self._resume_or_hold_receive()
        self._emulation_active = False

    def _resume_or_hold_receive(self) -> None:
        if self._emulation_active:
            self._log("receive remains paused so the loopback result stays visible")
        else:
            self.rx.start()
        self._resume_rx = False
        self._tx_finished_while_stopping_rx = False
        self._emulation_active = False

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append_line(f"{stamp}  {message}")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About AETV",
            "AETV — Autoencoder Television\n"
            "Live video built for challenging amateur-radio links.\n\n"
            "Standard channel: 192×108 @ 6 fps.\n"
            "Wide 8 kHz: 256×144 @ 12 fps.\n"
            "Identify every transmission. This software does not replace "
            "a license or a band-plan check.\n\n"
            "Artistic License 2.0",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._ft8_calibration is not None and self._ft8_calibration.busy():
            QMessageBox.warning(
                self,
                "AETV",
                "An FT8 calibration operation is active. Wait for it to finish before quitting.",
            )
            event.ignore()
            return
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
            self.rx.stop_blocking()
        self.tx.stop_preview()
        save_settings(self.settings)
        event.accept()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    smoke_test = "--smoke-test" in args
    if smoke_test:
        args.remove("--smoke-test")
    smoke_output = None
    if "--video-smoke-output" in args:
        index = args.index("--video-smoke-output")
        if index + 1 >= len(args):
            return 3
        smoke_output = Path(args[index + 1])
        del args[index : index + 2]
    if smoke_test:
        try:
            if smoke_output is not None:
                write_video_smoke_test(smoke_output)
            else:
                with tempfile.TemporaryDirectory(prefix="aetv-video-smoke-") as folder:
                    write_video_smoke_test(Path(folder) / "saved-video.mp4")
        except Exception as error:
            print(f"packaged video save smoke test failed: {error}", file=sys.stderr)
            return 3
    app = QApplication(args)
    app.setApplicationName("AETV")
    app.setApplicationDisplayName("AETV")
    app.setOrganizationName("AETV")
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    window = MainWindow(StationSettings() if smoke_test else None)
    window._smoke_test = smoke_test
    window.show()
    if smoke_test:
        result = {"code": 2, "finished": False}

        def finish(code: int) -> None:
            if result["finished"]:
                return
            result["finished"] = True
            result["code"] = code
            poll.stop()
            window.close()
            app.exit(code)

        def check_codec() -> None:
            # Startup now inventories checksum-valid models before constructing
            # a codec. Do not mistake that intentional first phase for failure.
            result_code = _smoke_codec_result(window)
            if result_code is not None:
                finish(result_code)

        poll = QTimer(window)
        poll.timeout.connect(check_codec)
        poll.start(100)
        QTimer.singleShot(120_000, lambda: finish(2))
        app.exec()
        return result["code"]
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
