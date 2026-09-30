"""All GUI log paths must reach Qt's text layout on the GUI thread."""

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import QApplication

import aetv.gui.app as app_module
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.tx_panel import TransmitPanel
from aetv.gui.widgets import LogPane
from aetv.settings import StationSettings


_APP = QApplication.instance() or QApplication([])


class RecordingLogPane(LogPane):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.writes = []

    @Slot(str)
    def appendPlainText(self, text):
        on_gui_thread = QThread.currentThread() == self.thread()
        self.writes.append((text, on_gui_thread))
        # Fail the assertion instead of intentionally corrupting Qt on regression.
        if on_gui_thread:
            super().appendPlainText(text)


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "LogPane", RecordingLogPane)
    monkeypatch.setattr(app_module.MainWindow, "_begin_startup_model_check", lambda self: None)
    for method in ("_start_preview", "_fill_cameras", "_fill_outputs"):
        monkeypatch.setattr(TransmitPanel, method, lambda self: None)
    monkeypatch.setattr(ReceivePanel, "_fill_inputs", lambda self: None)
    monkeypatch.setattr(app_module, "save_settings", lambda settings: None)
    window = app_module.MainWindow(StationSettings(receive_dir=str(tmp_path)))
    _APP.processEvents()
    window.log.clear()
    window.log.writes.clear()
    yield window
    window.close()
    window.deleteLater()
    _APP.processEvents()


@pytest.mark.parametrize("source", ["station", "receive", "transmit", "widget"])
def test_worker_logs_are_queued_until_gui_thread_handles_them(window, source):
    callback = {
        "station": window.station.log,
        "receive": window.rx.logMessage.emit,
        "transmit": window.tx.logMessage.emit,
        "widget": window.log.append_line,
    }[source]
    messages = [f"{source} cache/save diagnostic {index}" for index in range(20)]
    errors = []

    def worker():
        try:
            for message in messages:
                callback(message)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert window.log.writes == []
    # Panel signals may first deliver to MainWindow, then queue the widget write.
    for _ in range(3):
        _APP.processEvents()
    assert len(window.log.writes) == len(messages)
    assert all(on_gui_thread for _, on_gui_thread in window.log.writes)
    assert all(text.endswith(expected) for (text, _), expected in zip(window.log.writes, messages))
    assert window.log.toPlainText().splitlines()[-1].endswith(messages[-1])


@pytest.mark.parametrize("backend", ["hamlib", "flex"])
def test_apply_hf_frequency_restarts_receive_and_logs_hf_dial(window, monkeypatch, backend):
    window.settings.cat_backend = backend
    if backend == "flex":
        window.settings.flex_host = "192.0.2.1"
        window.settings.flex_native_audio = True
        window.settings.rx_source = "flex"
        window.rx.sync_from_config()
    window.settings.hamlib_model = 1
    stopped = []
    monkeypatch.setattr(window.rx, "listening", lambda: True)
    monkeypatch.setattr(window.rx, "stop", lambda: stopped.append(True))
    window._apply_rf_settings({"freq_mhz": 14.230123})
    _APP.processEvents()
    assert window.settings.freq_mhz == 14.230123
    assert window.settings.sdr_frequency_mhz == 439.0
    assert window.radio.frequency.value() == 14.230123
    assert stopped == [True]
    assert window._restart_rx_after_settings_stop
    assert "14.230123 MHz" in window.log.toPlainText()
    restarted = []
    monkeypatch.setattr(window.rx, "start", lambda: restarted.append(window.settings.freq_mhz))
    window._on_rx_stopped_for_settings_restart()
    assert restarted == [14.230123]
    assert not window._restart_rx_after_settings_stop


def test_apply_frequency_is_ignored_during_transmit(window, monkeypatch):
    monkeypatch.setattr(window.tx, "transmitting", lambda: True)
    original = window.settings.freq_mhz
    window._apply_rf_settings({"freq_mhz": 14.230123})
    assert window.settings.freq_mhz == original
    monkeypatch.setattr(window.tx, "transmitting", lambda: False)
