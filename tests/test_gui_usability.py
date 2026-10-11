"""Operator-focused GUI checks without radio, camera, audio, or network access."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QScrollArea

import aetv.gui.app as app_module
from aetv.gui.radio_panel import RadioPanel
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.tx_panel import TransmitPanel
from aetv.gui.widgets import ElidingLabel
from aetv.settings import StationSettings
from aetv.station import TxPhase, TxState


_APP = QApplication.instance() or QApplication([])


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.MainWindow, "_begin_startup_model_check", lambda self: None)
    monkeypatch.setattr(app_module, "save_settings", lambda settings: None)
    for method in ("_start_preview", "_fill_cameras", "_fill_outputs"):
        monkeypatch.setattr(TransmitPanel, method, lambda self: None)
    monkeypatch.setattr(ReceivePanel, "_fill_inputs", lambda self: None)
    window = app_module.MainWindow(StationSettings(receive_dir=str(tmp_path)))
    window.show()
    _APP.processEvents()
    yield window
    window.close()
    window.deleteLater()
    _APP.processEvents()


def test_default_window_fits_laptop_and_transmit_is_visible(window):
    assert window.width() == 1280
    assert window.height() == 800
    assert window._log_dock.isHidden()
    assert window.tx.clip_grid.isHidden()
    assert window.radio.advanced.isHidden()
    tx = window.tx
    scroll = next(area for area in window.findChildren(QScrollArea) if area.widget() is tx)
    assert not scroll.isAncestorOf(tx.send_button)
    point = tx.send_button.mapTo(window, QPoint(0, 0))
    assert point.y() >= 0
    assert point.y() + tx.send_button.height() < window.height()


def test_optional_controls_expand_without_forcing_window_offscreen(window):
    window.tx.clips_toggle.click()
    window.radio.advanced_toggle.click()
    window.radio.backend.setCurrentIndex(window.radio.backend.findData("hackrf"))
    _APP.processEvents()
    assert not window.tx.clip_grid.isHidden()
    assert not window.radio.advanced.isHidden()
    assert window.height() == 800
    assert any(area.verticalScrollBar().maximum() > 0 for area in window.findChildren(QScrollArea))
    for control in (window.tx.cancel_button, window.rx.stop_button):
        point = control.mapTo(window, QPoint(0, 0))
        assert 0 <= point.y() < window.height() - control.height()


def test_clip_selection_never_transmits(window, monkeypatch):
    panel = window.tx
    calls = []
    monkeypatch.setattr(panel, "send", lambda: calls.append("transmit"))
    panel.clip_grid.cells[0].set_path("prepared.mp4")
    panel._prepared_clips[0] = SimpleNamespace(
        path="prepared.mp4", preview_frames=np.zeros((1, 4, 4, 3), dtype=np.uint8)
    )
    panel._activate_clip(0)
    assert panel._selected_clip == 0
    assert panel.file_radio.isChecked()
    assert calls == []
    assert "press Transmit" in panel.status.toolTip()


def test_route_is_explicit_and_recovers_after_failure(window):
    panel = window.tx
    assert panel.send_button.text() == "Transmit"
    assert panel.qrl_button.text() == "Transmit QRL?"
    panel.channel_route.setCurrentIndex(panel.channel_route.findData("clean"))
    assert panel.emulating()
    assert panel.send_button.text() == "Run local test"
    assert not panel.qrl_button.isEnabled()
    assert not panel.output.isEnabled()
    assert "no radio output" in panel.route_hint.text()
    panel.channel_route.setEnabled(False)
    panel._apply_state(TxState(TxPhase.FAILED, 0.0, "Test failure"))
    assert panel.channel_route.isEnabled()
    panel.channel_route.setCurrentIndex(0)
    assert panel.qrl_button.isEnabled()
    assert panel.output.isEnabled()
    assert panel.send_button.text() == "Transmit"


@pytest.mark.parametrize("mode,filter_text", [
    ("V8", "0.35–2.75 kHz"), ("V9", "0.9–5.05 kHz"),
    ("V7", "0.5–9.5 kHz"), ("AC16", "0.25–15.8 kHz"),
    ("V8_AV", "0–5 kHz"), ("AC16_AV", "0–20 kHz"),
])
def test_mode_displays_bandwidth_and_filter_without_loading_models(window, mode, filter_text):
    panel = window.tx
    panel.mode.blockSignals(True)
    panel.mode.setCurrentIndex(panel.mode.findData(mode))
    panel._sync_av_controls()
    assert "kHz" in panel.mode.currentText()
    assert filter_text in panel.mode_hint.text()


def test_radio_only_shows_relevant_details_and_manual_correction():
    radio = RadioPanel(StationSettings())
    assert radio.uri.isHidden()
    assert radio.tx_gain.isHidden()
    assert radio.rx_gain.isHidden()
    radio.set_receive_source("rtlsdr")
    assert not radio.serial.isHidden()
    assert not radio.rx_gain.isHidden()
    radio.auto_correct.setChecked(True)
    assert not radio.correction.isEnabled()
    radio.auto_correct.setChecked(False)
    assert radio.correction.isEnabled()
    radio.backend.setCurrentIndex(radio.backend.findData("pluto"))
    assert not radio.uri.isHidden()
    assert not radio.tx_gain.isHidden()
    radio.deleteLater()


def test_truncated_status_preserves_full_recovery_message():
    label = ElidingLabel()
    message = "Cannot connect to the receiver. Check the address and press Start receiving to retry."
    label.resize(60, 20)
    label.setText(message)
    assert label.toolTip() == message
    label.deleteLater()


def test_idle_stop_does_not_leave_cancelling_status(window):
    panel = window.tx
    panel.status.setText("ready")
    panel.cancel()
    assert panel.status.text() == "ready"


def test_terminal_state_waits_for_worker_cleanup(window):
    panel = window.tx
    panel._thread = SimpleNamespace(is_alive=lambda: True)
    panel.send_button.setEnabled(False)
    panel.channel_route.setEnabled(False)
    finished = []
    panel.transmitFinished.connect(lambda: finished.append(True))
    panel._apply_state(TxState(TxPhase.CANCELLED, 0.1, "cancelled"))
    assert not finished
    assert not panel.channel_route.isEnabled()
    assert "cleanup" in panel.status.text()
    panel._thread = None
    panel._on_worker_finished()
    assert finished == [True]
    assert panel.channel_route.isEnabled()
    assert panel.status.text() == "cancelled"


def test_failed_unkey_keeps_warning_and_does_not_resume_receive(window, monkeypatch):
    started = []
    monkeypatch.setattr(window.rx, "start", lambda: started.append(True))
    window._resume_rx = True
    window.tx._apply_state(TxState(TxPhase.FAILED, 0.1, "PTT OFF failed; check the radio immediately", ptt_uncertain=True))
    assert "UNKNOWN" in window.ptt_lamp.text()
    assert "check" in window.tx.status.text()
    assert started == []
    assert window._resume_rx is False


def test_local_emulator_never_lights_radio_ptt(window):
    panel = window.tx
    panel.channel_route.setCurrentIndex(panel.channel_route.findData("clean"))
    panel._apply_state(TxState(TxPhase.SENDING, .1, "local test"))
    assert window.ptt_lamp.text() == "PTT idle"


def test_repeated_cancel_before_start_and_unexpected_failure_recover_controls(window):
    panel = window.tx
    # Hold half-duplex authorization until cancellation. No external I/O.
    panel.transmitStarted.disconnect()
    for _ in range(3):
        calls = []
        panel._begin_transmit(lambda: calls.append(True), "preparing")
        panel.cancel()
        worker = panel._thread
        if worker is not None:
            worker.join(1)
        _APP.processEvents()
        assert not panel.transmitting()
        assert not calls
        assert panel.channel_route.isEnabled()
        assert not panel.cancel_button.isEnabled()
        assert panel.status.text() == "cancelled"

    def failed_job():
        raise RuntimeError("device disappeared")

    panel._begin_transmit(failed_job, "preparing")
    panel.allow_transmit()
    worker = panel._thread
    if worker is not None:
        worker.join(1)
    _APP.processEvents()
    assert panel.channel_route.isEnabled()
    assert "device disappeared" in panel.status.text()


def test_quit_waits_for_transmit_to_finish(window, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    panel = window.tx
    panel._thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    event = QCloseEvent()
    try:
        window.closeEvent(event)
        assert not event.isAccepted()
        assert "still stopping" in warnings[0]
    finally:
        panel._thread = None


@pytest.mark.parametrize("action", ["send", "send_qrl"])
def test_unconfirmed_off_blocks_new_transmit_until_operator_checks_radio(window, monkeypatch, action):
    from PySide6.QtWidgets import QMessageBox

    panel = window.tx
    panel._apply_state(TxState(TxPhase.FAILED, .2, "PTT OFF failed", ptt_uncertain=True))
    attempts, questions = [], []
    monkeypatch.setattr(panel, "_apply_panel_settings", lambda: attempts.append("prepared"))

    def reject(*args):
        questions.append(args[2])
        assert args[-1] == QMessageBox.StandardButton.Cancel
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "warning", reject)
    getattr(panel, action)()
    assert questions and "radio is now in receive" in questions[0]
    assert attempts == []
    assert panel.ptt_uncertain
    assert "UNKNOWN" in window.ptt_lamp.text()
    assert not panel.transmitting()

    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Yes)
    # Stop after validation, proving preparation only follows acknowledgment.
    monkeypatch.setattr(panel.station.settings, "validate", lambda **kwargs: ["test preflight stop"])
    getattr(panel, action)()
    assert attempts == ["prepared"]
    assert not panel.ptt_uncertain
    assert window.ptt_lamp.text() == "PTT idle"
    assert not panel.transmitting()


def test_unrelated_states_cannot_clear_unconfirmed_off(window):
    panel = window.tx
    panel._apply_state(TxState(TxPhase.FAILED, .2, "PTT OFF failed", ptt_uncertain=True))
    for phase in (TxPhase.IDLE, TxPhase.CANCELLED, TxPhase.FAILED, TxPhase.DONE):
        panel._apply_state(TxState(phase, 0, "unrelated update"))
        assert panel.ptt_uncertain
        assert "UNKNOWN" in window.ptt_lamp.text()
        assert "PTT OFF failed" in panel.status.text()


def test_manual_receive_and_tuning_preserve_unconfirmed_off_on_cancel(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel = window.tx
    panel._apply_state(TxState(TxPhase.FAILED, .2, "PTT OFF failed", ptt_uncertain=True))
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(window.rx.engine, "start", lambda: calls.append("receive"))
    frequency = window.settings.freq_mhz
    assert window.rx.start() is False
    window._apply_rf_settings({"freq_mhz": 14.230})
    assert window.settings.freq_mhz == frequency
    assert not calls
    assert panel.ptt_uncertain
    assert "UNKNOWN" in window.ptt_lamp.text()
