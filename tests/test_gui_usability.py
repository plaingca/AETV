"""Operator-focused GUI checks without radio, camera, audio, or network access."""

import os
import sys
from pathlib import Path
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


@pytest.fixture(scope="session", autouse=True)
def native_font():
    from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QRawFont

    font = QFont(_APP.font())
    if sys.platform == "win32" and _APP.platformName() == "offscreen":
        # Qt's Windows offscreen plugin uses QFreeTypeFontDatabase, which
        # searches Qt's own fonts directory rather than the Windows registry.
        # Load an existing native UI font; no downloads or system changes.
        system_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        assert system_root, "Windows font discovery needs WINDIR or SystemRoot"
        directory = Path(system_root) / "Fonts"
        loaded_family = None
        for name in ("segoeui.ttf", "arial.ttf", "tahoma.ttf"):
            path = directory / name
            if not path.is_file():
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if font_id >= 0 and families:
                loaded_family = families[0]
                break
        assert loaded_family, f"No native UI font could be loaded from {directory}"
        font.setFamily(loaded_family)
        # Supply the native monospace face for the frequency field as well.
        monospace = directory / "consola.ttf"
        if monospace.is_file():
            assert QFontDatabase.addApplicationFont(str(monospace)) >= 0
        _APP.setFont(font)
    info = QFontInfo(font)
    assert info.family() and info.pixelSize() > 0, (
        f"No resolved application font: {font.toString()}; "
        f"available={QFontDatabase.families()}"
    )
    assert QRawFont.fromFont(font).isValid(), "Layout checks need a real font engine"
    return font


@pytest.fixture
def window(monkeypatch, tmp_path, native_font):
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


@pytest.mark.parametrize("style", ["Windows", "Fusion"])
@pytest.mark.parametrize("mode", ["V8", "V8_AV", "AC16_AV"])
def test_transmit_form_fits_viewport_with_long_source_names(window, native_font, style, mode):
    from PySide6.QtGui import QFont, QRawFont
    from PySide6.QtWidgets import QComboBox, QPushButton, QStyleFactory

    original_style = _APP.style().objectName()
    original_font = QFont(_APP.font())
    try:
        _APP.setStyle(QStyleFactory.create(style))
        _APP.setFont(QFont(native_font))
        assert QRawFont.fromFont(_APP.font()).isValid()
        tx = window.tx
        long_name = "A device or application with a long descriptive name " * 12
        for selector in (tx.camera, tx.screen_target, tx.output, tx.microphone):
            selector.addItem(long_name)
        # This is a layout test: switching the actual station mode can open
        # a model-install dialog on clean CI machines without local weights.
        tx.mode.blockSignals(True)
        tx.mode.setCurrentIndex(tx.mode.findData(mode))
        tx.mode.blockSignals(False)
        tx._sync_av_controls()
        _APP.processEvents()
        window.resize(1280, 800)
        _APP.processEvents()
        scroll = next(area for area in window.findChildren(QScrollArea) if area.widget() is tx)
        viewport = scroll.viewport()
        assert window.size().toTuple() == (1280, 800)
        assert tx.width() <= viewport.width()
        assert scroll.horizontalScrollBar().maximum() == 0
        controls = [tx.gops, tx.level_db, *tx._strip.findChildren(QComboBox),
                    *tx._strip.findChildren(QPushButton)]
        for control in controls:
            if control.isHidden():
                continue
            # Vertical scrolling remains available; each row must fit across
            # the viewport without hiding a setting behind horizontal scroll.
            point = control.mapTo(viewport, QPoint(0, 0))
            assert 0 <= point.x() <= viewport.width() - control.width()
            assert control.width() >= control.minimumSizeHint().width()
        window.ptt_lamp.set_unknown()
        _APP.processEvents()
        for control in (tx.cancel_button, window.rx.stop_button, window.ptt_lamp):
            rect = control.rect().translated(control.mapTo(window, QPoint(0, 0)))
            assert window.rect().contains(rect)
            assert control.height() >= control.minimumSizeHint().height()
    finally:
        _APP.setFont(original_font)
        _APP.setStyle(QStyleFactory.create(original_style))
        _APP.processEvents()


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


@pytest.mark.parametrize("style", ["Windows", "Fusion"])
@pytest.mark.parametrize("font_units,font_size", [("points", 18), ("pixels", 48)])
def test_laptop_layout_fits_large_native_metrics_with_all_actions_visible(window, native_font, style, font_units, font_size):
    from PySide6.QtGui import QFont, QFontInfo, QRawFont
    from PySide6.QtWidgets import QStyleFactory, QWidget

    original_style = _APP.style().objectName()
    original_font = QFont(_APP.font())
    try:
        native_style = QStyleFactory.create(style)
        assert native_style is not None
        _APP.setStyle(native_style)
        # Exposes the single-row minimum-width bug on Linux too; does not
        # reduce Windows font sizes or loosen the 1280 x 800 requirement.
        font = QFont(native_font)
        if font_units == "points":
            font.setPointSize(font_size)
        else:
            # Exercise much larger physical glyphs independently of Linux vs
            # Windows point-size/DPI conversion and installed-font fallback.
            font.setPixelSize(font_size)
        resolved = QFontInfo(font)
        assert resolved.family() and resolved.pixelSize() > 0, font.toString()
        assert QRawFont.fromFont(font).isValid(), font.toString()
        if font_units == "pixels":
            assert resolved.pixelSize() == font_size
        else:
            assert resolved.pointSizeF() == pytest.approx(font_size)
        _APP.setFont(font)
        # Exercise the widest safety warning, not only the shorter idle lamp.
        window.ptt_lamp.set_unknown()
        _APP.processEvents()
        window.resize(1280, 800)
        _APP.processEvents()

        def describe(widget):
            info = QFontInfo(widget.font())
            label = widget.objectName() or type(widget).__name__
            text = getattr(widget, "text", lambda: "")()
            return (f"{label} {text!r}: size={widget.size().toTuple()} "
                    f"minimum={widget.minimumSize().toTuple()} "
                    f"hint={widget.minimumSizeHint().toTuple()} "
                    f"font={info.family()!r}/{info.pixelSize()}px")

        roots = [window.centralWidget(), window.radio, window.tx,
                 window.tx.action_strip(), window.rx, window.rx.action_strip(),
                 window.statusBar()]
        widest = sorted(window.findChildren(QWidget),
                        key=lambda item: item.minimumSizeHint().width(), reverse=True)[:12]
        screen = _APP.primaryScreen()
        diagnostics = (f"style={_APP.style().objectName()} font={font.toString()} "
                       f"DPI={screen.logicalDotsPerInch()} DPR={screen.devicePixelRatio()}\n"
                       + "\n".join(describe(item) for item in roots + widest))
        assert window.minimumSizeHint().width() <= 1280, diagnostics
        assert window.width() == 1280, diagnostics
        assert window.height() == 800, diagnostics
        controls = (
            window.radio.apply, window.radio.advanced_toggle,
            window.radio.frequency_label, window.radio.frequency, window.radio.backend,
            window.ptt_lamp,
            window.tx.send_button, window.tx.cancel_button, window.tx.qrl_button,
            window.rx.start_button, window.rx.stop_button,
            window.rx.save_button, window.rx.open_saved_videos_button,
        )
        rectangles = []
        for control in controls:
            point = control.mapTo(window, QPoint(0, 0))
            rect = control.rect().translated(point)
            assert window.rect().contains(rect), diagnostics + "\n" + describe(control)
            assert control.width() >= control.minimumSizeHint().width(), diagnostics + "\n" + describe(control)
            assert control.height() >= control.minimumSizeHint().height(), diagnostics + "\n" + describe(control)
            assert not any(rect.intersects(other) for other in rectangles), diagnostics + "\n" + describe(control)
            rectangles.append(rect)
        for label in (window.station_label, window.rig_label, window.model_label):
            assert label.height() >= label.minimumSizeHint().height(), diagnostics + "\n" + describe(label)
        assert window.statusBar().height() >= window.statusBar().minimumSizeHint().height(), diagnostics
    finally:
        _APP.setFont(original_font)
        _APP.setStyle(QStyleFactory.create(original_style))
        _APP.processEvents()


def test_status_fields_share_space_and_keep_full_text_without_expanding_window(window):
    for label in (window.station_label, window.rig_label, window.model_label):
        message = "Connection status with a long device name and full recovery instructions " * 4
        label.setText(message)
        _APP.processEvents()
        assert label.width() > 0
        assert label.text()
        assert label.toolTip() == message
    window.resize(1280, 800)
    _APP.processEvents()
    assert window.width() == 1280
    assert window.height() == 800
