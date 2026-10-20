"""Resize regression for issue #39; no models, devices or radio I/O."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from aetv.gui.app import MainWindow
from aetv.gui.clip_grid import ClipGrid
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.tx_panel import TransmitPanel
from aetv.settings import StationSettings


@pytest.fixture
def window_factory(monkeypatch):
    app = QApplication.instance() or QApplication([])
    original_font = app.font()
    original_style = app.style().objectName()
    # Isolate geometry from device discovery, camera capture, model downloads,
    # settings on disk and the startup model-manager dialog.
    for method in ("_start_preview", "_fill_cameras", "_fill_outputs", "_fill_screen_targets"):
        monkeypatch.setattr(TransmitPanel, method, lambda self: None)
    monkeypatch.setattr(ReceivePanel, "_fill_inputs", lambda self: None)
    monkeypatch.setattr(MainWindow, "_begin_startup_model_check", lambda self: None)
    monkeypatch.setattr("aetv.gui.app.save_settings", lambda settings: None)
    windows = []

    def make(*, font_size=9, style="Fusion", waveform_mode="video"):
        app.setStyle(style)
        font = QFont(original_font)
        # The offscreen plugin's generic Windows font can resolve to the
        # last-resort box font. Exercise installed native glyph metrics.
        if "Segoe UI" in QFontDatabase.families():
            font.setFamily("Segoe UI")
        font.setPointSize(font_size)
        app.setFont(font)
        window = MainWindow(StationSettings(
            kiwi_auto_select=False, waveform_mode=waveform_mode,
        ))
        windows.append(window)
        return app, window

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    app.setStyle(original_style)
    app.setFont(original_font)


def _settle(app):
    for _ in range(5):
        app.processEvents()


def _scroll_for(widget):
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, QScrollArea):
        parent = parent.parentWidget()
    assert parent is not None
    return parent


def _assert_reachable(app, widget, scroll):
    scroll.ensureWidgetVisible(widget, 0, 0)
    # Compound editors can forward focus to a smaller child. Scroll to the
    # widget's far corner as well, as an operator can with the scrollbars.
    corner = widget.mapTo(scroll.widget(), widget.rect().bottomRight())
    scroll.ensureVisible(corner.x() + 1, corner.y() + 1, 0, 0)
    _settle(app)
    bounds = QRect(widget.mapTo(scroll.viewport(), QPoint()), widget.size())
    assert scroll.viewport().rect().contains(bounds), (widget, bounds)


def _assert_visible_in_window(window, widget):
    assert widget.isVisibleTo(window)
    bounds = QRect(widget.mapTo(window, QPoint()), widget.size())
    assert window.rect().contains(bounds), (widget, bounds, window.size())


@pytest.mark.parametrize("style", ["Fusion", "Windows"])
@pytest.mark.parametrize("font_size", [9, 12, 18])
@pytest.mark.parametrize("waveform_mode", ["video", "analog_av"])
def test_controls_and_clip_cells_survive_resize(window_factory, style, font_size, waveform_mode):
    app, window = window_factory(font_size=font_size, style=style, waveform_mode=waveform_mode)
    preview = np.zeros((108, 192, 3), dtype=np.uint8)
    for cell in window.tx.clip_grid.cells:
        cell.set_path(f"clip-{cell.index}-" + "long descriptive filename " * 12 + ".mp4")
        cell.set_ready(preview)
    window.show()
    sizes = [(1366, 704), (1920, 1016), (2560, 1376), (3840, 2096), (1366, 704)]
    if font_size == 18:
        # Accessibility fonts use scrolling inside setup tabs. Test them on a
        # larger logical desktop separately from physical monitor DPI scaling.
        sizes = [(1920, 1016), (3840, 2096), (1920, 1016)]
    for width, height in sizes:
        window.resize(width, height)
        _settle(app)
        assert window.size().toTuple() == (width, height), {
            "font": app.font().toString(), "line_spacing": window.fontMetrics().lineSpacing(),
            "window_min": window.minimumSizeHint().toTuple(),
            "receive_min": window.rx.minimumSizeHint().toTuple(),
            "transmit_min": window.tx.minimumSizeHint().toTuple(),
        }
        for index in range(window.tx.controls.count()):
            window.tx.controls.setCurrentIndex(index)
            _settle(app)
            for control in (window.tx.send_button, window.tx.cancel_button,
                            window.tx.qrl_button, window.tx.mode,
                            window.rx.start_button, window.rx.stop_button,
                            window.rx.save_button, window.rx.open_saved_videos_button):
                _assert_visible_in_window(window, control)
                assert control.height() >= control.minimumSizeHint().height()
            for panel in (window.tx, window.rx):
                _assert_visible_in_window(window, panel.preview)
                assert panel.preview.height() >= panel.preview.minimumHeight()
            if window.tx.controls.tabText(index) == "Clips":
                for cell in window.tx.clip_grid.cells:
                    _assert_reachable(app, cell, _scroll_for(cell))
                    assert cell.rect().contains(cell.picture.geometry())
                    assert cell.rect().contains(cell.name.geometry())
                if font_size == 9:
                    assert _scroll_for(window.tx.clip_grid).verticalScrollBar().maximum() == 0
                    assert _scroll_for(window.tx.clip_grid).horizontalScrollBar().maximum() == 0
                    assert window.tx.clip_grid._columns == (8 if width >= 2560 else 4)
        assert window.waterfall.height() >= window.waterfall.minimumHeight()
        for control in (window.radio.backend, window.radio.frequency, window.radio.apply):
            _assert_reachable(app, control, _scroll_for(window.radio))


def test_initial_window_fits_available_desktop(window_factory, monkeypatch):
    available = QRect(0, 0, 1280, 680)
    monkeypatch.setattr(MainWindow, "screen", lambda self: SimpleNamespace(
        availableGeometry=lambda: available,
    ))
    app, window = window_factory()
    window.show()
    _settle(app)
    assert window.frameGeometry().width() <= available.width()
    assert window.frameGeometry().height() <= available.height()


def test_rf_details_and_source_changes_keep_actions_visible(window_factory):
    app, window = window_factory()
    window.resize(1366, 704)
    window.show()
    _settle(app)
    compact_height = window.radio.height()
    window.radio.details_button.click()
    window.settings.waveform_mode = "analog_av"
    window.tx.sync_from_config()
    window.rx.source.setCurrentIndex(window.rx.source.findData("kiwi"))
    _settle(app)
    assert window.radio.details.isVisible()
    assert window.radio.height() > compact_height
    for widget in (window.tx.send_button, window.tx.cancel_button,
                   window.rx.start_button, window.rx.stop_button,
                   window.radio.details_button):
        _assert_visible_in_window(window, widget)
    _assert_reachable(app, window.rx.kiwi_host, _scroll_for(window.rx.kiwi_host))
    window.radio.details_button.click()
    _settle(app)
    assert not window.radio.details.isVisible()
    assert window.radio.height() == compact_height


def test_source_tabs_show_only_the_active_device_row(window_factory):
    app, window = window_factory()
    window.show()
    for button, visible, hidden in (
        (window.tx.cam_radio, window.tx.camera, window.tx.screen_target),
        (window.tx.screen_radio, window.tx.screen_target, window.tx.file_button),
        (window.tx.file_radio, window.tx.file_button, window.tx.camera),
    ):
        button.setChecked(True)
        _settle(app)
        assert visible.isVisibleTo(window)
        assert not hidden.isVisibleTo(window)


def test_thumbnail_click_keeps_its_slot_after_reflow(window_factory):
    app, _window = window_factory()
    grid = ClipGrid()
    clicked = []
    grid.activated.connect(clicked.append)
    try:
        grid.show()
        for width in (1800, 680):
            grid.resize(width, 200)
            _settle(app)
            QTest.mouseClick(grid.cells[7].picture, Qt.MouseButton.LeftButton)
        assert clicked == [7, 7]
    finally:
        grid.close()
        grid.deleteLater()
