"""Transmit-panel preview routing."""

from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QDialog

from aetv.config import AETV_MODES
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.tx_panel import TransmitPanel
from aetv.station import RxState


class _Preview:
    def __init__(self):
        self.shown = []

    def set_rgb(self, frames):
        self.shown.append(frames)


def _panel(*, emulating: bool):
    return SimpleNamespace(
        transmitting=lambda: True,
        cam_radio=SimpleNamespace(isChecked=lambda: True),
        emulating=lambda: emulating,
        preview=_Preview(),
    )


def test_webcam_loopback_keeps_source_preview_updating():
    panel = _panel(emulating=True)
    frames = np.zeros((12, 2, 3, 3), dtype=np.uint8)

    TransmitPanel._show_preview(panel, frames)

    assert panel.preview.shown == [frames]


def test_webcam_radio_transmit_still_suppresses_gop_preview():
    panel = _panel(emulating=False)
    frames = np.zeros((12, 2, 3, 3), dtype=np.uint8)

    TransmitPanel._show_preview(panel, frames)

    assert panel.preview.shown == []


def test_mode_selection_requests_reload_before_transmit():
    requested = []
    button = SimpleNamespace(setEnabled=lambda enabled: setattr(button, "enabled", enabled))
    status = SimpleNamespace(setText=lambda text: setattr(status, "text", text))
    panel = SimpleNamespace(
        station=SimpleNamespace(settings=SimpleNamespace(mode="V8")),
        mode=SimpleNamespace(currentData=lambda: "V7"),
        send_button=button,
        status=status,
        modeRequested=SimpleNamespace(emit=requested.append),
        _restart_preview=lambda _index: None,
    )

    TransmitPanel._on_mode_changed(panel, 1)

    assert panel.station.settings.mode == "V8"
    assert requested == ["V7"]
    assert not button.enabled
    assert status.text == "loading V7 model…"


def test_av_clip_editor_uses_v8_codec_mode(monkeypatch):
    opened = {}

    class Dialog:
        def __init__(self, path, mode, **_kwargs):
            opened["path"] = path
            opened["mode"] = mode

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr("aetv.gui.tx_panel.ClipEditorDialog", Dialog)
    panel = SimpleNamespace(
        mode=SimpleNamespace(currentData=lambda: "V8_AV"),
        station=SimpleNamespace(settings=SimpleNamespace(mode="V8")),
        status=SimpleNamespace(setText=lambda text: setattr(panel.status, "text", text)),
        gops=SimpleNamespace(value=lambda: 10),
        _selected_mode_name=lambda: "V8",
    )

    TransmitPanel._open_clip_editor(panel, 0, "program.mp4", None)

    assert opened == {"path": "program.mp4", "mode": AETV_MODES["V8"]}


def test_av_clip_preparation_uses_loaded_v8_codec(monkeypatch):
    started = []

    class Thread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            started.append(self.args)

    monkeypatch.setattr("aetv.gui.tx_panel.threading.Thread", Thread)
    cell = SimpleNamespace(set_progress=lambda value: setattr(cell, "progress", value))
    edit = SimpleNamespace()
    panel = SimpleNamespace(
        _clip_generation=0,
        _selected_clip=3,
        _prepared_clips={3: object()},
        _clip_edits={3: edit},
        clip_grid=SimpleNamespace(cells=[None, None, None, cell]),
        station=SimpleNamespace(codec=SimpleNamespace(mode=SimpleNamespace(name="V8"))),
        _selected_mode_name=lambda: "V8",
        _prepare_clip_batch=lambda *_args: None,
    )

    TransmitPanel._queue_clip_preparation(panel)

    assert cell.progress == 0.0
    assert started == [([(3, edit)], "V8", 1)]


def test_ten_gop_loopback_reaches_full_progress_and_is_recorded():
    class Preview:
        def enqueue_rgb(self, *_args, **_kwargs):
            pass

    class Progress:
        def setValue(self, value):
            self.value = value

    panel = SimpleNamespace(
        station=SimpleNamespace(
            settings=SimpleNamespace(gops=10),
            require_codec=lambda: SimpleNamespace(
                mode=SimpleNamespace(fps=6, gop_frames=2)
            ),
        ),
        preview=Preview(),
        status=SimpleNamespace(setText=lambda _text: None),
        statusChanged=SimpleNamespace(emit=lambda _text: None),
        progress=Progress(),
        _emulated_video=None,
    )
    first = np.zeros((2, 2, 3, 3), dtype=np.uint8)
    last = np.ones((2, 2, 3, 3), dtype=np.uint8)

    ReceivePanel.show_emulated(panel, first, RxState(gops=1, message="1/10"))
    assert panel.progress.value == 10
    ReceivePanel.show_emulated(panel, last, RxState(gops=10, message="10/10"))

    assert panel.progress.value == 100
    assert np.array_equal(panel._emulated_video[:2], first)
    assert np.array_equal(panel._emulated_video[2:], last)
