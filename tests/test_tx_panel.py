"""Transmit-panel preview routing."""

from types import SimpleNamespace

import numpy as np

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
