"""Transmit-panel preview routing."""

from types import SimpleNamespace

import numpy as np

from aetv.gui.tx_panel import TransmitPanel


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
