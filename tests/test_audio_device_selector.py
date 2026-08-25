"""Audio-device GUI selection behavior."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox

from aetv.audio_io import DeviceInfo
from aetv.gui.settings_dialog import _populate_audio_combo


_APP = QApplication.instance() or QApplication([])


def _devices():
    return [
        DeviceInfo(0, "Radio DAX", 2, 0, False, "dax-id"),
        DeviceInfo(1, "USB microphone", 1, 0, True, "mic-id"),
    ]


def test_audio_combo_preserves_selected_endpoint_across_refresh():
    combo = QComboBox()
    _populate_audio_combo(combo, _devices(), "wasapi:mic-id")

    assert combo.currentText() == "1: USB microphone (default)"
    assert combo.currentData() == "wasapi:mic-id"


def test_audio_combo_migrates_legacy_friendly_name():
    combo = QComboBox()
    _populate_audio_combo(combo, _devices(), "Radio DAX")

    assert combo.currentData() == "wasapi:dax-id"
