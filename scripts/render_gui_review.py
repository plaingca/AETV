#!/usr/bin/env python3
"""Render real Qt widgets for offline visual review, with hardware/model I/O stubbed.

Run from the repository root with the GUI extras installed. This does not test
radio connections, codecs, audio, camera capture, or RF performance.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AETV_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("gui-review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from PySide6.QtWidgets import QApplication
    import aetv.gui.app as app_module
    from aetv.gui.rx_panel import ReceivePanel
    from aetv.gui.settings_dialog import SettingsDialog
    from aetv.gui.tx_panel import TransmitPanel
    from aetv.settings import StationSettings
    from aetv.station import TxPhase, TxState

    app = QApplication.instance() or QApplication([])
    with (
        patch.object(app_module.MainWindow, "_begin_startup_model_check", lambda self: None),
        patch.object(app_module, "save_settings", lambda settings: None),
        patch.object(TransmitPanel, "_start_preview", lambda self: None),
        patch.object(TransmitPanel, "_fill_cameras", lambda self: None),
        patch.object(TransmitPanel, "_fill_outputs", lambda self: None),
        patch.object(ReceivePanel, "_fill_inputs", lambda self: None),
        patch.object(SettingsDialog, "_load_device_inventory", lambda self: None),
    ):
        window = app_module.MainWindow(StationSettings(receive_dir=str(args.output / "received")))
        window.model_label.setText("GUI review · model not loaded")
        window.tx.send_button.setEnabled(False)
        window.rx.start_button.setEnabled(False)
        window.tx.camera.addItem("Camera not enumerated in review", 0)
        window.tx.output.addItem("Audio not enumerated in review", "")
        window.rx.input_device.addItem("Audio not enumerated in review", "")
        window.show()

        def snapshot(name: str) -> None:
            app.processEvents()
            if not window.grab().save(str(args.output / name)):
                raise RuntimeError(f"Could not save {name}")

        snapshot("main.png")
        print(f"Main window: {window.width()} × {window.height()}")
        window.tx.channel_route.setCurrentIndex(window.tx.channel_route.findData("clean"))
        snapshot("local-test.png")
        window.tx.cancel_button.setEnabled(True)
        window.tx.mode.setEnabled(False)
        window.tx.channel_route.setEnabled(False)
        window.tx._apply_state(TxState(TxPhase.SENDING, .45, "Local test · simulated HF channel · no radio output"))
        snapshot("local-test-running.png")
        window.tx._apply_state(TxState(TxPhase.CANCELLED, .45, "Local test cancelled"))
        snapshot("local-test-cancelled.png")
        window.tx.channel_route.setCurrentIndex(window.tx.channel_route.findData("radio"))
        window.tx._apply_state(TxState(TxPhase.FAILED, .45,
            "PTT OFF failed; check the radio immediately", ptt_uncertain=True))
        snapshot("ptt-off-warning.png")
        from PySide6.QtWidgets import QMessageBox
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
            window.tx.confirm_radio_receive()
        window.tx._apply_state(TxState(TxPhase.IDLE, 0, "GUI review · model not loaded"))
        window.tx.channel_route.setCurrentIndex(window.tx.channel_route.findData("clean"))
        window.radio.backend.setCurrentIndex(window.radio.backend.findData("hackrf"))
        window.rx.source.setCurrentIndex(window.rx.source.findData("hackrf"))
        window.radio.advanced_toggle.click()
        window.tx.clips_toggle.click()
        snapshot("advanced.png")
        dialog = SettingsDialog(window.settings)
        dialog.show()
        app.processEvents()
        if not dialog.grab().save(str(args.output / "settings.png")):
            raise RuntimeError("Could not save settings.png")
        dialog.reject()
        window.close()


if __name__ == "__main__":
    main()
