"""Capture the real Qt layout with deterministic, explicitly synthetic previews.

Run under a desktop or Xvfb at the desired physical resolution, for example:
  xvfb-run -a -s '-screen 0 1366x768x24 -dpi 96' env QT_SCALE_FACTOR=1 \
    python scripts/capture_gui_layout.py --output /tmp/laptop.png --page Clips

Device discovery/capture, model startup and settings writes are disabled. The
screenshots demonstrate layout only, not decoding, RF or native Windows behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from aetv.gui.app import MainWindow
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.tx_panel import TransmitPanel
from aetv.settings import StationSettings


def sample_frame(index: int = 0) -> np.ndarray:
    """A test card with circles and a grid that make stretching easy to see."""
    image = Image.new("RGB", (512, 288), "#182635")
    draw = ImageDraw.Draw(image)
    colors = ["#d9d9d9", "#d9c942", "#52bbce", "#59ad77", "#c968b7", "#d77753", "#5b78bd"]
    colors = colors[index % len(colors):] + colors[:index % len(colors)]
    for column, color in enumerate(colors):
        draw.rectangle((column * 512 // 7, 0, (column + 1) * 512 // 7, 170), fill=color)
    for x in range(0, 512, 32):
        draw.line((x, 170, x, 288), fill="#405266")
    for y in range(170, 288, 24):
        draw.line((0, y, 512, y), fill="#405266")
    draw.ellipse((198, 80, 314, 196), fill="#182635", outline="white", width=3)
    draw.line((198, 138, 314, 138), fill="white", width=1)
    draw.line((256, 80, 256, 196), fill="white", width=1)
    draw.text((18, 210), f"AETV  /  SAMPLE {index + 1:02d}", fill="white", font_size=24)
    draw.text((18, 250), "LAYOUT TEST CARD - NO RF", fill="#b8cad8", font_size=16)
    return np.asarray(image).copy()


def bounds_in(window, widget):
    return QRect(widget.mapTo(window, QPoint()), widget.size())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page", choices=["Source", "Clips", "Audio", "Route"], default="Clips")
    parser.add_argument("--av", action="store_true")
    parser.add_argument("--rf-details", action="store_true")
    args = parser.parse_args()
    app = QApplication([])
    app.setStyle("Fusion")
    font = QFont(app.font())
    font.setPointSize(9)
    app.setFont(font)
    app.setApplicationName("AETV layout validation")
    with (
        patch.object(MainWindow, "_begin_startup_model_check"),
        patch.object(TransmitPanel, "_start_preview"),
        patch.object(TransmitPanel, "_fill_cameras"),
        patch.object(TransmitPanel, "_fill_outputs"),
        patch.object(TransmitPanel, "_fill_screen_targets"),
        patch.object(ReceivePanel, "_fill_inputs"),
        patch("aetv.gui.app.save_settings"),
    ):
        window = MainWindow(StationSettings(kiwi_auto_select=False,
                            waveform_mode="analog_av" if args.av else "video"))
        window.setWindowTitle("AETV — layout preview (generated samples; devices idle)")
        window.model_label.setText("Layout preview · RF/audio idle")
        window.tx.camera.addItem("Sample camera (not opened)", 0)
        window.tx.output.addItem("System default (not opened)", "")
        window.tx.microphone.addItem("System default (not opened)", "")
        window.rx.input_device.addItem("System default (not opened)", "")
        window.rx.playback_device.addItem("System default (not opened)", "")
        window.rx.preview.set_rgb(sample_frame())
        window.tx.preview.set_rgb(sample_frame())
        window.rx.status.setText("Stopped · sample preview")
        window.tx.status.setText("Ready · sample preview")
        for index, cell in enumerate(window.tx.clip_grid.cells):
            cell.set_path(f"Sample clip {index + 1:02d}.mp4")
            cell.set_ready(sample_frame(index))
        window.tx.controls.setCurrentIndex([
            window.tx.controls.tabText(i) for i in range(window.tx.controls.count())
        ].index(args.page))
        if args.rf_details:
            window.radio.details_button.setChecked(True)
        window.log.clear()
        window.log.append_line("Layout fixture: generated test cards only; no model, camera, audio or RF started.")
        screen = window.screen()
        available = screen.availableGeometry()
        # Reserve 64 logical pixels for title bar/taskbar rather than hiding
        # overflow by capturing an application taller than the target desktop.
        window.resize(available.width(), available.height() - 64)
        window.move(available.topLeft())
        window.show()
        for _ in range(8):
            app.processEvents()
        QTest.qWait(100)
        controls = {
            "send": window.tx.send_button, "cancel": window.tx.cancel_button,
            "qrl": window.tx.qrl_button, "mode": window.tx.mode,
            "start_receive": window.rx.start_button, "stop_receive": window.rx.stop_button,
            "save_video": window.rx.save_button, "open_folder": window.rx.open_saved_videos_button,
            "receive_preview": window.rx.preview, "transmit_preview": window.tx.preview,
            "ptt": window.ptt_lamp, "rf_details": window.radio.details_button,
        }
        checks = {}
        for name, widget in controls.items():
            bounds = bounds_in(window, widget)
            checks[name] = {"visible": widget.isVisibleTo(window) and window.rect().contains(bounds),
                            "rect": [bounds.x(), bounds.y(), bounds.width(), bounds.height()]}
        assert all(item["visible"] for item in checks.values()), checks
        visible_cells = 0
        if args.page == "Clips":
            for cell in window.tx.clip_grid.cells:
                parent = cell.parentWidget()
                while not isinstance(parent, QScrollArea):
                    parent = parent.parentWidget()
                bounds = QRect(cell.mapTo(parent.viewport(), QPoint()), cell.size())
                visible_cells += int(cell.isVisibleTo(window) and parent.viewport().rect().contains(bounds))
            assert visible_cells == 8, visible_cells
        args.output.parent.mkdir(parents=True, exist_ok=True)
        screenshot = screen.grabWindow(window.winId())
        assert not screenshot.isNull()
        assert screenshot.save(str(args.output))
        metadata = {
            "physical_screen": [round(available.width() * screen.devicePixelRatio()),
                                round(available.height() * screen.devicePixelRatio())],
            "device_pixel_ratio": screen.devicePixelRatio(),
            "logical_window": list(window.size().toTuple()),
            "screenshot_pixels": [screenshot.width(), screenshot.height()],
            "reserved_desktop_logical_pixels": 64,
            "platform": app.platformName(), "style": app.style().objectName(),
            "font_point_size": app.font().pointSize(), "page": args.page,
            "waveform_mode": window.settings.waveform_mode,
            "rf_details": args.rf_details, "clip_columns": window.tx.clip_grid._columns,
            "visible_clip_cells": visible_cells, "controls": checks,
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "scope": "Actual Qt/X11 window capture with generated test cards; no model/device/RF execution; not native Windows validation.",
        }
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"file": str(args.output), "pixels": metadata["screenshot_pixels"],
                          "scale": metadata["device_pixel_ratio"], "visible_clips": visible_cells}))
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
