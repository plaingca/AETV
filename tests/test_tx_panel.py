"""Transmit-panel preview routing."""

import threading
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QDialog

from aetv.config import AETV_MODES
from aetv.gui.rx_panel import ReceivePanel
from aetv.gui.tx_panel import TransmitPanel
from aetv.source import ScreenCaptureSpec
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
        _clip_tokens={},
        _selected_clip=3,
        _prepared_clips={3: object()},
        _clip_edits={3: edit},
        clip_grid=SimpleNamespace(cells=[None, None, None, cell]),
        station=SimpleNamespace(codec=SimpleNamespace(mode=SimpleNamespace(name="V8"))),
        _selected_mode_name=lambda: "V8",
        _prepare_clip_batch=lambda *_args: None,
    )
    panel._invalidate_clip_slots = lambda indices: TransmitPanel._invalidate_clip_slots(
        panel, indices
    )

    TransmitPanel._queue_clip_preparation(panel)

    assert cell.progress == 0.0
    assert started == [([(3, edit)], "V8", 1)]


def test_adding_clip_only_queues_changed_slot(monkeypatch):
    started = []

    class Thread:
        def __init__(self, *, target, args, **_kwargs):
            self.args = args

        def start(self):
            started.append(self.args)

    monkeypatch.setattr("aetv.gui.tx_panel.threading.Thread", Thread)
    existing = object()
    old_edit = SimpleNamespace(path="old.mp4")
    new_edit = SimpleNamespace(path="new.mp4")
    cells = [SimpleNamespace(set_progress=lambda value: None) for _ in range(4)]
    progress = []
    cells[3].set_progress = progress.append
    panel = SimpleNamespace(
        _clip_generation=7,
        _clip_tokens={0: 4},
        _selected_clip=0,
        _prepared_clips={0: existing},
        _clip_edits={0: old_edit, 3: new_edit},
        clip_grid=SimpleNamespace(cells=cells),
        station=SimpleNamespace(codec=SimpleNamespace(mode=SimpleNamespace(name="V8"))),
        _selected_mode_name=lambda: "V8",
        _prepare_clip_batch=lambda *_args: None,
    )
    panel._invalidate_clip_slots = lambda indices: TransmitPanel._invalidate_clip_slots(
        panel, indices
    )

    TransmitPanel._queue_clip_preparation(panel, [3])

    assert panel._prepared_clips == {0: existing}
    assert panel._selected_clip == 0
    assert panel._clip_tokens == {0: 4, 3: 8}
    assert progress == [0.0]
    assert started == [([(3, new_edit)], "V8", 8)]


def test_invalidated_batch_slot_does_not_cancel_other_slots():
    prepared_indices = []
    panel = SimpleNamespace(
        _clip_tokens={0: 2, 1: 1},
        engine=SimpleNamespace(
            prepare_clip=lambda path, *_args, **_kwargs: f"prepared:{path}"
        ),
        _clipProgress=SimpleNamespace(emit=lambda *_args: None),
        _clipReady=SimpleNamespace(
            emit=lambda index, _prepared, _generation: prepared_indices.append(index)
        ),
        _clipFailed=SimpleNamespace(emit=lambda *_args: None),
    )
    edits = [
        (0, SimpleNamespace(path="changed.mp4", duration_s=1, start_s=0, framing="crop")),
        (1, SimpleNamespace(path="keep.mp4", duration_s=1, start_s=0, framing="crop")),
    ]

    TransmitPanel._prepare_clip_batch(panel, edits, "V8", 1)

    assert prepared_indices == [1]


def test_live_source_selection_updates_during_transmit():
    screen = ScreenCaptureSpec("Monitor 1", (0, 0, 1920, 1080))
    panel = SimpleNamespace(
        cam_radio=SimpleNamespace(isChecked=lambda: False),
        screen_radio=SimpleNamespace(isChecked=lambda: True),
        screen_target=SimpleNamespace(currentData=lambda: screen),
        _live_source_lock=threading.Lock(),
        _active_live_source="webcam",
    )

    TransmitPanel._update_active_live_source(panel)

    assert panel._active_live_source == screen


def test_audio_level_update_routes_independent_channel_peaks():
    class Meter:
        def set_level(self, peak, clipping):
            self.reading = (peak, clipping)

    panel = SimpleNamespace(mic_meter=Meter(), clip_meter=Meter())

    TransmitPanel._apply_audio_levels(
        panel,
        {
            "microphone_peak": 0.4,
            "microphone_clipping": False,
            "clip_peak": 1.1,
            "clip_clipping": True,
        },
    )

    assert panel.mic_meter.reading == (0.4, False)
    assert panel.clip_meter.reading == (1.1, True)


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


def test_loopback_save_passes_recovered_audio_to_mp4_writer():
    captured = {}
    audio = np.ones(8000, dtype=np.float32)

    class Engine:
        def save_video(self, video, **kwargs):
            captured["video"] = video
            captured.update(kwargs)
            return Path("loopback.mp4")

    panel = SimpleNamespace(
        _emulated_video=np.zeros((2, 2, 3, 3), dtype=np.uint8),
        station=SimpleNamespace(loopback_audio=audio, loopback_audio_rate=8000),
        engine=Engine(),
        status=SimpleNamespace(setText=lambda text: setattr(panel.status, "text", text)),
        logMessage=SimpleNamespace(emit=lambda _message: None),
    )

    ReceivePanel.save_current(panel)

    assert np.array_equal(captured["audio"], audio)
    assert captured["audio_rate"] == 8000
    assert panel.status.text == "saved loopback.mp4"
