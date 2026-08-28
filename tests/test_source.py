"""Shared live-camera buffering."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from aetv.source import CameraFrameBuffer, iter_video_file, write_video_smoke_test


class _StuckThread:
    def __init__(self):
        self.joins = []

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.joins.append(timeout)


def test_camera_buffer_opens_once_for_preview_and_transmit(monkeypatch):
    opens = []

    def camera(_mode, camera=0, duration_s=None):
        opens.append(camera)
        index = 0
        while True:
            yield np.full((2, 3, 3), index, dtype=np.uint8)
            index += 1
            time.sleep(0.005)

    monkeypatch.setattr("aetv.source.iter_webcam", camera)
    mode = SimpleNamespace(name="V7")
    buffer = CameraFrameBuffer(history_frames=8)
    preview = buffer.frames(mode, camera=2)
    first = next(preview)
    transmit = buffer.frames(mode, camera=2)
    second = next(transmit)
    assert opens == [2]
    assert int(second[0, 0, 0]) > int(first[0, 0, 0])
    preview.close()
    transmit.close()
    buffer.close()


def test_camera_preview_skips_to_latest_buffered_frame(monkeypatch):
    def camera(_mode, camera=0, duration_s=None):
        index = 0
        while True:
            yield np.full((2, 3, 3), index, dtype=np.uint8)
            index += 1
            time.sleep(0.005)

    monkeypatch.setattr("aetv.source.iter_webcam", camera)
    mode = SimpleNamespace(name="V7")
    buffer = CameraFrameBuffer(history_frames=16)
    preview = buffer.frames(mode, latest=True)
    first = int(next(preview)[0, 0, 0])
    time.sleep(0.04)
    newest = int(next(preview)[0, 0, 0])
    assert newest >= first + 4
    preview.close()
    buffer.close()


def test_camera_buffer_never_overlaps_a_stuck_native_worker():
    buffer = CameraFrameBuffer()
    worker = _StuckThread()
    buffer._thread = worker
    buffer._key = (0, "old")

    with pytest.raises(RuntimeError, match="refusing to open a second camera backend"):
        buffer.configure(SimpleNamespace(name="new"), camera=1)

    assert buffer._thread is worker
    assert worker.joins == [3.0]


def test_video_file_repeats_to_fill_requested_transmission(monkeypatch):
    mode = SimpleNamespace(fps=6.0, width=3, height=2)
    requested_frames = 12
    expected_bytes = requested_frames * mode.width * mode.height * 3
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=bytes(expected_bytes),
            stderr=b"",
        )

    monkeypatch.setattr("aetv.source.subprocess.run", run)
    monkeypatch.setattr("aetv.source.ffmpeg_executable", lambda: "C:/AETV/ffmpeg.exe")

    frames = iter_video_file("short.mp4", mode, frames=requested_frames)

    assert captured["command"][captured["command"].index("-stream_loop") + 1] == "-1"
    assert captured["command"][0] == "C:/AETV/ffmpeg.exe"
    assert frames.shape == (requested_frames, mode.height, mode.width, 3)


def test_video_save_smoke_test_creates_mp4(tmp_path):
    target = tmp_path / "saved-video.mp4"

    write_video_smoke_test(target)

    assert target.stat().st_size > 32
    assert b"ftyp" in target.read_bytes()[:32]
