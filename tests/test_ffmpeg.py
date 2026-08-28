"""FFmpeg executable resolution for source and portable builds."""

from pathlib import Path

import pytest

from aetv import ffmpeg


def test_frozen_app_resolves_bundled_ffmpeg_without_path(monkeypatch, tmp_path):
    app = tmp_path / "AETV.exe"
    bundled = tmp_path / "ffmpeg.exe"
    app.touch()
    bundled.touch()
    monkeypatch.delenv("AETV_FFMPEG", raising=False)
    monkeypatch.setattr(ffmpeg.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ffmpeg.sys, "platform", "win32")
    monkeypatch.setattr(ffmpeg.sys, "executable", str(app))

    assert ffmpeg.ffmpeg_executable() == str(bundled.resolve())


def test_frozen_app_does_not_fall_back_to_system_path(monkeypatch, tmp_path):
    app = tmp_path / "AETV.exe"
    app.touch()
    monkeypatch.delenv("AETV_FFMPEG", raising=False)
    monkeypatch.setattr(ffmpeg.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ffmpeg.sys, "platform", "win32")
    monkeypatch.setattr(ffmpeg.sys, "executable", str(app))

    with pytest.raises(RuntimeError, match="missing its bundled FFmpeg"):
        ffmpeg.ffmpeg_executable()


def test_explicit_ffmpeg_override_is_validated(monkeypatch, tmp_path):
    candidate = tmp_path / "custom-ffmpeg"
    candidate.touch()
    monkeypatch.setenv("AETV_FFMPEG", str(candidate))

    assert Path(ffmpeg.ffmpeg_executable()) == candidate.resolve()
