"""Desktop helpers must not load AETV's bundled Qt/OpenCV libraries."""

import os
from pathlib import Path
import sys

import pytest

from aetv.gui import desktop


@pytest.mark.parametrize("original", [None, "", "/opt/system-libraries"])
def test_frozen_desktop_environment_restores_loader_paths(monkeypatch, original):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/aetv/_internal:/opt/system-libraries")
    if original is None:
        monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", original)
    overrides = ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR",
                 "QML_IMPORT_PATH", "QML2_IMPORT_PATH")
    for name in overrides:
        monkeypatch.setenv(name, "/aetv/_internal/cv2/qt")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    monkeypatch.setenv("QT_QPA_PLATFORMTHEME", "lxqt")

    environment = desktop._desktop_environment()

    assert environment.get("LD_LIBRARY_PATH") == (original or None)
    assert "LD_LIBRARY_PATH_ORIG" not in environment
    assert all(name not in environment for name in overrides)
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
    assert environment["QT_QPA_PLATFORMTHEME"] == "lxqt"
    assert os.environ["LD_LIBRARY_PATH"] == "/aetv/_internal:/opt/system-libraries"
    assert all(os.environ[name] == "/aetv/_internal/cv2/qt" for name in overrides)


def test_source_desktop_environment_removes_opencv_overrides_only(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/system-libraries")
    monkeypatch.setenv("QT_PLUGIN_PATH", "/opt/system-qt-plugins")
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/venv/cv2/qt/plugins")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/venv/cv2/qt/fonts")
    environment = desktop._desktop_environment()
    assert environment["LD_LIBRARY_PATH"] == "/opt/system-libraries"
    assert environment["QT_PLUGIN_PATH"] == "/opt/system-qt-plugins"
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in environment
    assert "QT_QPA_FONTDIR" not in environment


@pytest.mark.parametrize("started", [False, True])
def test_linux_folder_launch_passes_arguments_and_child_environment(monkeypatch, tmp_path, started):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/aetv/_internal")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    captured = {}

    class Process:
        def setProgram(self, value):
            captured["program"] = value

        def setArguments(self, value):
            captured["arguments"] = value

        def setProcessEnvironment(self, value):
            captured["environment"] = value

        def startDetached(self):
            return started

    monkeypatch.setattr(desktop, "QProcess", Process)
    folder = tmp_path / "received clips & samples"
    assert desktop.open_directory(folder) is started
    assert captured["program"] == "xdg-open"
    assert captured["arguments"] == [str(folder.resolve())]
    assert not captured["environment"].contains("LD_LIBRARY_PATH")
    assert os.environ["LD_LIBRARY_PATH"] == "/aetv/_internal"


@pytest.mark.parametrize("exists", [False, True])
def test_linux_folder_launch_uses_real_qprocess_result(monkeypatch, tmp_path, exists):
    executable = sys.executable if exists else str(tmp_path / "missing-desktop-opener")

    class Process(desktop.QProcess):
        def setProgram(self, value):
            super().setProgram(executable)

        def setArguments(self, value):
            super().setArguments(["-c", "pass"])

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(desktop, "QProcess", Process)
    assert desktop.open_directory(tmp_path) is exists


def test_other_platforms_keep_native_desktop_opening(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    opened = []
    monkeypatch.setattr(desktop.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    assert desktop.open_directory(tmp_path)
    assert Path(opened[0].toLocalFile()) == tmp_path.resolve()
