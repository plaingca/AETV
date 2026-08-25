"""First-run model selection UI behavior."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from aetv.codec import ReleaseModelStatus
import aetv.gui.app as app_module
from aetv.gui.app import MainWindow
from aetv.gui.model_manager import ModelManagerDialog


_APP = QApplication.instance() or QApplication([])


def test_first_run_preselects_current_mode_and_requires_an_install():
    statuses = {
        mode: ReleaseModelStatus(mode, False, problem="not installed")
        for mode in ("V8", "V7")
    }

    dialog = ModelManagerDialog("V8", statuses=statuses, first_run=True)

    assert dialog._checks["V8"].isChecked()
    assert not dialog._checks["V7"].isChecked()
    assert dialog.download_button.isEnabled()
    assert not dialog.continue_button.isEnabled()
    dialog.close()


def test_first_run_can_continue_when_either_mode_is_valid(tmp_path):
    statuses = {
        "V8": ReleaseModelStatus("V8", False, problem="not installed"),
        "V7": ReleaseModelStatus(
            "V7", True, tmp_path / "wide.runtime.json", "ONNX Runtime"
        ),
    }

    dialog = ModelManagerDialog("V8", statuses=statuses, first_run=True)

    assert dialog.installed_modes() == ["V7"]
    assert dialog.continue_button.isEnabled()
    assert not dialog._checks["V7"].isEnabled()
    dialog.close()


def test_installed_release_replaces_a_failed_explicit_model(tmp_path, monkeypatch):
    settings = SimpleNamespace(mode="V7", checkpoint="missing.runtime.json")
    loaded = []
    window = SimpleNamespace(
        settings=settings,
        station=SimpleNamespace(
            codec=SimpleNamespace(mode=SimpleNamespace(name="V8")),
            settings=settings,
        ),
        rx=SimpleNamespace(sync_from_config=lambda: None),
        tx=SimpleNamespace(sync_from_config=lambda: None),
        waterfall=SimpleNamespace(set_mode=lambda _mode: None),
        _log=lambda _message: None,
        _refresh_station_label=lambda: None,
        _load_codec=lambda: loaded.append(True),
    )
    monkeypatch.setattr(app_module, "save_settings", lambda _settings: None)
    statuses = {
        "V7": ReleaseModelStatus(
            "V7", True, tmp_path / "wide.runtime.json", "ONNX Runtime"
        )
    }

    MainWindow._activate_available_model(window, statuses)

    assert settings.checkpoint == ""
    assert loaded == [True]
