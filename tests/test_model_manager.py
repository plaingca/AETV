"""First-run model selection UI behavior."""

import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from aetv.codec import ReleaseModelStatus
import aetv.gui.app as app_module
from aetv.gui.app import MainWindow
from aetv.settings import StationSettings
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


def test_tx_mode_request_updates_config_and_starts_model_load(monkeypatch):
    settings = StationSettings(mode="V8", checkpoint="local-v8.runtime.json")
    loaded = []
    saved = []
    window = SimpleNamespace(
        settings=settings,
        station=SimpleNamespace(settings=settings),
        tx=SimpleNamespace(transmitting=lambda: False),
        rx=SimpleNamespace(listening=lambda: False),
        waterfall=SimpleNamespace(set_mode=lambda mode: setattr(window, "waterfall_mode", mode)),
        _refresh_station_label=lambda: None,
        _log=lambda message: setattr(window, "message", message),
        _load_codec=lambda: loaded.append(True),
    )
    monkeypatch.setattr(app_module, "save_settings", saved.append)

    MainWindow._on_tx_mode_requested(window, "V7")

    assert settings.mode == "V7"
    assert settings.checkpoint == ""
    assert loaded == [True]
    assert saved == [settings]


def test_stale_background_model_load_cannot_replace_requested_mode():
    settings = StationSettings(mode="V7")
    current = SimpleNamespace(mode=SimpleNamespace(name="V7"))
    stale = SimpleNamespace(mode=SimpleNamespace(name="V8"))
    worker = SimpleNamespace(config=("V8", "", ""))
    window = SimpleNamespace(
        settings=settings,
        station=SimpleNamespace(codec=current, codec_lock=threading.Lock()),
        _log=lambda message: setattr(window, "message", message),
    )

    MainWindow._on_model_loaded(window, worker, stale, "V8 on CPU")

    assert window.station.codec is current
    assert "discarding stale codec load" in window.message


def test_smoke_waits_for_queued_codec_result_after_worker_stops():
    window = SimpleNamespace(
        _model_inventory_thread=None,
        station=SimpleNamespace(codec=None),
        _last_codec_error="",
    )

    assert app_module._smoke_codec_result(window) is None
    window.station.codec = object()
    assert app_module._smoke_codec_result(window) == 0
    window.station.codec = None
    window._last_codec_error = "model failed"
    assert app_module._smoke_codec_result(window) == 1


def test_active_receive_restarts_when_source_settings_change(monkeypatch):
    settings = StationSettings(mode="V8", rx_source="soundcard")
    stopped = []

    class FakeDialog:
        class DialogCode:
            Accepted = 1

        def __init__(self, _settings, _parent):
            pass

        def exec(self):
            return self.DialogCode.Accepted

        def apply_to(self, target):
            target.rx_source = "kiwi"
            target.kiwi_host = "kiwi.example:8073"

    window = SimpleNamespace(
        settings=settings,
        station=SimpleNamespace(
            settings=settings,
            codec=SimpleNamespace(mode=SimpleNamespace(name="V8")),
        ),
        rx=SimpleNamespace(
            sync_from_config=lambda: None,
            listening=lambda: True,
            stop=lambda: stopped.append(True),
        ),
        tx=SimpleNamespace(
            sync_from_config=lambda: None,
            transmitting=lambda: False,
        ),
        waterfall=SimpleNamespace(set_mode=lambda _mode: None),
        _refresh_station_label=lambda: None,
        _log=lambda _message: None,
        _reload_codec_after_rx_stop=False,
        _resume_rx_after_codec_reload=False,
        _restart_rx_after_settings_stop=False,
        _load_codec=lambda: None,
    )
    monkeypatch.setattr(app_module, "SettingsDialog", FakeDialog)
    monkeypatch.setattr(app_module, "save_settings", lambda _settings: None)

    MainWindow.open_settings(window)

    assert stopped == [True]
    assert window._restart_rx_after_settings_stop
    assert not window._reload_codec_after_rx_stop


def test_settings_cannot_mutate_active_transmission(monkeypatch):
    opened = []
    status = SimpleNamespace(setText=lambda text: setattr(status, "text", text))
    window = SimpleNamespace(
        tx=SimpleNamespace(transmitting=lambda: True, status=status),
        _log=lambda message: setattr(window, "message", message),
    )
    monkeypatch.setattr(
        app_module, "SettingsDialog", lambda *_args: opened.append(True)
    )

    MainWindow.open_settings(window)

    assert opened == []
    assert status.text == "settings cannot be changed during a transmission"
