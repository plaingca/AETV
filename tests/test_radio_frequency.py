"""Main-panel HF tuning and CAT frequency routing."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from aetv import cat
from aetv.gui.radio_panel import RadioPanel
from aetv.settings import StationSettings


_APP = QApplication.instance() or QApplication([])


@pytest.mark.parametrize("backend", ["flex", "hamlib", "rigctld"])
def test_hf_selector_emits_dial_without_overwriting_sdr(backend):
    panel = RadioPanel(StationSettings(cat_backend=backend, freq_mhz=7.088))
    selected = []
    panel.applyRequested.connect(selected.append)
    assert panel.frequency.isEnabled()
    assert panel.frequency.value() == 7.088
    panel.frequency.setValue(14.230123)
    panel.apply.click()
    assert selected[-1]["freq_mhz"] == 14.230123
    assert "sdr_frequency_mhz" not in selected[-1]
    panel.frequency.setValue(0)
    panel.apply.click()
    assert selected[-1]["freq_mhz"] is None
    panel.close()


def test_switching_routes_and_settings_sync_preserve_separate_dials():
    settings = StationSettings(cat_backend="flex", freq_mhz=7.088, sdr_frequency_mhz=439.025)
    panel = RadioPanel(settings)
    panel.frequency.setValue(14.230)
    panel.backend.setCurrentIndex(panel.backend.findData("hackrf"))
    assert panel.frequency.value() == 439.025
    panel.frequency.setValue(145.5)
    panel.backend.setCurrentIndex(panel.backend.findData("audio"))
    assert panel.frequency.value() == 14.230
    panel.set_receive_source("rtlsdr")
    assert panel.frequency.value() == 145.5
    panel.set_receive_source("soundcard")
    assert panel.frequency.value() == 14.230
    settings.freq_mhz = 3.573
    panel.sync(settings)
    assert panel.frequency.value() == 3.573
    panel.close()


@pytest.mark.parametrize("backend", ["none", "rts", "dtr"])
def test_ptt_only_routes_do_not_offer_hf_tuning(backend):
    panel = RadioPanel(StationSettings(cat_backend=backend))
    assert not panel.frequency.isEnabled()
    panel.set_receive_source("flex")
    assert panel.frequency.isEnabled()
    panel.close()


@pytest.mark.parametrize("backend", ["hamlib", "rigctld"])
@pytest.mark.parametrize("frequency", [None, 7.088])
def test_cat_open_applies_requested_dial(monkeypatch, backend, frequency):
    calls = []
    client = SimpleNamespace(set_frequency_hz=calls.append, close=lambda: calls.append("close"))
    monkeypatch.setattr(cat, "HamlibDirect", lambda *args: client)
    monkeypatch.setattr(cat, "RigctldClient", lambda *args: client)
    assert cat.open_ptt(cat.CatConfig(backend=backend, freq_mhz=frequency)) is client
    assert calls == ([] if frequency is None else [7088000])


def test_failed_cat_tune_closes_connection(monkeypatch):
    calls = []
    def fail(hz):
        raise cat.CatError("tune failed")
    client = SimpleNamespace(set_frequency_hz=fail, close=lambda: calls.append("close"))
    monkeypatch.setattr(cat, "HamlibDirect", lambda *args: client)
    with pytest.raises(cat.CatError, match="tune failed"):
        cat.open_ptt(cat.CatConfig(backend="hamlib", freq_mhz=7.088))
    assert calls == ["close"]


def test_cat_frequency_commands():
    rigctld = cat.RigctldClient.__new__(cat.RigctldClient)
    commands = []
    rigctld._command = commands.append
    rigctld.set_frequency_hz(14230123)
    assert commands == ["F 14230123"]
    hamlib = cat.HamlibDirect.__new__(cat.HamlibDirect)
    hamlib.rig = object()
    calls = []
    hamlib.lib = SimpleNamespace(rig_set_freq=lambda *args: calls.append(args) or 0)
    hamlib.set_frequency_hz(14230123)
    assert calls == [(hamlib.rig, hamlib._VFO_CURR, 14230123)]


def test_soundcard_receive_tunes_before_opening_audio(monkeypatch):
    from aetv import station
    settings = StationSettings(cat_backend="hamlib", freq_mhz=7.088, rx_source="soundcard")
    radio = station.Station(settings)
    monkeypatch.setattr(radio, "require_codec", lambda: None)
    calls = []
    def open_cat(config):
        calls.append(config.freq_mhz)
        return SimpleNamespace(close=lambda: calls.append("close"))
    monkeypatch.setattr(station, "open_ptt", open_cat)
    def stop_before_capture(*args):
        raise RuntimeError("capture reached")
    monkeypatch.setattr(station, "RingBuffer", stop_before_capture)
    receiver = station.RxEngine(radio)
    with pytest.raises(RuntimeError, match="capture reached"):
        receiver.start()
    assert calls == [7.088, "close"]


@pytest.mark.parametrize("frequency", [14.230123, None])
def test_main_selector_routes_frequency_to_native_flex_tx_and_rx(monkeypatch, frequency):
    import numpy as np
    from aetv import station
    from aetv.config import AETV_MODES

    sessions = []

    class FakeFlex:
        def __init__(self, host, **kwargs):
            self.config = kwargs
            self.events = []
            sessions.append(self)
            assert host == "192.0.2.1"

        def prepare_tx(self):
            self.events.append("prepare")

        def describe(self):
            return "native Flex"

        def set_ptt(self, on):
            self.events.append(("ptt", on))

        def send_audio_stream(self, chunks, sample_rate, **kwargs):
            list(chunks)
            self.events.append("transmit")
            return True

        def start_rx(self, callback, **kwargs):
            self.events.append("receive")

        def close(self):
            self.events.append("close")

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("native Flex must not fall back to CAT PTT or soundcard audio")

    monkeypatch.setattr(station, "FlexVitaSession", FakeFlex)
    monkeypatch.setattr(station, "open_ptt", unexpected_fallback)
    monkeypatch.setattr(station, "open_input_stream", unexpected_fallback)
    monkeypatch.setattr(station.RxEngine, "_loop", lambda self: None)
    settings = StationSettings(
        mode="V8", cat_backend="flex", flex_host="192.0.2.1",
        flex_native_audio=True, rx_source="flex", freq_mhz=7.088,
        ptt_lead_s=0, ptt_tail_s=0, debug_capture=False,
    )
    panel = RadioPanel(settings)
    panel.applyRequested.connect(lambda values: [setattr(settings, key, value) for key, value in values.items()])
    panel.frequency.setValue(frequency or 0)
    panel.apply.click()
    radio = station.Station(settings)
    radio.codec = SimpleNamespace(mode=AETV_MODES["V8"])
    transmitter = station.TxEngine(radio)
    chunks = [np.zeros(800, dtype=np.float32)]
    assert transmitter._keyed_send_stream(iter(chunks), 8000, len(chunks))
    receiver = station.RxEngine(radio)
    try:
        receiver.start()
        assert len(sessions) == 2
        assert [session.config["frequency_mhz"] for session in sessions] == [frequency, frequency]
        assert "transmit" in sessions[0].events
        assert "receive" in sessions[1].events
        assert ("ptt", True) not in sessions[1].events
    finally:
        receiver.stop()
        panel.close()
