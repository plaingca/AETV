import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aetv.config import AETV_MODES
from aetv.framing import pack_gop_symbols, unpack_gop_symbols
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.sdr import IQPreview, stop_pluto_tx, transmit_pluto
from aetv.sdr_dsp import IQToModem, ModemToIQ
from aetv.settings import StationSettings


def test_ac16_wire_interleaver_roundtrip_and_reserved_slot():
    z = np.random.default_rng(439).normal(size=19200).astype(np.float32)
    symbols = pack_gop_symbols(z, np.ones(32), band="A")
    assert symbols.shape == (32, 302)
    assert np.all(symbols[:, 300] == 0)
    actual, weights = unpack_gop_symbols(symbols, np.ones_like(symbols.real), band="A")
    np.testing.assert_array_equal(z, actual)
    assert weights.shape == (19200,)


def test_causal_sdr_adapter_recovers_ac16_continuous_stream():
    sent = np.random.default_rng(439).normal(size=(3, 19200)).astype(np.float32)
    tx = ModemToIQ(sample_rate=960000)
    rx = IQToModem(signal_offset_hz=100000)
    demod = StreamingDemodulator(
        "A", continuous=True, mode_name="AC16", boundary_tracking=True
    )
    output = []
    for audio in list(modulate_continuous_chunks(sent, mode_name="AC16")) + [
        np.zeros(4800)
    ]:
        audio = audio * 0.2
        for i in range(0, len(audio), 4800):
            for result in demod.feed(rx.feed(tx.feed(audio[i : i + 4800]))):
                output.extend(result.gops_latents)
    assert np.array(output).shape == sent.shape
    for a, b in zip(sent, output):
        assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.9


def test_automatic_iq_headroom_is_linear_and_does_not_pump_gain():
    t = np.arange(4800) / 48000
    audio = 4*np.sin(2*np.pi*2000*t)
    with pytest.raises(ValueError, match='DAC component range'):
        ModemToIQ(sample_rate=960000).feed(audio)
    protected = ModemToIQ(sample_rate=960000, peak_limit=.9)
    reference = ModemToIQ(sample_rate=960000)
    for scale in (1, .01, .1):
        output = protected.feed(audio*scale)
        expected = reference.feed(audio*scale*.01) * (100*protected.headroom_gain)
        np.testing.assert_allclose(output, expected, rtol=2e-6, atol=1e-7)
        assert abs(output).max() <= .900001
        if scale == 1:
            initial_gain = protected.headroom_gain
            assert initial_gain < 1
        else:
            assert protected.headroom_gain <= initial_gain


def test_pluto_transmit_handles_short_audio_peaks_at_maximum_gui_level():
    from aetv.ac16_av_smoke import pluto_headroom_smoke
    assert pluto_headroom_smoke()['decoded_gops'] == 3


@pytest.mark.parametrize("failure", ["push", "source", "cancel", None])
def test_pluto_always_powers_down_and_destroys_buffer(monkeypatch, failure):
    import aetv.sdr as module

    power = SimpleNamespace(value="1")
    closed = []
    cancel = threading.Event()

    class Radio:
        _ctrl = SimpleNamespace(
            find_channel=lambda *_: SimpleNamespace(attrs={"powerdown": power})
        )
        tx_hardwaregain_chan0 = -89.75

        def disable_dds(self):
            pass

        def tx_destroy_buffer(self):
            closed.append(True)

        def tx(self, values):
            assert len(values) == 240000
            if failure == "push":
                raise OSError("USB disconnected")
            if failure == "cancel":
                cancel.set()

    radio = Radio()
    monkeypatch.setattr(module, "open_pluto", lambda _: radio)

    def chunks():
        yield np.sin(2 * np.pi * 2000 * np.arange(4800) / 48000) * 0.1
        if failure == "source":
            raise RuntimeError("capture failed")

    if failure in {"push", "source"}:
        with pytest.raises((RuntimeError, OSError)):
            transmit_pluto(
                chunks(),
                48000,
                StationSettings(mode="AC16"),
                cancel,
                lambda _: None,
                max_seconds=1,
            )
    else:
        assert transmit_pluto(
            chunks(),
            48000,
            StationSettings(mode="AC16"),
            cancel,
            lambda _: None,
            max_seconds=1,
        ) == (failure is None)
    assert power.value == "1" and radio.tx_hardwaregain_chan0 == -89.75 and closed


def test_shutdown_attempts_gain_and_buffer_even_if_lo_write_fails():
    calls = []

    class Radio:
        _ctrl = SimpleNamespace(
            find_channel=lambda *_: (_ for _ in ()).throw(OSError("no link"))
        )

        def tx_destroy_buffer(self):
            calls.append("destroyed")

    radio = Radio()
    with pytest.raises(RuntimeError, match="shutdown failed"):
        stop_pluto_tx(radio)
    assert radio.tx_hardwaregain_chan0 == -89.75 and calls == ["destroyed"]


def test_iq_waterfall_retains_complex_samples_and_bounds_history():
    preview = IQPreview(960000, 439e6, AETV_MODES["AC16"])
    values = np.arange(100000) * (1 + 2j)
    preview.write(values)
    assert preview.tail(100000).shape == (65536,)
    np.testing.assert_array_equal(preview.tail(20), values[-20:])
    assert preview.lo_hz == 439100000


def test_sdr_settings_roundtrip_and_hardware_ranges(tmp_path):
    from aetv.settings import load_settings, save_settings

    settings = StationSettings(
        mode="AC16",
        tx_backend="pluto",
        rx_source="rtlsdr",
        receive_dir=str(tmp_path / "received"),
    )
    assert not settings.validate()
    path = tmp_path / "settings.json"
    save_settings(settings, path)
    assert load_settings(path) == settings
    settings.pluto_tx_gain = 1
    assert any("TX gain" in problem for problem in settings.validate())
    settings.pluto_tx_gain = -20
    settings.sdr_frequency_mhz = 2000
    assert any("RTL-SDR frequency" in problem for problem in settings.validate())
    settings.tx_backend = "audio"
    settings.sdr_frequency_mhz = 50
    settings.pluto_uri = ""
    assert not settings.validate()


def test_radio_controls_emit_operator_selection():
    from PySide6.QtWidgets import QApplication

    from aetv.gui.radio_panel import RadioPanel

    app = QApplication.instance() or QApplication([])
    panel = RadioPanel(StationSettings(mode="AC16", rx_source="rtlsdr"))
    selected = []
    panel.applyRequested.connect(selected.append)
    panel.backend.setCurrentIndex(panel.backend.findData("pluto"))
    panel.frequency.setValue(439.025)
    panel.tx_gain.setValue(-80)
    panel.rx_gain.setValue(372)
    panel.apply.click()
    assert selected[-1]["sdr_frequency_mhz"] == 439.025
    assert selected[-1]["pluto_tx_gain"] == -20
    assert selected[-1]["rtl_rx_gain"] == 37.2
    panel.set_receive_source("pluto")
    assert panel.rx_gain.maximum() == 700
    panel.close()
    app.processEvents()


@pytest.mark.parametrize('validated', [False, True])
def test_sdr_retries_unvalidated_frequency_fit_but_keeps_valid_payload(monkeypatch, validated):
    from aetv import sdr
    clock = [0.]
    messages, resets = [], []
    capture = sdr.SDRCapture(
        StationSettings(rx_source='hackrf', mode='V8', sdr_auto_correct=True),
        AETV_MODES['V8'], SimpleNamespace(write=lambda _: None),
        on_error=lambda _: None, on_status=messages.append,
        on_discontinuity=lambda: resets.append(clock[0]),
    )
    monkeypatch.setattr(sdr.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(sdr, 'estimate_mode_signal_offset',
                        lambda *a, **k: {'offset_hz': -101250.})
    monkeypatch.setattr(sdr, 'IQToModem', lambda *a: SimpleNamespace(feed=lambda x: x.real))
    monkeypatch.setattr(sdr, 'StreamResampler', lambda *a: lambda x: x)

    class Input:
        def get(self, **kwargs):
            clock[0] += .1
            if clock[0] >= 65:
                capture._stop.set()
            return np.zeros(16, np.complex64)

    capture._queue = Input()
    if validated:
        capture.ring.write = lambda _: capture.confirm_signal()
    capture._convert()
    corrections = [m for m in messages if 'received-only' in m]
    assert len(corrections) == (1 if validated else 3)
    assert len(resets) == (0 if validated else 2)
    capture.settings.sdr_auto_correct = False
    clock[0] += 100
    assert not capture._correction_expired()
