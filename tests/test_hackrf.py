import ctypes as C
import queue
import threading
from dataclasses import replace

import numpy as np
import pytest

from aetv.hackrf import (
    HackRF, SAMPLE_RATE, Transfer, decode_iq, encode_iq, transmit_hackrf,
)
from aetv.settings import StationSettings


class FakeLibrary:
    """Drive real ctypes callbacks, including short/nonuniform USB transfers."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.output = bytearray()

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name == "hackrf_error_name":
                return b"TEST_USB_ERROR"
            if name == self.fail:
                return -1000
            if name == "hackrf_open_by_serial":
                C.cast(args[1], C.POINTER(C.c_void_p))[0] = C.c_void_p(123)
            if name == "hackrf_start_rx":
                self.receive = args[1]
            if name == "hackrf_enable_tx_flush":
                self.flush = args[1]
            if name == "hackrf_start_tx":
                self.transmit = args[1]
            if name == "hackrf_is_streaming":
                return 1
            return 0
        return call

    def rx(self, data, valid=None):
        buf = (C.c_uint8 * len(data)).from_buffer_copy(data)
        transfer = Transfer(None, buf, len(data), len(data) if valid is None else valid, None, None)
        return self.receive(C.pointer(transfer))

    def tx(self, size=262144):
        buf = (C.c_uint8 * size)()
        transfer = Transfer(None, buf, size, size, None, None)
        result = self.transmit(C.pointer(transfer))
        # libhackrf submits only buffers whose callback returns zero.
        if result == 0:
            self.output.extend(bytes(buf[:transfer.valid_length]))
        return result


def settings(**kwargs):
    return StationSettings(mode="AC16", tx_backend="hackrf", rx_source="hackrf", **kwargs)


def test_signed_iq_full_modem_roundtrip():
    from aetv.hackrf_smoke import transport_smoke
    assert transport_smoke()["gops"] == 3


def test_signed_iq_byte_order_and_range():
    iq = np.array([-1 + 0j, -0.5 + 0.25j, 0.5 - 0.75j], np.complex64)
    np.testing.assert_array_equal(decode_iq(bytes([128, 0, 192, 32, 64, 160])), iq)
    assert encode_iq(iq[1:]) == bytes([192, 32, 64, 160])
    assert encode_iq(np.array([0.999 + 0j])) == bytes([127, 0])
    for bad in ([complex(np.nan, 0)], [1 + 0j], [0 - 1j]):
        with pytest.raises(ValueError, match="range"):
            encode_iq(bad)
    with pytest.raises(ValueError, match="Truncated"):
        decode_iq(b"\0")


@pytest.mark.parametrize("direction,lo", [("rx", 439100000), ("tx", 438900000)])
def test_settings_reach_native_api_and_radio_is_closed(direction, lo):
    lib = FakeLibrary()
    radio = HackRF(settings(hackrf_serial="0123abcd", hackrf_tx_gain=12), direction, library=lib)
    if direction == "rx":
        radio.start_rx()
    radio.close()
    radio.close()
    assert ("hackrf_set_freq", (radio.device, lo)) in lib.calls
    assert ("hackrf_set_sample_rate", (radio.device, SAMPLE_RATE)) in lib.calls
    assert ("hackrf_set_amp_enable", (radio.device, 0)) in lib.calls
    assert ("hackrf_set_antenna_enable", (radio.device, 0)) in lib.calls
    assert [name for name, _ in lib.calls][-2:] == ["hackrf_close", "hackrf_exit"]
    assert sum(name == "hackrf_close" for name, _ in lib.calls) == 1


@pytest.mark.parametrize("failure", ["hackrf_open_by_serial", "hackrf_set_freq", "hackrf_start_rx"])
def test_partial_startup_failure_releases_native_state_and_lock(failure):
    lib = FakeLibrary(fail=failure)
    radio = None
    with pytest.raises(RuntimeError, match="TEST_USB_ERROR"):
        try:
            radio = HackRF(settings(), "rx", library=lib)
            radio.start_rx()
        finally:
            if radio:
                radio.close()
    assert lib.calls[-1][0] == "hackrf_exit"
    HackRF(settings(), "rx", library=FakeLibrary()).close()


def test_half_duplex_guard_does_not_open_second_device():
    radio = HackRF(settings(), "rx", library=FakeLibrary())
    try:
        lib = FakeLibrary()
        with pytest.raises(RuntimeError, match="half duplex"):
            HackRF(settings(), "tx", library=lib)
        assert not lib.calls
    finally:
        radio.close()


def test_receive_nonuniform_buffers_preserve_every_signed_sample():
    lib = FakeLibrary()
    radio = HackRF(settings(), "rx", library=lib)
    try:
        radio.start_rx()
        data = bytes(range(256)) * 8000
        for pos in range(0, len(data), 131070):
            assert lib.rx(data[pos:pos + 131070]) == 0
        actual = radio.read(threading.Event())
        np.testing.assert_array_equal(actual, decode_iq(data[:SAMPLE_RATE // 10 * 2]))
        remaining = radio.rx_pending
        while not radio.rx_queue.empty():
            remaining += radio.rx_queue.get_nowait()
        assert remaining == data[SAMPLE_RATE // 10 * 2:]
    finally:
        radio.close()


@pytest.mark.parametrize("kind", ["overflow", "odd", "oversize"])
def test_rx_callback_failure_is_reported_without_crossing_c_boundary(kind):
    lib = FakeLibrary()
    radio = HackRF(settings(), "rx", library=lib)
    try:
        radio.start_rx()
        if kind == "overflow":
            for _ in range(128):
                assert lib.rx(b"\0\0") == 0
            assert lib.rx(b"\0\0") != 0
        else:
            assert lib.rx(b"\0\0", 1 if kind == "odd" else 4) != 0
        with pytest.raises(RuntimeError, match="overflow|Invalid"):
            radio.read(threading.Event())
    finally:
        radio.close()


def test_tx_preserves_partial_last_buffer_and_waits_for_flush():
    lib = FakeLibrary()
    radio = HackRF(settings(), "tx", library=lib)
    ready, end = queue.Queue(), object()
    payload = bytes(range(256)) * 3000
    for begin, finish in [(0, 1234), (1234, 310008), (310008, len(payload))]:
        ready.put(payload[begin:finish])
    ready.put(end)
    try:
        radio.start_tx(ready, end, threading.Event())
        while lib.tx(size=131072) == 0:
            pass
        assert bytes(lib.output) == payload
        assert radio.tx_eof and not radio.flushed.is_set() and not radio.error
        lib.flush(None, 1)
        assert radio.flushed.is_set()
    finally:
        radio.close()
    assert "hackrf_stop_tx" in [name for name, _ in lib.calls]


def test_underrun_is_an_error_and_shutdown_attempts_close_after_stop_failure():
    lib = FakeLibrary(fail="hackrf_stop_tx")
    radio = HackRF(settings(), "tx", library=lib)
    radio.start_tx(queue.Queue(), object(), threading.Event())
    assert lib.tx() != 0
    assert "underrun" in radio.error
    with pytest.raises(RuntimeError, match="shutdown failed"):
        radio.close()
    assert [name for name, _ in lib.calls][-2:] == ["hackrf_close", "hackrf_exit"]


@pytest.mark.parametrize("failure", [None, "cancel", "source", "flush", "start"])
def test_transport_cleans_up_on_completion_cancel_and_failure(monkeypatch, failure):
    import aetv.hackrf as module
    lib = FakeLibrary(fail="hackrf_start_tx" if failure == "start" else None)
    original_start = lib.__getattr__("hackrf_start_tx")
    cancel = threading.Event()
    workers = []

    def start(*args):
        result = original_start(*args)
        if result:
            return result
        def consume():
            while lib.tx() == 0:
                if failure == "cancel":
                    cancel.set()
                cancel.wait(0.002)
            lib.flush(None, int(failure != "flush"))
        worker = threading.Thread(target=consume)
        workers.append(worker)
        worker.start()
        return 0

    lib.hackrf_start_tx = start
    monkeypatch.setattr(module, "load_library", lambda: lib)
    def chunks():
        yield np.sin(2 * np.pi * 8000 * np.arange(4800) / 48000) * .1
        if failure == "source":
            raise RuntimeError("test source failed")
    try:
        if failure in {"source", "flush", "start"}:
            with pytest.raises(RuntimeError):
                transmit_hackrf(chunks(), 48000, settings(), cancel, lambda _: None, max_seconds=1)
        else:
            assert transmit_hackrf(chunks(), 48000, settings(), cancel, lambda _: None, max_seconds=1) == (failure is None)
    finally:
        for worker in workers:
            worker.join(timeout=2)
    assert [name for name, _ in lib.calls][-2:] == ["hackrf_close", "hackrf_exit"]
    assert not any(t.name == "hackrf-modem" and t.is_alive() for t in threading.enumerate())


def test_settings_persist_and_reject_invalid_steps(tmp_path):
    from aetv.settings import load_settings, save_settings
    selected = settings(hackrf_serial="abcd", hackrf_tx_gain=47,
                        hackrf_rx_lna_gain=40, hackrf_rx_vga_gain=62)
    selected.receive_dir = str(tmp_path)
    assert not selected.validate()
    save_settings(selected, tmp_path / "settings.json")
    assert load_settings(tmp_path / "settings.json") == selected
    for change in [{"hackrf_tx_gain": 48}, {"hackrf_rx_lna_gain": 7},
                   {"hackrf_rx_vga_gain": 3}, {"hackrf_serial": "USB?"},
                   {"sdr_frequency_mhz": 1}, {"sdr_frequency_mhz": float("nan")}]:
        assert replace(selected, **change).validate()
        assert not replace(selected, **change).validate(radio_tx=False, receive=False)


def test_hackrf_controls_preserve_pluto_and_rtl_gains():
    from PySide6.QtWidgets import QApplication
    from aetv.gui.radio_panel import RadioPanel
    app = QApplication.instance() or QApplication([])
    panel = RadioPanel(settings())
    selected = []
    panel.applyRequested.connect(selected.append)
    panel.hackrf_tx_gain.setValue(12)
    panel.hackrf_lna.setValue(3)
    panel.hackrf_vga.setValue(9)
    panel.hackrf_serial.setText("abcd")
    panel.apply.click()
    values = selected[-1]
    assert values["hackrf_tx_gain"] == 12
    assert values["hackrf_rx_lna_gain"] == 24
    assert values["hackrf_rx_vga_gain"] == 18
    assert values["hackrf_serial"] == "abcd"
    assert "rtl_rx_gain" not in values and "pluto_rx_gain" not in values
    assert not panel.tx_gain.isEnabled() and not panel.rx_gain.isEnabled()
    panel.close()
    app.processEvents()


@pytest.mark.parametrize("rx_source,paused", [("hackrf", True), ("rtlsdr", False), ("pluto", False)])
def test_gui_waits_for_hackrf_receive_shutdown_before_transmitting(rx_source, paused):
    from types import SimpleNamespace
    from aetv.gui.app import MainWindow
    events = []
    window = SimpleNamespace(
        settings=replace(settings(), rx_source=rx_source),
        radio=SimpleNamespace(setEnabled=lambda _: None),
        tx=SimpleNamespace(emulating=lambda: False, allow_transmit=lambda: events.append("transmit")),
        rx=SimpleNamespace(listening=lambda: True, stop=lambda: events.append("stop_rx")),
        _log=lambda _: None,
    )
    MainWindow._on_tx_started(window)
    assert events == (["stop_rx"] if paused else ["transmit"])
    assert window._resume_rx == paused
    if paused:
        MainWindow._on_rx_stopped_for_tx(window)
        assert events == ["stop_rx", "transmit"]


def test_station_routes_prepared_clip_to_hackrf_without_cat(monkeypatch):
    from types import SimpleNamespace
    from aetv.config import AETV_MODES
    from aetv.station import Station, TxEngine, PreparedClip
    selected = settings(debug_capture=False)
    station = Station(selected)
    station.codec = SimpleNamespace(mode=AETV_MODES["AC16"])
    prepared = PreparedClip("fixture.mp4", "AC16", (np.ones(19200, np.float32),),
                            np.zeros((10, 144, 256, 3), np.uint8))
    observed = []
    def send(chunks, fs, actual, cancel, progress, *, max_seconds, diagnostics):
        assert diagnostics is engine.sdr_health
        observed.append((sum(len(x) for x in chunks), fs, actual.tx_backend, max_seconds))
        return True
    monkeypatch.setattr("aetv.hackrf.transmit_hackrf", send)
    engine = TxEngine(station)
    engine._keyed_send_stream = lambda *_: pytest.fail("HackRF must not key an audio/CAT radio")
    assert engine.transmit(prepared)
    assert observed[0][1:] == (48000, "hackrf", 1.65)
    assert engine.state.message == "HackRF off · sent"


def test_ota_validation_selects_hackrf_but_rejects_half_duplex_loopback(tmp_path):
    import json
    from aetv.gui.ota_validation import load_validation
    path = tmp_path / "trial.json"
    config = {"transmit": True, "transmitter": "hackrf", "receiver": "rtlsdr",
              "output": str(tmp_path / "output"), "hackrf_tx_gain": 12}
    path.write_text(json.dumps(config))
    _, selected = load_validation(path)
    assert selected.tx_backend == "hackrf" and selected.hackrf_tx_gain == 12
    config["receiver"] = "hackrf"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="half duplex"):
        load_validation(path)


def test_capture_feeds_existing_modem_and_raw_iq_waterfall(monkeypatch):
    from aetv.config import AETV_MODES
    from aetv.sdr import SDRCapture
    import aetv.hackrf as module
    done = threading.Event()
    writes, closed, errors = [], [], []
    class Radio:
        def __init__(self, *_):
            self.iq = (.2 * np.exp(2j * np.pi * -99000 *
                                  np.arange(SAMPLE_RATE // 10) / SAMPLE_RATE)).astype(np.complex64)
        def start_rx(self):
            pass
        def read(self, cancel):
            cancel.wait(.1)
            return self.iq
        def close(self):
            closed.append(True)
    class Sink:
        def write(self, audio):
            writes.append(audio)
            if len(audio) > 4000:
                done.set()
    monkeypatch.setattr(module, "HackRF", Radio)
    capture = SDRCapture(settings(sdr_auto_correct=False), AETV_MODES["AC16"], Sink(),
                         on_error=errors.append, on_status=lambda _: None)
    try:
        capture.start()
        assert done.wait(5)
        assert capture.preview.fs == SAMPLE_RATE
        assert capture.preview.tail(65536).dtype == np.complex64
    finally:
        capture.stop()
    assert closed and not errors
    audio = np.concatenate(writes)[1000:]
    peak = np.argmax(abs(np.fft.rfft(audio))) * 48000 / len(audio)
    assert abs(peak - 9000) < 30


def test_tx_primes_past_header_before_starting_usb(monkeypatch):
    """Encoding the first payload must not empty an already-transmitting header."""
    import time
    import aetv.hackrf as module
    lib = FakeLibrary()
    original_start = lib.__getattr__('hackrf_start_tx')
    first_payload = threading.Event()
    sent_bytes = []
    cancel = threading.Event()
    workers = []
    health = {}

    def start(*args):
        assert first_payload.is_set(), 'USB started with only the header prepared'
        original_start(*args)
        def consume():
            # Callback pacing represents the configured sample rate; capture bytes without the
            # per-element ctypes conversion in the small API unit-test fake.
            count = 0
            epoch = time.monotonic()
            while True:
                size = 262144
                buf = (C.c_uint8 * size)()
                transfer = Transfer(None, buf, size, size, None, None)
                if lib.transmit(C.pointer(transfer)) != 0:
                    break
                count += transfer.valid_length
                cancel.wait(max(0, epoch + count/(2*SAMPLE_RATE) - time.monotonic()))
            sent_bytes.append(count)
            lib.flush(None, 1)
        worker = threading.Thread(target=consume)
        workers.append(worker)
        worker.start()
        return 0

    lib.hackrf_start_tx = start
    monkeypatch.setattr(module, 'load_library', lambda: lib)
    def chunks():
        yield np.zeros(31200)  # 650 ms header, enough for the old five buffers.
        time.sleep(.25)
        first_payload.set()
        yield np.zeros(48000)
        yield np.zeros(48000)
    try:
        assert transmit_hackrf(chunks(), 48000, settings(), cancel, lambda _: None,
                               max_seconds=2.65, diagnostics=health)
    finally:
        for worker in workers:
            worker.join(timeout=5)
    assert health['completed'] and health['late_data_events'] == 0
    assert health['iq_queue_high_water'] <= 16
    assert health['produced_samples'] == sent_bytes[0] // 2
    assert abs(health['produced_samples'] / SAMPLE_RATE - 2.85) < .01


@pytest.mark.parametrize('available,result', [(False, 0), (True, -1000), (True, 0)])
def test_device_shortfall_counters_are_optional_and_use_public_abi(available, result):
    from aetv.hackrf import M0State
    assert C.sizeof(M0State) == 40
    assert M0State.num_shortfalls.offset == 16
    lib = FakeLibrary()

    def query(device, output):
        state = C.cast(output, C.POINTER(M0State)).contents
        state.num_shortfalls = 7
        state.longest_shortfall = 8064
        return result

    lib.hackrf_get_m0_state = query if available else None
    radio = HackRF(settings(), 'tx', library=lib)
    try:
        report = radio.device_state()
        assert report['available'] == (available and result == 0)
        if report['available']:
            assert report['num_shortfalls'] == 7
            assert report['longest_shortfall'] == 8064
    finally:
        radio.close()


@pytest.mark.parametrize('av', [False, True])
@pytest.mark.parametrize('prepared', [False, True])
def test_ac16_sources_reach_usb_with_nonzero_iq(monkeypatch, av, prepared):
    """Real modem/conversion/callback path; stub only capture, inference and USB."""
    import time
    from types import SimpleNamespace
    import aetv.hackrf as module
    import aetv.station as station_module
    from aetv.config import AETV_MODES
    from aetv.source import PreparedClip
    selected = replace(settings(), gops=2, debug_capture=False,
                       waveform_mode='analog_av' if av else 'video')
    station = station_module.Station(selected)
    station.codec = SimpleNamespace(mode=AETV_MODES['AC16'])
    errors = []
    engine = station_module.TxEngine(station, on_error=errors.append)
    latents = np.random.default_rng(172).normal(size=(2, 19200)).astype(np.float32)
    monkeypatch.setattr(engine, '_live_webcam_gops', lambda *a: iter(latents))
    monkeypatch.setattr(station_module, 'read_video_audio', lambda *a, **k: np.zeros(16000))
    monkeypatch.setattr(station_module, 'open_input_stream', lambda *a, **k: (None, 8000))
    lib = FakeLibrary()
    original_start = lib.__getattr__('hackrf_start_tx')
    workers, blocks, counts = [], [], []

    def start(*args):
        original_start(*args)
        def consume():
            total = 0
            epoch = time.monotonic()
            while True:
                buf = (C.c_uint8 * 262144)()
                transfer = Transfer(None, buf, len(buf), len(buf), None, None)
                if lib.transmit(C.pointer(transfer)) != 0:
                    break
                values = np.ctypeslib.as_array(buf)[:transfer.valid_length].view(np.int8)
                blocks.append(bool(np.any(values)))
                total += transfer.valid_length
                engine._cancel.wait(max(0, epoch + total/(2*SAMPLE_RATE) - time.monotonic()))
            counts.append(total)
            lib.flush(None, 1)
        thread = threading.Thread(target=consume)
        workers.append(thread)
        thread.start()
        return 0

    lib.hackrf_start_tx = start
    monkeypatch.setattr(module, 'load_library', lambda: lib)
    source = (PreparedClip('unused.mp4', 'AC16', tuple(latents),
                           np.zeros((6, 144, 256, 3), np.uint8)) if prepared else 'webcam')
    try:
        assert engine.transmit(source), errors
    finally:
        for thread in workers:
            thread.join(timeout=5)
    assert not errors
    assert engine.sdr_health['completed']
    assert engine.sdr_health['produced_samples'] == counts[0] // 2
    assert sum(blocks) > 100
    assert engine.sdr_health['late_data_events'] == 0


def test_conversion_initialization_error_reaches_caller_before_radio_open(monkeypatch):
    import aetv.sdr_dsp as dsp
    import aetv.hackrf as module

    def fail(*args, **kwargs):
        raise ValueError('test conversion setup failed')

    monkeypatch.setattr(dsp, 'ModemToIQ', fail)
    monkeypatch.setattr(module, 'HackRF', lambda *a, **k: pytest.fail('radio must stay closed'))
    health = {}
    with pytest.raises(ValueError, match='test conversion setup failed'):
        transmit_hackrf(iter(()), 48000, settings(), threading.Event(), lambda _: None,
                       max_seconds=1, diagnostics=health)
    assert health['producer_error'] == 'test conversion setup failed'
    assert not health['completed']
