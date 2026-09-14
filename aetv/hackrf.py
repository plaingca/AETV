"""Bounded signed-IQ streaming through libhackrf's stable C API.

Callbacks only copy buffers and signal events. All device control, including
stop/close after cancellation or failure, runs outside the USB callback.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import queue
import sys
import threading
import time
from pathlib import Path

from .analog_av import waveform_center_hz

import numpy as np

SAMPLE_RATE = 9600000  # >=8 MHz ADC/DAC recommendation; integer 48 kHz ratio.
FILTER_BANDWIDTH = 1750000
_device_lock = threading.Lock()  # One shared TX/RX selector: half duplex.


class Transfer(C.Structure):
    _fields_ = [
        ("device", C.c_void_p), ("buffer", C.POINTER(C.c_uint8)),
        ("buffer_length", C.c_int), ("valid_length", C.c_int),
        ("rx_ctx", C.c_void_p), ("tx_ctx", C.c_void_p),
    ]


SampleCallback = C.CFUNCTYPE(C.c_int, C.POINTER(Transfer))
FlushCallback = C.CFUNCTYPE(None, C.c_void_p, C.c_int)


def load_library():
    package = Path(__file__).parent / "bin" / "hackrf"
    roots = [package]
    if getattr(sys, "frozen", False):
        roots.append(Path(sys._MEIPASS))
    candidates = [root / name for root in roots for name in
                  ("libhackrf.dll", "hackrf.dll", "libhackrf.so.0", "libhackrf.dylib")]
    path = next((str(p) for p in candidates if p.is_file()), None)
    if path is None and not getattr(sys, "frozen", False):
        path = ctypes.util.find_library("hackrf")
    if not path:
        raise RuntimeError("HackRF runtime missing. Re-extract the portable package; "
                           "source installations need libhackrf.")
    try:
        lib = C.CDLL(path)
        signatures = {
            "init": ([], C.c_int), "exit": ([], C.c_int),
            "library_version": ([], C.c_char_p),
            "library_release": ([], C.c_char_p),
            "error_name": ([C.c_int], C.c_char_p),
            "open_by_serial": ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int),
            "close": ([C.c_void_p], C.c_int),
            "set_sample_rate": ([C.c_void_p, C.c_double], C.c_int),
            "set_freq": ([C.c_void_p, C.c_uint64], C.c_int),
            "set_baseband_filter_bandwidth": ([C.c_void_p, C.c_uint32], C.c_int),
            "set_amp_enable": ([C.c_void_p, C.c_uint8], C.c_int),
            "set_antenna_enable": ([C.c_void_p, C.c_uint8], C.c_int),
            "set_lna_gain": ([C.c_void_p, C.c_uint32], C.c_int),
            "set_vga_gain": ([C.c_void_p, C.c_uint32], C.c_int),
            "set_txvga_gain": ([C.c_void_p, C.c_uint32], C.c_int),
            "start_rx": ([C.c_void_p, SampleCallback, C.c_void_p], C.c_int),
            "start_tx": ([C.c_void_p, SampleCallback, C.c_void_p], C.c_int),
            "stop_rx": ([C.c_void_p], C.c_int),
            "stop_tx": ([C.c_void_p], C.c_int),
            "is_streaming": ([C.c_void_p], C.c_int),
            "enable_tx_flush": ([C.c_void_p, FlushCallback, C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, "hackrf_" + name)
            function.argtypes, function.restype = args, result
        return lib
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"Cannot load HackRF runtime {path}: {error}") from error


def runtime_smoke():
    """Resolve the complete API without enumerating/opening any USB device."""
    lib = load_library()
    return {"library": str(lib._name),
            "version": lib.hackrf_library_version().decode(),
            "release": lib.hackrf_library_release().decode(),
            "device_opened": False}


def encode_iq(iq):
    values = np.asarray(iq, np.complex64).reshape(-1)
    components = values.view(np.float32)
    if not np.all(np.isfinite(components)) or np.any(np.abs(components) >= 1):
        raise ValueError("HackRF IQ exceeds signed 8-bit range; reduce TX level")
    return np.clip(np.rint(components * 128), -128, 127).astype(np.int8).tobytes()


def decode_iq(data):
    if len(data) % 2:
        raise ValueError("Truncated HackRF IQ sample")
    raw = np.frombuffer(data, np.int8).astype(np.float32) / 128
    return raw.view(np.complex64)


class HackRF:
    def __init__(self, settings, direction, *, library=None):
        if direction not in {"rx", "tx"}:
            raise ValueError("Unknown HackRF direction")
        if not _device_lock.acquire(blocking=False):
            raise RuntimeError("HackRF is half duplex; stop its receiver before transmitting")
        self.lib = None
        self.device = C.c_void_p()
        self.direction = direction
        self.initialized = self.started = self.closed = False
        self.error = ""
        self.callbacks = []
        self.flushed = threading.Event()
        self.rx_queue = queue.Queue(maxsize=128)
        self.rx_pending = b""
        try:
            self.lib = library if library is not None else load_library()
            self.check("init")
            self.initialized = True
            serial = settings.hackrf_serial.strip().encode("ascii") or None
            self.check("open_by_serial", serial, C.byref(self.device))
            self.check("set_sample_rate", self.device, SAMPLE_RATE)
            self.check("set_baseband_filter_bandwidth", self.device, FILTER_BANDWIDTH)
            offset = -100000 if direction == "tx" else 100000
            self.check("set_freq", self.device, round(settings.sdr_frequency_mhz * 1e6 + offset))
            # RF amplifier and antenna bias power deliberately remain off.
            self.check("set_amp_enable", self.device, 0)
            self.check("set_antenna_enable", self.device, 0)
            if direction == "tx":
                self.check("set_txvga_gain", self.device, settings.hackrf_tx_gain)
            else:
                self.check("set_lna_gain", self.device, settings.hackrf_rx_lna_gain)
                self.check("set_vga_gain", self.device, settings.hackrf_rx_vga_gain)
        except Exception:
            self.close()
            raise

    def check(self, name, *args):
        result = getattr(self.lib, "hackrf_" + name)(*args)
        if result != 0:
            reason = self.lib.hackrf_error_name(result).decode(errors="replace")
            raise RuntimeError(f"HackRF {name}: {reason} ({result}). "
                               "Check the serial, USB driver, and other SDR applications.")

    def start_rx(self):
        def receive(pointer):
            try:
                transfer = pointer.contents
                if not 0 < transfer.valid_length <= transfer.buffer_length or transfer.valid_length % 2:
                    raise ValueError("Invalid HackRF receive buffer")
                self.rx_queue.put_nowait(C.string_at(transfer.buffer, transfer.valid_length))
                return 0
            except queue.Full:
                self.error = "HackRF receive USB queue overflow; stop and restart reception"
            except Exception as error:
                self.error = str(error)
            return -1
        callback = SampleCallback(receive)
        self.callbacks.append(callback)
        self.started = True
        self.check("start_rx", self.device, callback, None)

    def read(self, cancel):
        target = SAMPLE_RATE // 10 * 2
        parts, count = [self.rx_pending], len(self.rx_pending)
        deadline = time.monotonic() + 3
        while count < target and not cancel.is_set():
            if self.error:
                raise RuntimeError(self.error)
            try:
                block = self.rx_queue.get(timeout=0.1)
                parts.append(block)
                count += len(block)
            except queue.Empty:
                if self.lib.hackrf_is_streaming(self.device) != 1 or time.monotonic() > deadline:
                    raise RuntimeError("HackRF stopped producing IQ; check USB connection")
        joined = b"".join(parts)
        self.rx_pending = joined[target:]
        return decode_iq(joined[:target])

    def start_tx(self, ready, end, cancel):
        pending = memoryview(b"")
        self.tx_eof = False
        self.tx_samples = 0

        def transmit(pointer):
            nonlocal pending
            transfer = pointer.contents
            transfer.valid_length = 0
            if cancel.is_set() or self.tx_eof:
                return -1
            try:
                if transfer.buffer_length <= 0 or transfer.buffer_length % 2:
                    raise ValueError("Invalid HackRF transmit buffer")
                while transfer.valid_length < transfer.buffer_length:
                    if not pending:
                        block = ready.get_nowait()
                        if block is end:
                            self.tx_eof = True
                            break
                        if isinstance(block, Exception):
                            raise block
                        pending = memoryview(block)
                    count = min(len(pending), transfer.buffer_length - transfer.valid_length)
                    C.memmove(C.addressof(transfer.buffer.contents) + transfer.valid_length,
                              pending[:count].tobytes(), count)
                    transfer.valid_length += count
                    pending = pending[count:]
                self.tx_samples += transfer.valid_length // 2
                return 0 if transfer.valid_length else -1
            except queue.Empty:
                self.error = "HackRF TX underrun: inference/modem could not supply IQ in real time"
            except Exception as error:
                self.error = str(error)
            return -1

        def flushed(_context, success):
            if not success:
                self.error = self.error or "HackRF USB transmit flush failed"
            self.flushed.set()

        callback, flush = SampleCallback(transmit), FlushCallback(flushed)
        self.callbacks.extend((callback, flush))
        self.check("enable_tx_flush", self.device, flush, None)
        self.started = True
        self.check("start_tx", self.device, callback, None)

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        try:
            actions = []
            if self.device.value:
                if self.started:
                    actions.append(("stop_" + self.direction, self.device))
                if self.direction == "tx":
                    actions.append(("set_txvga_gain", self.device, 0))
                actions.extend([("set_amp_enable", self.device, 0),
                                ("set_antenna_enable", self.device, 0),
                                ("close", self.device)])
            if self.initialized:
                actions.append(("exit",))
            for action in actions:
                try:
                    self.check(*action)
                except Exception as error:
                    errors.append(str(error))
        finally:
            _device_lock.release()
        if errors:
            raise RuntimeError("HackRF shutdown failed: " + "; ".join(errors))


def transmit_hackrf(chunks, fs, settings, cancel, on_progress, *, max_seconds):
    from dataclasses import replace
    from .audio_io import StreamResampler, resample_ratio
    from .hfchannel import _active_signal_power
    from .sdr_dsp import ModemToIQ

    settings = replace(settings)
    ready = queue.Queue(maxsize=16)
    stopped, produced = threading.Event(), threading.Event()
    end = object()
    deadline = time.monotonic() + max_seconds + 25
    radio = None

    def put(value):
        while not (stopped.is_set() or cancel.is_set()):
            try:
                ready.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        adapter = ModemToIQ(SAMPLE_RATE, center_hz=waveform_center_hz(settings))
        resample = StreamResampler(*resample_ratio(fs, 48000))
        count = 0
        try:
            for audio in chunks:
                if stopped.is_set() or cancel.is_set():
                    break
                audio = resample(audio)
                count += len(audio)
                if count > round((max_seconds + 0.1) * 48000):
                    raise RuntimeError("HackRF source exceeded the requested duration")
                power = _active_signal_power(audio)
                audio = audio * (0.2 * settings.tx_level / 0.7 / max(np.sqrt(power), 1e-12))
                for pos in range(0, len(audio), 4800):
                    if stopped.is_set() or cancel.is_set():
                        return
                    put(encode_iq(adapter.feed(audio[pos:pos + 4800])))
            # Drain the resampler and sideband FIR tail before the USB flush.
            put(encode_iq(adapter.feed(resample(np.zeros(9600)))))
        except Exception as error:
            put(error)
        finally:
            put(end)
            produced.set()

    producer = threading.Thread(target=produce, name="hackrf-modem", daemon=True)
    producer.start()
    try:
        # Half a second of prepared IQ absorbs callback and encoder jitter.
        while ready.qsize() < 5 and not produced.is_set() and not cancel.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("HackRF TX preparation exceeded its duration watchdog")
            cancel.wait(0.02)
        if cancel.is_set():
            return False
        radio = HackRF(settings, "tx")
        radio.start_tx(ready, end, cancel)
        while not cancel.is_set():
            if radio.error:
                raise RuntimeError(radio.error)
            if radio.flushed.wait(0.02):
                if radio.error:
                    raise RuntimeError(radio.error)
                return radio.tx_eof
            if time.monotonic() > deadline:
                raise TimeoutError("HackRF TX exceeded its duration watchdog")
            on_progress(min(1, radio.tx_samples / SAMPLE_RATE / max(1, max_seconds)))
        return False
    finally:
        stopped.set()
        cancel.set()
        try:
            if radio is not None:
                radio.close()
        finally:
            producer.join(timeout=5)
            if producer.is_alive():
                raise RuntimeError("HackRF producer did not stop within five seconds")
