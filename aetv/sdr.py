"""Direct SDR transports with bounded queues and explicit transmitter shutdown.

Radio drivers are imported/opened only when the operator starts a radio path.
All receive conversion uses captured IQ; source video is never an input.
"""

from __future__ import annotations

import queue
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio_io import StreamResampler, resample_ratio
from .hfchannel import _active_signal_power
from .sdr_dsp import IQDecimator, IQToModem, ModemToIQ, estimate_signal_offset


def rtl_executable() -> str:
    root = Path(__file__).parent / "bin"
    for folder in (root / "rtlsdr", root):
        for name in ("rtl_sdr", "rtl_sdr.exe"):
            bundled = folder / name
            if bundled.is_file():
                return str(bundled)
    for name in ("rtl_sdr", "rtl_sdr.exe"):
        if found := shutil.which(name):
            return found
    raise RuntimeError(
        "RTL-SDR runtime not found. Re-extract the complete portable package; "
        "for a source installation, install rtl-sdr (rtl_sdr on PATH)."
    )


def _rtl_process_options():
    options = {}
    if sys.platform == "win32":
        environment = os.environ.copy()
        if getattr(sys, "frozen", False):
            # DLL search directories registered in Python do not propagate to
            # rtl_sdr.exe. Let the child find the packaged VC runtime too.
            environment["PATH"] = (
                str(sys._MEIPASS) + os.pathsep + environment.get("PATH", "")
            )
        options.update(env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
    return options


def sdr_runtime_smoke():
    """Load native transports without opening a radio or transmitting."""
    import adi
    import iio
    from .modem_smoke import acquisition_smoke
    from .hackrf import runtime_smoke as hackrf_smoke
    from .hackrf_smoke import transport_smoke

    executable = rtl_executable()
    result = subprocess.run(
        [executable, "-h"], capture_output=True, timeout=15, **_rtl_process_options()
    )
    help_text = (result.stdout + result.stderr).decode(errors="replace")
    if "rtl_sdr" not in help_text.lower() or "sample" not in help_text.lower():
        raise RuntimeError(
            f"Bundled rtl_sdr could not start ({result.returncode}): {help_text}"
        )
    if not {"usb", "ip", "xml"}.issubset(iio.backends):
        raise RuntimeError(f"libiio is missing Pluto backends: {iio.backends}")
    if not callable(adi.Pluto):
        raise RuntimeError("pyadi-iio does not expose Pluto")
    # Exercise libxml as well as libiio; imports alone miss some DLL failures.
    with tempfile.TemporaryDirectory(prefix="aetv-sdr-smoke-") as folder:
        xml = Path(folder) / "context.xml"
        xml.write_text(
            '<?xml version="1.0"?><!DOCTYPE context ['
            '<!ELEMENT context EMPTY><!ATTLIST context name CDATA #REQUIRED '
            'description CDATA #IMPLIED>]><context name="xml" description="AETV smoke"/>',
            encoding="ascii",
        )
        context = iio.XMLContext(str(xml))
    if context.devices:
        raise RuntimeError("Unexpected devices in the empty XML fixture")
    library = str(iio._lib._name)
    frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        root = Path(sys._MEIPASS).resolve()
        for path in (executable, library):
            if not Path(path).resolve().is_relative_to(root):
                raise RuntimeError(f"SDR runtime escaped the portable package: {path}")
    return {
        "passed": True,
        "frozen": frozen,
        "rtl_executable": executable,
        "rtl_help_returncode": result.returncode,
        "libiio": library,
        "libiio_version": iio.version,
        "libiio_backends": iio.backends,
        "pluto_class": f"{adi.Pluto.__module__}.{adi.Pluto.__name__}",
        "xml_context": True,
        "acquisition": acquisition_smoke(),
        "hackrf": hackrf_smoke(),
        "hackrf_transport": transport_smoke(),
    }


def open_pluto(uri):
    try:
        import adi

        radio = adi.Pluto(uri=uri)
        radio._ctx.set_timeout(2000)
        return radio
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"Cannot open Pluto at {uri}: {error}. Check the radio connection and "
            "USB setup in docs/sdr-portable-setup.md. For a source installation, install libiio."
        ) from error


def stop_pluto_tx(radio):
    """Attempt every shutdown step, even after a failed USB/network operation."""
    errors = []
    for action in (
        lambda: (
            radio._ctrl.find_channel("altvoltage1", True)
            .attrs["powerdown"]
            .__setattr__("value", "1")
        ),
        lambda: setattr(radio, "tx_hardwaregain_chan0", -89.75),
        radio.tx_destroy_buffer,
    ):
        try:
            action()
        except Exception as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("Pluto shutdown failed: " + "; ".join(errors))


class IQPreview:
    """A bounded, thread-safe raw-IQ window for the RF waterfall."""

    def __init__(self, fs, frequency_hz, mode):
        self.fs = fs
        self.frequency_hz = frequency_hz
        self.lo_hz = frequency_hz + 100000
        self.bandwidth_hz = mode.geometry.tx_bandpass[1] - mode.geometry.tx_bandpass[0]
        self._values = np.empty(0, np.complex64)
        self._lock = threading.Lock()
        self.sequence = 0

    def write(self, values):
        with self._lock:
            self._values = np.asarray(values[-65536:], np.complex64).copy()
            self.sequence += 1

    def tail(self, n):
        with self._lock:
            return self._values[-n:].copy()


class SDRCapture:
    def __init__(self, settings, mode, ring, *, on_error, on_status):
        self.settings = replace(settings)
        self.mode, self.ring = mode, ring
        self.on_error, self.on_status = on_error, on_status
        self.rate = 2400000 if settings.rx_source == "pluto" else 960000
        self._hackrf = None
        self._decimator = None
        if settings.rx_source == "hackrf":
            from .hackrf import SAMPLE_RATE
            self.rate = SAMPLE_RATE
            self._decimator = IQDecimator(self.rate // 960000)
        self.conversion_rate = 960000 if self._decimator else self.rate
        self.preview = IQPreview(self.rate, settings.sdr_frequency_mhz * 1e6, mode)
        self._stop = threading.Event()
        self._radio = None
        self._process = None
        self._threads = []
        self._queue = queue.Queue(maxsize=80)
        self.error = ""

    def start(self):
        settings = self.settings
        if settings.rx_source == "pluto":
            self._radio = open_pluto(settings.pluto_uri)
            self._radio.sample_rate = self.rate
            self._radio.rx_lo = round(self.preview.lo_hz)
            self._radio.rx_rf_bandwidth = 600000
            self._radio.gain_control_mode_chan0 = "manual"
            self._radio.rx_hardwaregain_chan0 = settings.pluto_rx_gain
            self._radio.rx_buffer_size = self.rate // 10
        elif settings.rx_source == "hackrf":
            from .hackrf import HackRF
            self._hackrf = HackRF(settings, "rx")
            try:
                self._hackrf.start_rx()
            except Exception:
                self._hackrf.close()
                self._hackrf = None
                raise
            self.on_status(
                f"HackRF: {self.rate} S/s, LNA {settings.hackrf_rx_lna_gain} dB, "
                f"VGA {settings.hackrf_rx_vga_gain} dB, RF amp/bias off"
            )
        else:
            command = [
                rtl_executable(),
                "-d",
                settings.rtl_serial,
                "-f",
                str(round(self.preview.lo_hz)),
                "-s",
                str(self.rate),
                "-g",
                str(settings.rtl_rx_gain),
                "-b",
                "65536",
                "-",
            ]
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_rtl_process_options(),
            )
            self._worker("rtl-log", self._read_log)
        self._worker("sdr-convert", self._convert)
        self._worker("sdr-capture", self._capture)

    def _worker(self, name, function):
        def guarded():
            try:
                function()
            except Exception as error:
                if not self._stop.is_set():
                    self.error = str(error)
                    self.on_error(self.error)
                    self._stop.set()

        thread = threading.Thread(target=guarded, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _read_log(self):
        for raw in self._process.stderr:
            line = raw.decode(errors="replace").strip()
            if line:
                self.on_status("RTL: " + line)

    def _capture(self):
        while not self._stop.is_set():
            if self._hackrf is not None:
                iq = self._hackrf.read(self._stop)
            elif self._radio is not None:
                iq = np.asarray(self._radio.rx(), np.complex64) / 2048
            else:
                data = self._process.stdout.read(self.rate // 10 * 2)
                if not data:
                    raise RuntimeError(
                        "RTL-SDR stopped producing IQ; check the selected serial and USB driver"
                    )
                raw = np.frombuffer(data, np.uint8).astype(np.float32)
                if len(raw) % 2:
                    raise RuntimeError("Truncated RTL IQ sample")
                iq = ((raw[::2] - 127.5) + 1j * (raw[1::2] - 127.5)) / 128
            self.preview.write(iq)
            if self._decimator is not None:
                iq = self._decimator.feed(iq)
            try:
                self._queue.put_nowait(iq)
            except queue.Full as error:
                raise RuntimeError(
                    "SDR processing exceeded its 8-second IQ queue; stop and restart reception"
                ) from error

    def _convert(self):
        offset = -100000 + self.settings.sdr_rx_correction_hz
        adapter = IQToModem(self.conversion_rate, offset, self.mode.geometry.fcenter_hz)
        resample = StreamResampler(*resample_ratio(48000, self.mode.geometry.fs))
        calibration = []
        estimates = []
        auto = self.settings.sdr_auto_correct and self.mode.name == "AC16"
        while not self._stop.is_set():
            try:
                iq = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if auto:
                calibration.append(iq)
                calibration = calibration[-40:]
                if len(calibration) < 10:
                    continue
                try:
                    measured = estimate_signal_offset(
                        np.concatenate(calibration[-10:]), self.conversion_rate
                    )
                except ValueError:
                    estimates.clear()
                    continue
                # Partial preambles have biased spectral centers. Require five
                # stable payload-like spectra before committing to a correction.
                if not 14400 <= measured["obw99_hz"] <= 15600:
                    estimates.clear()
                    continue
                estimates.append(measured["offset_hz"])
                estimates = estimates[-5:]
                if len(estimates) < 5 or np.ptp(estimates) > 150:
                    continue
                offset = float(np.median(estimates))
                adapter = IQToModem(self.conversion_rate, offset, self.mode.geometry.fcenter_hz)
                resample = StreamResampler(
                    *resample_ratio(48000, self.mode.geometry.fs)
                )
                iq = np.concatenate(calibration)
                auto = False
                calibration.clear()
                self.on_status(
                    f"SDR received-only frequency correction: {offset + 100000:+.0f} Hz"
                )
            self.ring.write(resample(adapter.feed(iq)))

    def stop(self):
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        for thread in self._threads:
            thread.join(timeout=3)
        if self._hackrf is not None:
            self._hackrf.close()
            self._hackrf = None
        if self._radio is not None:
            self._radio.rx_destroy_buffer()
            self._radio = None
        if self._process is not None:
            self._process.stdout.close()
            self._process.stderr.close()
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("SDR worker did not stop within its driver timeout")


def transmit_pluto(chunks, fs, settings, cancel, on_progress, *, max_seconds):
    """Consume live modem chunks through finite DMA buffers, with a watchdog."""
    settings = replace(settings)
    ready = queue.Queue(maxsize=24)
    stopped = threading.Event()
    end = object()
    radio = None
    deadline = time.monotonic() + max_seconds + 25

    def put(value):
        while not (stopped.is_set() or cancel.is_set()):
            try:
                ready.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        from .config import AETV_MODES

        adapter = ModemToIQ(center_hz=AETV_MODES[settings.mode].geometry.fcenter_hz)
        resample = StreamResampler(*resample_ratio(fs, 48000))
        try:
            for audio in chunks:
                if stopped.is_set() or cancel.is_set():
                    break
                audio = resample(audio)
                power = _active_signal_power(audio)
                audio = audio * (
                    0.2 * settings.tx_level / 0.7 / max(np.sqrt(power), 1e-12)
                )
                for pos in range(0, len(audio), 4800):
                    if stopped.is_set() or cancel.is_set():
                        break
                    put(adapter.feed(audio[pos : pos + 4800]))
        except Exception as error:
            put(error)
        finally:
            put(end)

    producer = threading.Thread(target=produce, name="pluto-modem", daemon=True)
    producer.start()
    pending = np.empty(0, np.complex64)
    samples = 0
    complete = False
    try:
        while not cancel.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("Pluto TX exceeded its duration watchdog")
            try:
                audio = ready.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio is end:
                complete = True
                break
            if isinstance(audio, Exception):
                raise audio
            if radio is None:
                radio = open_pluto(settings.pluto_uri)
                radio._ctrl.find_channel("altvoltage1", True).attrs[
                    "powerdown"
                ].value = "1"
                radio.sample_rate = 2400000
                radio.tx_lo = round(settings.sdr_frequency_mhz * 1e6 - 100000)
                radio.tx_rf_bandwidth = 600000
                radio.tx_cyclic_buffer = False
                radio.disable_dds()
                radio.tx_hardwaregain_chan0 = settings.pluto_tx_gain
            pending = np.concatenate((pending, audio))
            while len(pending) >= 240000 and not cancel.is_set():
                if samples == 0:
                    radio._ctrl.find_channel("altvoltage1", True).attrs[
                        "powerdown"
                    ].value = "0"
                radio.tx(pending[:240000] * 16384)
                pending = pending[240000:]
                samples += 240000
                on_progress(min(1, samples / 2400000 / max(1, max_seconds)))
        if radio is not None and complete and not cancel.is_set():
            if len(pending):
                radio.tx(np.pad(pending, (0, 240000 - len(pending))) * 16384)
            for _ in range(4):
                radio.tx(np.zeros(240000, np.complex64))
            cancel.wait(0.4)
        return complete and not cancel.is_set()
    finally:
        stopped.set()
        try:
            if radio is not None:
                stop_pluto_tx(radio)
        finally:
            cancel.set()
            producer.join(timeout=5)
            if producer.is_alive():
                raise RuntimeError("Pluto producer did not stop within five seconds")
