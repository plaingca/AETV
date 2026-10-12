"""CAT / PTT backends for the ham station.

Anything used as a PTT object exposes `set_ptt(bool)` and optional
`close()` / `describe()`. The transmit engine keys inside try/finally
and a watchdog; backends must still be safe to call twice with False.

Backends:
- none: VOX / audio-only, no radio commands
- hamlib: Hamlib C API loaded in-process (no daemon)
- rigctld: legacy Hamlib TCP compatibility
- flex: FlexRadio 6000 SmartSDR TCP, PTT only (tune the slice yourself)
- rts / dtr: serial-line PTT for a SignaLink-style interface
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import math
import re
import threading
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .flex import FlexVitaSession


class CatError(RuntimeError):
    pass


class _CommandRejected(CatError):
    """A complete, valid error reply; the transport remains synchronized."""


class NullPtt:
    """No radio control. Use VOX or key the rig by hand."""

    def set_ptt(self, on: bool) -> None:
        return None

    def describe(self) -> str:
        return "no CAT (VOX / manual PTT)"

    def close(self) -> None:
        return None


def _frequency_hz(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CatError("frequency must be finite and positive") from error
    if not math.isfinite(value) or value <= 0:
        raise CatError("frequency must be finite and positive")
    return value


class RigctldClient:
    """Serialized classic rigctld protocol with bounded, complete reply framing.

    Failed transactions invalidate the socket. Never replay a key-on or tuning
    command: its effect may already have occurred. Only an explicit PTT-off may
    reconnect, so a watchdog can still attempt to release a disconnected rig.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4532, timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        if not host or not 1 <= self.port <= 65535:
            raise CatError("rigctld requires a host and a port between 1 and 65535")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise CatError("rigctld timeout must be finite and positive")
        self._lock = threading.RLock()
        self.sock = None
        self._buffer = b""
        self._closed = False
        self._connect()

    def _connect(self) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)
        except OSError as error:
            self._disconnect()
            raise CatError(f"cannot connect to rigctld {self.host}:{self.port}: {error}") from error

    def _disconnect(self) -> None:
        sock, self.sock = self.sock, None
        self._buffer = b""
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _readline(self, deadline: float) -> str:
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("reply deadline exceeded")
            self.sock.settimeout(remaining)
            part = self.sock.recv(4096)
            if not part:
                raise CatError("rigctld disconnected before completing its reply")
            self._buffer += part
            if len(self._buffer) > 8192:
                raise CatError("rigctld reply exceeds size limit")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("ascii").strip()

    def _command(self, line: str) -> str:
        with self._lock:
            if self._closed or self.sock is None:
                raise CatError("rigctld connection is closed; reopen the radio connection")
            try:
                deadline = time.monotonic() + self.timeout
                self.sock.settimeout(self.timeout)
                self.sock.sendall((line + "\n").encode("ascii"))
                reply = self._readline(deadline)
                if reply.startswith("RPRT"):
                    match = re.fullmatch(r"RPRT (-?\d+)", reply)
                    if match is None:
                        raise CatError(f"malformed rigctld status: {reply!r}")
                    code = int(match[1])
                    if code != 0:
                        raise _CommandRejected(f"rigctld rejected {line!r}: {reply}")
                    if line in {"f", "m", "t"}:
                        raise CatError(f"rigctld returned no value for {line!r}")
                elif line == "f":
                    _frequency_hz(reply)
                elif line == "t":
                    if reply not in {"0", "1", "2", "3"}:
                        raise CatError(f"malformed rigctld PTT state: {reply!r}")
                elif line == "m":
                    # Classic get_mode returns BOTH mode and passband, possibly
                    # in different TCP packets. Leaving width unread corrupts
                    # the acknowledgement of the next PTT command.
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", reply):
                        raise CatError(f"malformed rigctld mode: {reply!r}")
                    width = self._readline(deadline)
                    if not re.fullmatch(r"-?\d+", width):
                        raise CatError(f"malformed rigctld passband: {width!r}")
                else:
                    raise CatError(f"expected rigctld status, received {reply!r}")
                return reply
            except _CommandRejected:
                raise
            except (OSError, UnicodeError, CatError) as error:
                self._disconnect()
                if isinstance(error, CatError):
                    raise
                raise CatError(f"rigctld command {line!r} failed: {error}") from error

    def set_ptt(self, on: bool) -> None:
        with self._lock:
            if self._closed:
                if not on:
                    return
                raise CatError("rigctld connection is closed")
            if not on and self.sock is None:
                self._connect()
            try:
                self._command(f"T {1 if on else 0}")
            except _CommandRejected:
                raise
            except CatError:
                if on:
                    raise
                # OFF is idempotent. One fresh attempt also handles an idle
                # connection that the daemon dropped without notifying us.
                self._connect()
                self._command("T 0")

    def set_frequency_hz(self, frequency_hz: float) -> None:
        self._command(f"F {_frequency_hz(frequency_hz):.0f}")

    def get_frequency_hz(self) -> float:
        return _frequency_hz(self._command("f"))

    def get_mode(self) -> str:
        return self._command("m")

    def get_ptt(self) -> bool:
        """Read the backend's PTT state, not a measurement of physical RF."""
        return self._command("t") != "0"

    def describe(self) -> str:
        try:
            hz = self.get_frequency_hz()
            mode = self.get_mode()
            return f"rigctld {self.host}:{self.port}  {hz / 1e6:.6f} MHz {mode}"
        except CatError:
            return f"rigctld {self.host}:{self.port}"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.set_ptt(False)
            except CatError:
                pass
            finally:
                self._closed = True
                self._disconnect()


@dataclass(frozen=True)
class HamlibModel:
    model_id: int
    manufacturer: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.manufacturer} {self.model} ({self.model_id})"


def _rigctl_path() -> str | None:
    bundled = Path(__file__).with_name("bin") / ("rigctl.exe" if os.name == "nt" else "rigctl")
    return str(bundled) if bundled.is_file() else shutil.which("rigctl")


def list_hamlib_models() -> list[HamlibModel]:
    """Return Hamlib's installed rig list for the settings picker."""
    executable = _rigctl_path()
    if not executable:
        return []
    try:
        proc = subprocess.run(
            [executable, "-l"], capture_output=True, text=True, timeout=8, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    models: list[HamlibModel] = []
    header = next((line for line in proc.stdout.splitlines() if "Mfg" in line and "Model" in line and "Version" in line), "")
    columns = [header.find(name) for name in ("Mfg", "Model", "Version")]
    for line in proc.stdout.splitlines():
        fields = line.split(None, 4)
        if not fields or not fields[0].isdigit() or len(fields) < 3:
            continue
        if columns[0] >= 0 and columns == sorted(columns):
            manufacturer = line[columns[0]:columns[1]].strip()
            model = line[columns[1]:columns[2]].strip()
        else:
            manufacturer, model = fields[1:3]
        models.append(HamlibModel(int(fields[0]), manufacturer, model))
    return models


def _find_hamlib_library() -> str:
    names = ["hamlib", "libhamlib-4", "libhamlib"]
    candidates: list[str] = []
    local_bin = Path(__file__).with_name("bin")
    for name in ("libhamlib-4.dll", "hamlib.dll", "libhamlib.so.4", "libhamlib.dylib"):
        candidates.append(str(local_bin / name))
    executable = _rigctl_path()
    if executable:
        exe_dir = Path(executable).resolve().parent
        candidates.extend(str(exe_dir / name) for name in ("libhamlib-4.dll", "hamlib.dll"))
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend(
            str(path) for path in program_files.glob("FreeDV*/bin/libhamlib-4.dll")
        )
    candidates.extend(filter(None, (ctypes.util.find_library(name) for name in names)))
    # Let the platform loader honor LD_LIBRARY_PATH / its normal search path,
    # including portable runtimes that are deliberately absent from ldconfig.
    candidates.extend(
        ("libhamlib-4.dll", "hamlib.dll") if os.name == "nt" else
        ("libhamlib.so.4", "libhamlib.dylib")
    )
    for candidate in candidates:
        if Path(candidate).is_file() or not Path(candidate).is_absolute():
            try:
                if os.name == "nt" and Path(candidate).is_absolute():
                    with os.add_dll_directory(str(Path(candidate).parent)):
                        ctypes.CDLL(candidate)
                else:
                    ctypes.CDLL(candidate)
                return candidate
            except OSError:
                continue
    raise CatError(
        "Hamlib library not found. Install Hamlib 4.x (rigctl), or place its library in aetv/bin."
    )


class HamlibDirect:
    """In-process Hamlib rig control. No separately managed rigctld daemon."""

    # Hamlib's special value meaning the current VFO.
    _VFO_CURR = 0x20000000

    def __init__(self, model: int, device: str, baud: int = 0):
        if int(model) <= 0:
            raise CatError("choose a Hamlib radio model")
        if not device and int(model) != 1:
            raise CatError("choose the radio's serial or network device")
        self._lock = threading.RLock()
        self.model = int(model)
        self.device = device
        self.baud = int(baud)
        if self.baud < 0:
            raise CatError("Hamlib baud rate cannot be negative")
        if "\x00" in device:
            raise CatError("Hamlib device contains a null character")
        library = _find_hamlib_library()
        self._dll_dir = (
            os.add_dll_directory(str(Path(library).parent))
            if os.name == "nt" and Path(library).is_absolute() else None
        )
        self.rig = None
        try:
            self.lib = ctypes.CDLL(library)
            self._configure_api()
            self.rig = self.lib.rig_init(self.model)
            if not self.rig:
                raise CatError(f"Hamlib could not initialize model {self.model}")
            if device:
                self._set_conf("rig_pathname", device)
            if self.baud:
                self._set_conf("serial_speed", str(self.baud))
            if self.model == 1:
                # Dummy defaults to no PTT. Enable its simulated CAT PTT so
                # operator rehearsals exercise the same ON/OFF path as a rig.
                self._set_conf("ptt_type", "RIG")
            self._check(self.lib.rig_open(self.rig), "open radio")
        except Exception:
            try:
                if self.rig:
                    self.lib.rig_cleanup(self.rig)
            finally:
                self.rig = None
                if self._dll_dir is not None:
                    self._dll_dir.close()
                    self._dll_dir = None
            raise

    def _configure_api(self) -> None:
        lib = self.lib
        if hasattr(lib, "rig_set_debug"):
            lib.rig_set_debug.argtypes = [ctypes.c_int]
            lib.rig_set_debug.restype = None
            lib.rig_set_debug(0)
        lib.rig_init.argtypes = [ctypes.c_int]
        lib.rig_init.restype = ctypes.c_void_p
        lib.rig_cleanup.argtypes = [ctypes.c_void_p]
        lib.rig_cleanup.restype = ctypes.c_int
        lib.rig_open.argtypes = [ctypes.c_void_p]
        lib.rig_open.restype = ctypes.c_int
        lib.rig_close.argtypes = [ctypes.c_void_p]
        lib.rig_close.restype = ctypes.c_int
        lib.rig_token_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.rig_token_lookup.restype = ctypes.c_long
        lib.rig_set_conf.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_char_p]
        lib.rig_set_conf.restype = ctypes.c_int
        lib.rig_set_ptt.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int]
        lib.rig_set_ptt.restype = ctypes.c_int
        if hasattr(lib, "rig_get_ptt"):
            lib.rig_get_ptt.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
            lib.rig_get_ptt.restype = ctypes.c_int
        lib.rig_set_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_double]
        lib.rig_set_freq.restype = ctypes.c_int
        lib.rig_get_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_double)]
        lib.rig_get_freq.restype = ctypes.c_int
        lib.rig_get_mode.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_long)
        ]
        lib.rig_get_mode.restype = ctypes.c_int
        if hasattr(lib, "rig_strrmode"):
            lib.rig_strrmode.argtypes = [ctypes.c_uint64]
            lib.rig_strrmode.restype = ctypes.c_char_p
        lib.rigerror.argtypes = [ctypes.c_int]
        lib.rigerror.restype = ctypes.c_char_p

    def _check(self, code: int, action: str) -> None:
        if code < 0:
            raw = self.lib.rigerror(code)
            message = raw.decode("utf-8", "replace") if raw else f"error {code}"
            raise CatError(f"Hamlib could not {action}: {message}")

    def _set_conf(self, name: str, value: str) -> None:
        token = self.lib.rig_token_lookup(self.rig, name.encode("ascii"))
        if token <= 0:
            raise CatError(f"Hamlib model does not expose {name}")
        self._check(self.lib.rig_set_conf(self.rig, token, value.encode("utf-8")), f"set {name}")

    def _require_open(self) -> None:
        if not self.rig:
            raise CatError("Hamlib radio connection is closed")

    def set_ptt(self, on: bool) -> None:
        with self._lock:
            if not self.rig and not on:
                return
            self._require_open()
            self._check(self.lib.rig_set_ptt(self.rig, self._VFO_CURR, 1 if on else 0), "set PTT")

    def set_frequency_hz(self, frequency_hz: float) -> None:
        frequency_hz = _frequency_hz(frequency_hz)
        with self._lock:
            self._require_open()
            self._check(self.lib.rig_set_freq(self.rig, self._VFO_CURR, frequency_hz), "set frequency")

    def get_ptt(self) -> bool:
        """Read the backend's PTT state, not a measurement of physical RF."""
        with self._lock:
            self._require_open()
            if not hasattr(self.lib, "rig_get_ptt"):
                raise CatError("Hamlib library cannot read PTT state")
            value = ctypes.c_int()
            self._check(self.lib.rig_get_ptt(self.rig, self._VFO_CURR, ctypes.byref(value)), "read PTT")
            if value.value not in {0, 1, 2, 3}:
                raise CatError(f"Hamlib returned invalid PTT state: {value.value}")
            return value.value != 0

    def get_frequency_hz(self) -> float:
        with self._lock:
            self._require_open()
            value = ctypes.c_double()
            self._check(self.lib.rig_get_freq(self.rig, self._VFO_CURR, ctypes.byref(value)), "read frequency")
            return _frequency_hz(value.value)

    def get_mode(self) -> str:
        with self._lock:
            self._require_open()
            # Hamlib 4 ABI: rmode_t is uint64_t; pbwidth_t is signed long.
            mode = ctypes.c_uint64()
            width = ctypes.c_long()
            self._check(
                self.lib.rig_get_mode(self.rig, self._VFO_CURR, ctypes.byref(mode), ctypes.byref(width)),
                "read mode",
            )
            if hasattr(self.lib, "rig_strrmode"):
                label = self.lib.rig_strrmode(mode.value)
                if label:
                    return label.decode("ascii", "replace")
            return str(mode.value)

    def describe(self) -> str:
        try:
            return f"Hamlib {self.get_frequency_hz() / 1e6:.6f} MHz — {self.device}"
        except CatError:
            return f"Hamlib model {self.model} — {self.device}"

    def close(self) -> None:
        with self._lock:
            if not self.rig:
                return
            try:
                try:
                    self.set_ptt(False)
                except Exception:
                    pass
                self.lib.rig_close(self.rig)
            finally:
                rig, self.rig = self.rig, None
                try:
                    self.lib.rig_cleanup(rig)
                finally:
                    if self._dll_dir is not None:
                        self._dll_dir.close()
                        self._dll_dir = None


class FlexPtt:
    """Native FlexRadio PTT using an independent SmartSDR API session."""

    def __init__(
        self,
        host: str,
        freq_mhz: float | None = None,
        require_mode: str | None = "DIGU",
        power: int | None = None,
        filter_low: int = 800,
        filter_high: int = 9200,
    ):
        if not host:
            raise CatError("Flex host is empty")
        if power is not None and not 1 <= int(power) <= 100:
            raise CatError("Flex power must be between 1 and 100 W")
        self.host = host
        self.session = FlexVitaSession(
            host,
            frequency_mhz=freq_mhz,
            mode=require_mode or "DIGU",
            power=power or 5,
            filter_low=filter_low,
            filter_high=filter_high,
        )

    def set_ptt(self, on: bool) -> None:
        self.session.set_ptt(on)

    def describe(self) -> str:
        return self.session.describe()

    def close(self) -> None:
        self.session.close()


class SerialLinePtt:
    """Raise RTS or DTR on a COM port for a hardware PTT interface."""

    def __init__(self, port: str, line: str = "rts"):
        if not port:
            raise CatError("serial PTT port is empty")
        try:
            import serial
        except ImportError as error:
            raise CatError("pyserial is required for RTS/DTR PTT") from error
        line = line.lower()
        if line not in {"rts", "dtr"}:
            raise CatError(f"serial PTT line must be rts or dtr, not {line!r}")
        self.port = port
        self.line = line
        self._serial = serial.Serial(port, timeout=0.2)
        self.set_ptt(False)

    def set_ptt(self, on: bool) -> None:
        if self.line == "rts":
            self._serial.rts = bool(on)
        else:
            self._serial.dtr = bool(on)

    def describe(self) -> str:
        return f"serial {self.line.upper()} on {self.port}"

    def close(self) -> None:
        try:
            self.set_ptt(False)
        except Exception:
            pass
        try:
            self._serial.close()
        except Exception:
            pass


@dataclass(frozen=True)
class CatConfig:
    backend: str = "none"
    rigctld_host: str = "127.0.0.1"
    rigctld_port: int = 4532
    hamlib_model: int = 0
    hamlib_device: str = ""
    hamlib_baud: int = 0
    flex_host: str = ""
    flex_power: int | None = 5
    flex_filter_low: int = 800
    flex_filter_high: int = 9200
    freq_mhz: float | None = None
    require_mode: str | None = "DIGU"
    serial_port: str = ""
    serial_line: str = "rts"


def open_ptt(config: CatConfig):
    name = (config.backend or "none").lower()
    if name in {"none", "vox", "off", ""}:
        return NullPtt()
    if name in {"rigctld", "hamlib", "hamlib-direct"}:
        client = (
            RigctldClient(config.rigctld_host, config.rigctld_port)
            if name == "rigctld" else
            HamlibDirect(config.hamlib_model, config.hamlib_device, config.hamlib_baud)
        )
        try:
            if config.freq_mhz is not None:
                client.set_frequency_hz(config.freq_mhz * 1e6)
        except Exception:
            client.close()
            raise
        return client
    if name == "flex":
        return FlexPtt(
            config.flex_host,
            freq_mhz=config.freq_mhz,
            require_mode=config.require_mode,
            power=config.flex_power,
            filter_low=config.flex_filter_low,
            filter_high=config.flex_filter_high,
        )
    if name in {"rts", "dtr"}:
        return SerialLinePtt(config.serial_port, line=name)
    raise CatError(f"unknown CAT backend {config.backend!r}")


def list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [info.device for info in list_ports.comports()]
