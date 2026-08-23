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
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .flex import FlexVitaSession


class CatError(RuntimeError):
    pass


class NullPtt:
    """No radio control. Use VOX or key the rig by hand."""

    def set_ptt(self, on: bool) -> None:
        return None

    def describe(self) -> str:
        return "no CAT (VOX / manual PTT)"

    def close(self) -> None:
        return None


class RigctldClient:
    """Hamlib `rigctld` over TCP.

    Speaks the classic net protocol (`T 1` / `T 0` / `f` / `m`), which
    every Hamlib 4.x daemon accepts. Host and port are stored as names
    because a USB serial adapter's COM number is not stable; the daemon
    is what stays put.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4532, timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)

    def _command(self, line: str) -> str:
        self.sock.sendall((line.rstrip() + "\n").encode("ascii"))
        chunks: list[bytes] = []
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                part = self.sock.recv(4096)
            except TimeoutError as error:
                raise CatError(f"rigctld timed out on {line!r}") from error
            if not part:
                break
            chunks.append(part)
            text = b"".join(chunks)
            if b"\n" in text:
                break
        reply = b"".join(chunks).decode("ascii", "replace").strip()
        if reply.startswith("RPRT") and not reply.endswith("0"):
            raise CatError(f"rigctld rejected {line!r}: {reply}")
        return reply

    def set_ptt(self, on: bool) -> None:
        self._command(f"T {1 if on else 0}")

    def get_frequency_hz(self) -> float:
        return float(self._command("f").splitlines()[0])

    def get_mode(self) -> str:
        return self._command("m").splitlines()[0].strip()

    def describe(self) -> str:
        try:
            hz = self.get_frequency_hz()
            mode = self.get_mode()
            return f"rigctld {self.host}:{self.port}  {hz / 1e6:.6f} MHz {mode}"
        except Exception:
            return f"rigctld {self.host}:{self.port}"

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


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
    except OSError:
        return []
    models: list[HamlibModel] = []
    for line in proc.stdout.splitlines():
        fields = line.split(None, 4)
        if not fields or not fields[0].isdigit() or len(fields) < 3:
            continue
        models.append(HamlibModel(int(fields[0]), fields[1], fields[2]))
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
    for candidate in candidates:
        if Path(candidate).is_file() or not Path(candidate).is_absolute():
            try:
                if os.name == "nt" and Path(candidate).is_absolute():
                    os.add_dll_directory(str(Path(candidate).parent))
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
        self.model = int(model)
        self.device = device
        self.baud = int(baud)
        library = _find_hamlib_library()
        self._dll_dir = (
            os.add_dll_directory(str(Path(library).parent))
            if os.name == "nt" and Path(library).is_absolute() else None
        )
        self.lib = ctypes.CDLL(library)
        self._configure_api()
        self.rig = self.lib.rig_init(self.model)
        if not self.rig:
            raise CatError(f"Hamlib could not initialize model {self.model}")
        try:
            if device:
                self._set_conf("rig_pathname", device)
            if self.baud:
                self._set_conf("serial_speed", str(self.baud))
            self._check(self.lib.rig_open(self.rig), "open radio")
        except Exception:
            self.lib.rig_cleanup(self.rig)
            self.rig = None
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
        lib.rig_token_lookup.restype = ctypes.c_int
        lib.rig_set_conf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
        lib.rig_set_conf.restype = ctypes.c_int
        lib.rig_set_ptt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.rig_set_ptt.restype = ctypes.c_int
        lib.rig_get_freq.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        lib.rig_get_freq.restype = ctypes.c_int
        lib.rig_get_mode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        lib.rig_get_mode.restype = ctypes.c_int
        lib.rigerror.argtypes = [ctypes.c_int]
        lib.rigerror.restype = ctypes.c_char_p

    def _check(self, code: int, action: str) -> None:
        if code < 0:
            message = self.lib.rigerror(code).decode("utf-8", "replace")
            raise CatError(f"Hamlib could not {action}: {message}")

    def _set_conf(self, name: str, value: str) -> None:
        token = self.lib.rig_token_lookup(self.rig, name.encode("ascii"))
        if token <= 0:
            raise CatError(f"Hamlib model does not expose {name}")
        self._check(self.lib.rig_set_conf(self.rig, token, value.encode("utf-8")), f"set {name}")

    def set_ptt(self, on: bool) -> None:
        self._check(self.lib.rig_set_ptt(self.rig, self._VFO_CURR, 1 if on else 0), "set PTT")

    def get_frequency_hz(self) -> float:
        value = ctypes.c_double()
        self._check(self.lib.rig_get_freq(self.rig, self._VFO_CURR, ctypes.byref(value)), "read frequency")
        return float(value.value)

    def get_mode(self) -> str:
        mode = ctypes.c_int()
        width = ctypes.c_int()
        self._check(
            self.lib.rig_get_mode(self.rig, self._VFO_CURR, ctypes.byref(mode), ctypes.byref(width)),
            "read mode",
        )
        return str(mode.value)

    def describe(self) -> str:
        try:
            return f"Hamlib {self.get_frequency_hz() / 1e6:.6f} MHz — {self.device}"
        except Exception:
            return f"Hamlib model {self.model} — {self.device}"

    def close(self) -> None:
        if not self.rig:
            return
        try:
            self.set_ptt(False)
        except Exception:
            pass
        self.lib.rig_close(self.rig)
        self.lib.rig_cleanup(self.rig)
        self.rig = None


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
    if name == "rigctld":
        return RigctldClient(config.rigctld_host, config.rigctld_port)
    if name in {"hamlib", "hamlib-direct"}:
        return HamlibDirect(config.hamlib_model, config.hamlib_device, config.hamlib_baud)
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
