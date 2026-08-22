"""CAT / PTT backends for the ham station.

Anything used as a PTT object exposes `set_ptt(bool)` and optional
`close()` / `describe()`. The transmit engine keys inside try/finally
and a watchdog; backends must still be safe to call twice with False.

Backends:
- none: VOX / audio-only, no radio commands
- rigctld: Hamlib TCP (the same daemon WSJT-X and fldigi already use)
- flex: FlexRadio 6000 SmartSDR TCP, PTT only (tune the slice yourself)
- rts / dtr: serial-line PTT for a SignaLink-style interface
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from .flex import FlexClient, bind_to_gui_client, check_frequency


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


class FlexPtt:
    """Key a Flex 6000 that is already on frequency and in the right mode."""

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
        self.host = host
        self.radio = FlexClient(host)
        self._original: dict | None = None
        bound = bind_to_gui_client(self.radio)
        lines = self.radio.command("sub tx all")
        lines += self.radio._receive(0.5)
        status = next((line for line in lines if "|transmit freq=" in line), None)
        if status is None:
            self.radio.close()
            raise CatError("Flex did not provide transmit status")
        self.freq_mhz, self.mode = check_frequency(status, freq_mhz, require_mode)
        self.bound_client = bound
        from .flex import _status_value

        self._original = {
            "rfpower": _status_value(status, "rfpower"),
            "filter_low": _status_value(status, "lo"),
            "filter_high": _status_value(status, "hi"),
            "dax": _status_value(status, "dax"),
        }
        parts = ["dax=1", f"filter_low={filter_low}", f"filter_high={filter_high}"]
        if power is not None:
            if not 1 <= int(power) <= 100:
                raise CatError("Flex power must be between 1 and 100 W")
            parts.append(f"rfpower={int(power)}")
        self.radio.command("transmit set " + " ".join(parts))

    def set_ptt(self, on: bool) -> None:
        self.radio.command(f"xmit {1 if on else 0}")

    def describe(self) -> str:
        return f"Flex {self.host}  {self.freq_mhz:.6f} MHz {self.mode}"

    def close(self) -> None:
        try:
            self.radio.command("xmit 0")
        except Exception:
            pass
        if self._original is not None:
            try:
                original = self._original
                self.radio.command(
                    f"transmit set rfpower={original['rfpower']} dax={original['dax']} "
                    f"filter_low={original['filter_low']} filter_high={original['filter_high']}"
                )
            except Exception:
                pass
        self.radio.close()


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
