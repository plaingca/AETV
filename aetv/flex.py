"""FlexRadio DAX transmit helper with fail-safe unkey.

Frequency and mode are checked, never set. Keying is wrapped so PTT always
comes back down, including on playback failure.
"""

from __future__ import annotations

import re
import socket
import time
from pathlib import Path

import numpy as np

from .audio_io import play_audio, read_wav, resample_audio


class FlexClient:
    def __init__(self, host: str, port: int = 4992):
        self.sock = socket.create_connection((host, port), timeout=3)
        self.sock.settimeout(0.15)
        self.sequence = 10
        self.transcript: list[str] = []
        self._receive(0.25)

    def _receive(self, seconds: float) -> list[str]:
        deadline = time.monotonic() + seconds
        data = bytearray()
        while time.monotonic() < deadline:
            try:
                part = self.sock.recv(65536)
                if not part:
                    break
                data.extend(part)
            except TimeoutError:
                pass
        lines = data.decode("ascii", "replace").splitlines()
        self.transcript.extend(lines)
        return lines

    def command(self, body: str, timeout: float = 2.0) -> list[str]:
        sequence = self.sequence
        self.sequence += 1
        self.sock.sendall(f"C{sequence}|{body}\n".encode("ascii"))
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while time.monotonic() < deadline:
            seen.extend(self._receive(min(0.2, deadline - time.monotonic())))
            response = next((line for line in seen if line.startswith(f"R{sequence}|")), None)
            if response is not None:
                fields = response.split("|", 2)
                if len(fields) < 2 or fields[1] != "0":
                    raise RuntimeError(f"Flex rejected {body!r}: {response}")
                return seen
        raise TimeoutError(f"no Flex response to {body!r}")

    def close(self) -> None:
        self.sock.close()


def _status_value(text: str, name: str) -> int:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(-?\d+)(?:\s|$)", text)
    if not match:
        raise RuntimeError(f"missing {name} in Flex transmit status")
    return int(match.group(1))


def _status_float(text: str, name: str) -> float:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(-?\d+(?:\.\d+)?)(?:\s|$)", text)
    if not match:
        raise RuntimeError(f"missing {name} in Flex transmit status")
    return float(match.group(1))


def _status_text(text: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S*)", text)
    if not match:
        raise RuntimeError(f"missing {name} in Flex status")
    return match.group(1)


def bind_to_gui_client(radio: FlexClient, client_id: str | None = None) -> str:
    if client_id is None:
        lines = radio.command("sub client all")
        lines += radio._receive(1.0)
        found: dict[str, str] = {}
        for line in lines:
            match = re.search(
                r"\|client (0x[0-9A-Fa-f]+) connected .*?client_id=([0-9A-Fa-f-]+)", line
            )
            if match:
                program = re.search(r"program=(\S+)", line)
                found[match.group(2)] = program.group(1) if program else "?"
        if not found:
            raise RuntimeError(
                "no GUI client is connected to the radio, so there is no transmit "
                "slice to inherit — start SmartSDR (or pass --bind-client-id)"
            )
        if len(found) > 1:
            listing = ", ".join(f"{cid} ({prog})" for cid, prog in found.items())
            raise RuntimeError(f"several GUI clients connected; pass --bind-client-id: {listing}")
        client_id = next(iter(found))
    radio.command(f"client bind client_id={client_id}")
    return client_id


def check_frequency(status: str, want_mhz: float | None, want_mode: str | None) -> tuple[float, str]:
    freq_mhz = _status_float(status, "freq")
    mode = _status_text(status, "tx_slice_mode")
    if want_mhz is not None and abs(freq_mhz - want_mhz) > 1e-4:
        raise RuntimeError(
            f"radio is on {freq_mhz:.6f} MHz, not the requested {want_mhz:.6f} MHz — "
            "tune it and re-run rather than transmitting on the wrong frequency"
        )
    if want_mode and mode.upper() != want_mode.upper():
        raise RuntimeError(f"transmit slice is in {mode}, not {want_mode}")
    return freq_mhz, mode


def send_wav(
    wav_path: str | Path,
    host: str,
    device: str = "DAX TX (FlexRadio DAX)",
    power: int = 5,
    filter_low: int = 800,
    filter_high: int = 9200,
    freq_mhz: float | None = None,
    require_mode: str | None = "DIGU",
    bind_client_id: str | None = None,
    audio_only: bool = False,
) -> dict:
    """Play a prepared AETV WAV into DAX, optionally keying the Flex."""
    if not 1 <= power <= 100:
        raise RuntimeError("transmit power must be between 1 and 100 W")
    sample_rate, audio = read_wav(wav_path)
    output_rate = 48000
    if sample_rate != output_rate:
        audio = resample_audio(audio, sample_rate, output_rate)
    if audio_only:
        play_audio(audio, output_rate, device=device)
        return {"audio_only": True, "device": device, "duration_s": len(audio) / output_rate}

    radio = FlexClient(host)
    keyed = False
    original = None
    started = time.time()
    bound_client = None
    found_freq_mhz = None
    found_mode = None
    try:
        bound_client = bind_to_gui_client(radio, bind_client_id)
        lines = radio.command("sub tx all")
        lines += radio._receive(0.5)
        status = next((line for line in lines if "|transmit freq=" in line), None)
        if status is None:
            raise RuntimeError("Flex did not provide transmit status")
        original = {
            "rfpower": _status_value(status, "rfpower"),
            "filter_low": _status_value(status, "lo"),
            "filter_high": _status_value(status, "hi"),
            "dax": _status_value(status, "dax"),
        }
        found_freq_mhz, found_mode = check_frequency(status, freq_mhz, require_mode)
        radio.command(
            f"transmit set rfpower={power} dax=1 "
            f"filter_low={filter_low} filter_high={filter_high}"
        )
        radio.command("xmit 1")
        keyed = True
        play_audio(audio, output_rate, device=device)
    finally:
        try:
            radio.command("xmit 0")
            keyed = False
        except Exception:
            pass
        if original is not None:
            try:
                radio.command(
                    f"transmit set rfpower={original['rfpower']} dax={original['dax']} "
                    f"filter_low={original['filter_low']} filter_high={original['filter_high']}"
                )
            except Exception:
                pass
        radio.close()
    return {
        "host": host,
        "wav": str(Path(wav_path).resolve()),
        "device": device,
        "power_w": power,
        "tx_filter_hz": [filter_low, filter_high],
        "bound_client_id": bound_client,
        "found_freq_mhz": found_freq_mhz,
        "slice_mode": found_mode,
        "duration_s": len(audio) / output_rate,
        "started_unix": started,
        "keyed_at_exit": keyed,
    }
