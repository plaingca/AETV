"""FlexRadio discovery, native SmartSDR control, and VITA-49 audio.

The legacy WAV helper remains for command-line sound-device workflows. The
station GUI uses direct network audio and does not require SmartSDR or a DAX
sound device.
"""

from __future__ import annotations

import os
import re
import socket
import struct
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .audio_io import play_audio, read_wav, resample_audio


IPV4_HEADER_BYTES = 20
UDP_HEADER_BYTES = 8
VITA_HEADER_BYTES = 28
SAFE_PATH_MTU = 1200


def discover_path_mtu(sock, fallback: int = SAFE_PATH_MTU) -> int:
    """Read the OS path-MTU estimate from an already connected IPv4 socket."""
    if sock is None:
        return fallback
    # CPython does not currently export Winsock's IP_MTU (73). Linux exports
    # IP_MTU on most builds and uses value 14.
    option = getattr(socket, "IP_MTU", 73 if os.name == "nt" else 14)
    try:
        mtu = int(sock.getsockopt(socket.IPPROTO_IP, option))
    except (AttributeError, OSError):
        return fallback
    return mtu if 68 <= mtu <= 65535 else fallback


def vita_tx_samples_for_mtu(
    mtu: int,
    *,
    preferred_samples: int = 128,
    bytes_per_sample: int = 2,
) -> int:
    """Largest word-aligned VITA payload at or below FlexLib's packet size."""
    payload_budget = int(mtu) - IPV4_HEADER_BYTES - UDP_HEADER_BYTES - VITA_HEADER_BYTES
    samples = max(2, payload_budget // max(1, bytes_per_sample))
    if bytes_per_sample == 2:
        samples -= samples % 2  # VITA packet length is counted in 32-bit words
    return max(2, min(int(preferred_samples), samples))


class FlexClient:
    def __init__(self, host: str, port: int = 4992):
        self.sock = socket.create_connection((host, port), timeout=3)
        self.sock.settimeout(0.15)
        self.sequence = 10
        self.transcript: list[str] = []
        self._rx_buffer = bytearray()
        self._receive(0.25)

    def _receive(self, seconds: float) -> list[str]:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                part = self.sock.recv(65536)
                if not part:
                    break
                self._rx_buffer.extend(part)
            except TimeoutError:
                pass
        complete = bytes(self._rx_buffer).split(b"\n")
        self._rx_buffer = bytearray(complete.pop())
        lines = [line.rstrip(b"\r").decode("ascii", "replace") for line in complete]
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
                if len(fields) < 2 or int(fields[1] or "0", 16) != 0:
                    raise RuntimeError(f"Flex rejected {body!r}: {response}")
                return seen
        raise TimeoutError(f"no Flex response to {body!r}")

    def close(self) -> None:
        self.sock.close()


@dataclass(frozen=True)
class FlexRadioInfo:
    """A radio advertised by the SmartSDR UDP discovery protocol."""

    ip: str
    model: str = "FlexRadio"
    serial: str = ""
    nickname: str = ""
    callsign: str = ""
    version: str = ""
    port: int = 4992
    path_mtu: int | None = None

    @property
    def label(self) -> str:
        name = self.nickname or self.model
        suffix = f" ({self.callsign})" if self.callsign else ""
        mtu = f" · MTU {self.path_mtu}" if self.path_mtu else ""
        return f"{name}{suffix} — {self.ip}{mtu}"


def probe_radio_path_mtu(host: str, port: int = 4992, timeout: float = 1.5) -> int:
    """Connect without registering a client and return the current route MTU."""
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        return discover_path_mtu(sock)
    finally:
        sock.close()


def with_probed_path_mtu(radio: FlexRadioInfo, timeout: float = 1.5) -> FlexRadioInfo:
    """Return discovery information annotated with the current path MTU."""
    return replace(
        radio,
        path_mtu=probe_radio_path_mtu(radio.ip, radio.port, timeout),
    )


def _discovery_payload(packet: bytes) -> str:
    """Extract the key/value payload from either raw or VITA-49 discovery."""
    start = packet.find(b"model=")
    if start < 0:
        start = packet.find(b"serial=")
    if start < 0:
        return ""
    return packet[start:].rstrip(b"\x00\r\n").decode("utf-8", "replace")


def parse_discovery_packet(packet: bytes, source_ip: str = "") -> FlexRadioInfo | None:
    text = _discovery_payload(packet)
    if not text:
        return None
    fields = dict(item.split("=", 1) for item in text.split() if "=" in item)
    ip = fields.get("ip") or source_ip
    if not ip:
        return None
    try:
        port = int(fields.get("port", "4992"))
    except ValueError:
        port = 4992
    return FlexRadioInfo(
        ip=ip,
        model=fields.get("model", "FlexRadio"),
        serial=fields.get("serial", ""),
        nickname=fields.get("nickname", fields.get("name", "")),
        callsign=fields.get("callsign", ""),
        version=fields.get("version", ""),
        port=port,
    )


def discover_radios(timeout: float = 1.5) -> list[FlexRadioInfo]:
    """Listen for FlexRadio VITA-49 discovery broadcasts (UDP 4992)."""
    found: dict[str, FlexRadioInfo] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 4992))
        sock.settimeout(min(0.25, timeout))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                packet, source = sock.recvfrom(4096)
            except TimeoutError:
                continue
            info = parse_discovery_packet(packet, source[0])
            if info is not None:
                found[info.serial or info.ip] = info
    finally:
        sock.close()
    return sorted(found.values(), key=lambda item: (item.nickname, item.model, item.ip))


def _response_payload(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.startswith("R"):
            parts = line.split("|", 3)
            if len(parts) >= 3:
                return parts[2].strip()
    return ""


class FlexVitaSession:
    """Direct SmartSDR TCP control and VITA-49 audio, with no DAX driver.

    The radio still names these network streams ``dax_rx``/``dax_tx`` in its
    API.  Audio travels directly over UDP VITA-49; no Windows DAX sound device
    or SmartSDR process is involved.
    """

    AUDIO_INT16 = 0x0123
    AUDIO_FLOAT32 = 0x03E3
    FLEX_OUI = 0x001C2D
    FLEX_ICC = 0x534C

    def __init__(
        self,
        host: str,
        *,
        port: int = 4992,
        frequency_mhz: float | None = None,
        mode: str = "DIGU",
        power: int = 5,
        filter_low: int = 800,
        filter_high: int = 9200,
    ):
        self.host = host
        self.control = FlexClient(host, port)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        if os.name == "nt":
            # Winsock IP_DONTFRAGMENT. Packets are sized from IP_MTU below, so
            # fragmentation indicates a stale route and should fail visibly.
            try:
                self.udp.setsockopt(socket.IPPROTO_IP, 14, 1)
            except OSError:
                pass
        self.udp.bind(("", 0))
        self.udp.settimeout(0.25)
        self.udp_port = int(self.udp.getsockname()[1])
        self.slice_index: int | None = None
        self.rx_stream_id: int | None = None
        self.tx_stream_id: int | None = None
        self._tx_claimed = False
        self._created_slice = False
        self._running = threading.Event()
        self._rx_thread: threading.Thread | None = None
        self._packet_count = 0
        self.path_mtu = discover_path_mtu(getattr(self.control, "sock", None))
        self.tx_packet_samples = vita_tx_samples_for_mtu(self.path_mtu)
        self._radio_mtu: int | None = None
        self.radio_mtu_enforced = False

        # Bind to the existing SmartSDR/Maestro GUI whenever it exists so the
        # visible TX slice, local-PTT permission, and DAX ownership stay in one
        # client context. This is also required when a requested frequency is
        # supplied: an independent API client can see and tune a GUI-owned
        # slice without actually moving the radio's active transmit chain.
        # If no GUI exists, a frequency still permits a standalone slice.
        self.bound_client = None
        try:
            self.bound_client = bind_to_gui_client(self.control)
        except RuntimeError:
            pass
        self.client_id = self.bound_client or _response_payload(
            self.control.command("client gui")
        )
        self._apply_network_mtu()
        # Some radio firmware only accepts a built-in whitelist for the
        # optional `client program` label, so AETV deliberately does not send
        # it. Reduced-bandwidth DAX is also a capability hint, not a
        # prerequisite; full-rate float VITA audio is supported below.
        try:
            self.control.command("client set send_reduced_bw_dax=1")
        except RuntimeError:
            pass
        # The zero-length registration datagram is required by some 6000-series
        # firmware even when client udpport is also supplied.
        self.udp.sendto(b"\x00", (self.host, 4992))
        self.control.command(f"client udpport {self.udp_port}")
        lines = self.control.command("sub slice all")
        lines += self.control.command("sub tx all")
        lines += self.control._receive(0.5)
        slices = self._parse_slices(lines)
        tx_slices = [idx for idx, values in slices.items() if values.get("tx") == "1"]
        if tx_slices:
            self.slice_index = tx_slices[0]
        elif slices:
            self.slice_index = sorted(slices)[0]
        elif frequency_mhz is not None:
            payload = _response_payload(
                self.control.command(
                    f"slice create freq={frequency_mhz:.6f} mode={mode.lower()}"
                )
            )
            self.slice_index = int(payload.split()[0])
            self._created_slice = True
        else:
            self.close()
            raise RuntimeError("Flex has no slice; choose a frequency so AETV can create one")

        idx = self.slice_index
        if frequency_mhz is not None:
            self.control.command(f"slice tune {idx} {frequency_mhz:.6f}")
        self.control.command(f"slice set {idx} mode={mode} tx=1 dax=1")
        # Filter edges are status fields, not writable `slice set` fields.
        # SmartSDR exposes a dedicated FILT command on all supported 6000
        # firmware generations.
        self.control.command(f"filt {idx} {filter_low} {filter_high}")
        self.control.command(f"dax audio set 1 slice={idx} tx=1")
        # FILT controls the slice passband, while these transmit fields control
        # the actual RF transmit chain.  Setting only FILT can leave a previous
        # Flex-8k TX passband active even when a narrow V8 slice is displayed.
        self.control.command(
            f"transmit set dax=1 rfpower={max(1, min(100, int(power)))} "
            f"filter_low={filter_low} filter_high={filter_high}"
        )
        if frequency_mhz is not None:
            self._verify_transmit_frequency(float(frequency_mhz))

    def _verify_transmit_frequency(self, requested_mhz: float) -> None:
        """Refuse to key if the active Flex TX chain did not follow the slice."""
        lines = self.control.command("sub tx all")
        lines += self.control._receive(0.5)
        statuses = [line for line in lines if "|transmit freq=" in line]
        if not statuses:
            raise RuntimeError("Flex did not confirm its active transmit frequency")
        actual_mhz = _status_float(statuses[-1], "freq")
        if abs(actual_mhz - requested_mhz) > 1e-4:
            raise RuntimeError(
                f"Flex TX chain stayed on {actual_mhz:.6f} MHz after tuning the "
                f"selected slice to {requested_mhz:.6f} MHz; transmission was blocked"
            )

    @staticmethod
    def _parse_slices(lines: list[str]) -> dict[int, dict[str, str]]:
        found: dict[int, dict[str, str]] = {}
        for line in lines:
            match = re.search(r"\|slice (\d+) (.*)", line)
            if not match or "in_use=1" not in line:
                continue
            values = dict(item.split("=", 1) for item in match.group(2).split() if "=" in item)
            found[int(match.group(1))] = values
        return found

    def _create_stream(self, body: str) -> int:
        payload = _response_payload(self.control.command(body))
        try:
            return int(payload.split()[0], 16)
        except (ValueError, IndexError) as error:
            raise RuntimeError(f"Flex returned no stream id for {body!r}: {payload!r}") from error

    def start_rx(self, callback) -> None:
        if self.rx_stream_id is None:
            self.rx_stream_id = self._create_stream("stream create type=dax_rx dax_channel=1")
        self._running.set()

        def receive() -> None:
            while self._running.is_set():
                try:
                    packet, _source = self.udp.recvfrom(65536)
                except TimeoutError:
                    continue
                audio = self._decode_audio_packet(packet, self.rx_stream_id)
                if audio is not None and audio.size:
                    callback(audio)

        self._rx_thread = threading.Thread(target=receive, name="flex-vita-rx", daemon=True)
        self._rx_thread.start()

    @classmethod
    def _decode_audio_packet(cls, packet: bytes, stream_id: int | None) -> np.ndarray | None:
        if len(packet) < 28:
            return None
        words = struct.unpack_from(">7I", packet)
        if stream_id is not None and words[1] != stream_id:
            return None
        pcc = words[3] & 0xFFFF
        payload = packet[28 : (words[0] & 0xFFFF) * 4]
        if pcc == cls.AUDIO_INT16:
            return np.frombuffer(payload, dtype=">i2").astype(np.float32) / 32768.0
        if pcc == cls.AUDIO_FLOAT32:
            stereo = np.frombuffer(payload, dtype=">f4").astype(np.float32)
            if stereo.size % 2:
                return None
            # Full-band DAX is 48 kHz stereo. AETV's V7 modem consumes 24 kHz
            # mono, so use the left channel and decimate once. Averaging L/R
            # can cancel radios that provide opposite-polarity channels.
            return stereo[0::2][::2].copy()
        return None

    def set_ptt(self, on: bool) -> None:
        self.control.command(f"xmit {1 if on else 0}")

    def describe(self) -> str:
        return (
            f"Flex {self.host} — direct VITA-49 audio "
            f"(path MTU {self.path_mtu}, {self.tx_packet_samples} samples/packet)"
        )

    def _refresh_path_mtu(self) -> None:
        discovered = discover_path_mtu(
            getattr(self.control, "sock", None), fallback=self.path_mtu
        )
        changed = discovered != self.path_mtu
        self.path_mtu = discovered
        self.tx_packet_samples = vita_tx_samples_for_mtu(self.path_mtu)
        if changed or getattr(self, "_radio_mtu", None) != self.path_mtu:
            self._apply_network_mtu()

    def _apply_network_mtu(self) -> None:
        """Make the radio packetize its outbound VITA for this client's PMTU."""
        try:
            self.control.command(
                f"client set enforce_network_mtu=1 network_mtu={self.path_mtu}"
            )
        except RuntimeError:
            # Older firmware may not expose this hint. AETV still sizes its own
            # outbound packets conservatively in that case.
            self.radio_mtu_enforced = False
            return
        self._radio_mtu = self.path_mtu
        self.radio_mtu_enforced = True

    def prepare_tx(self) -> None:
        """Create and claim this client's DAX TX stream before keying."""
        self._refresh_path_mtu()
        if self.tx_stream_id is None:
            self.tx_stream_id = self._create_stream("stream create type=dax_tx")
        if not self._tx_claimed:
            # SmartSDR 3/4 MultiFlex assigns DAX TX ownership per stream.
            # ``transmit set dax=1`` selects DAX as the mic input but does not
            # grant this particular network stream permission to feed it.
            self.control.command(f"stream set 0x{self.tx_stream_id:08X} tx=1")
            self._tx_claimed = True

    def _encode_tx_packet(self, chunk: np.ndarray) -> bytes:
        """Encode one reduced-bandwidth Flex DAX TX packet (24 kHz mono)."""
        if self.tx_stream_id is None:
            raise RuntimeError("Flex TX stream is not prepared")
        values = np.asarray(chunk, dtype=np.float32).reshape(-1)
        int16 = (np.clip(values, -1.0, 1.0) * 32767).astype(np.int16)
        payload = int16.astype(">i2").tobytes()
        total_words = 7 + len(payload) // 4
        word0 = (
            (1 << 28) | (1 << 27) | (3 << 22) | (1 << 20)
            | ((self._packet_count & 0xF) << 16) | total_words
        )
        header = struct.pack(
            ">7I", word0, self.tx_stream_id, self.FLEX_OUI,
            (self.FLEX_ICC << 16) | self.AUDIO_INT16, 0, 0, 0,
        )
        return header + payload

    def send_audio(self, audio: np.ndarray, sample_rate: int, should_stop=None) -> bool:
        """Pace mono audio to the radio as Flex DAX TX VITA-49 packets."""
        self.prepare_tx()
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 24000:
            samples = resample_audio(samples, sample_rate, 24000)
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        # FlexLib 4.2 normally sends 128 samples. Keep that size when possible,
        # but never exceed the current route's unfragmented UDP payload.
        self._refresh_path_mtu()
        packet_samples = self.tx_packet_samples
        start = time.monotonic()
        sent = 0
        for offset in range(0, len(samples), packet_samples):
            if should_stop is not None and should_stop():
                return False
            chunk = samples[offset : offset + packet_samples]
            if len(chunk) < packet_samples:
                chunk = np.pad(chunk, (0, packet_samples - len(chunk)))
            self.udp.sendto(self._encode_tx_packet(chunk), (self.host, 4991))
            self._packet_count += 1
            sent += len(chunk)
            delay = start + sent / 24000.0 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return True

    def send_audio_stream(
        self,
        chunks,
        sample_rate: int,
        should_stop=None,
        on_chunk=None,
    ) -> bool:
        """Continuously packetize chunks without padding their boundaries.

        ``send_audio`` is intentionally a complete-buffer API and pads its last
        VITA packet. Calling it once per GOP therefore inserted route-MTU
        dependent silence into the RF stream. This method carries a partial
        packet into the next GOP and pads only once, after final lead-out.
        """
        if sample_rate != 24000:
            raise ValueError("continuous Flex audio currently requires 24 kHz input")
        self.prepare_tx()
        self._refresh_path_mtu()
        packet_samples = self.tx_packet_samples
        pending = np.zeros(0, dtype=np.float32)
        started = time.monotonic()
        sent = 0

        def send_packet(values: np.ndarray) -> None:
            nonlocal sent
            self.udp.sendto(self._encode_tx_packet(values), (self.host, 4991))
            self._packet_count += 1
            sent += len(values)
            delay = started + sent / 24000.0 - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        for index, chunk in enumerate(chunks):
            if should_stop is not None and should_stop():
                return False
            values = np.clip(
                np.asarray(chunk, dtype=np.float32).reshape(-1), -1.0, 1.0
            )
            if pending.size:
                values = np.concatenate([pending, values])
                pending = np.zeros(0, dtype=np.float32)
            complete = len(values) // packet_samples * packet_samples
            for offset in range(0, complete, packet_samples):
                if should_stop is not None and should_stop():
                    return False
                send_packet(values[offset : offset + packet_samples])
            pending = values[complete:].copy()
            if on_chunk is not None:
                on_chunk(index + 1)

        if pending.size:
            send_packet(np.pad(pending, (0, packet_samples - len(pending))))
        return not (should_stop is not None and should_stop())

    def close(self) -> None:
        self._running.clear()
        try:
            self.control.command("xmit 0")
        except Exception:
            pass
        if self.tx_stream_id is not None:
            try:
                self.control.command(f"stream set 0x{self.tx_stream_id:08X} tx=0")
            except Exception:
                pass
            self._tx_claimed = False
        for stream_id in (self.rx_stream_id, self.tx_stream_id):
            if stream_id is not None:
                try:
                    self.control.command(f"stream remove 0x{stream_id:08X}")
                except Exception:
                    pass
        if self._created_slice and self.slice_index is not None:
            try:
                self.control.command(f"slice remove {self.slice_index}")
            except Exception:
                pass
        try:
            self.udp.close()
        except OSError:
            pass
        try:
            self.control.close()
        except OSError:
            pass


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
            # Prefer the desktop SmartSDR client that owns the operator's
            # visible slice. Maestro and API utility sessions may coexist.
            preferred = [
                cid for cid, program in found.items()
                if program.lower().startswith("smartsdr")
            ]
            if len(preferred) != 1:
                listing = ", ".join(f"{cid} ({prog})" for cid, prog in found.items())
                raise RuntimeError(
                    f"several GUI clients connected and slice ownership is ambiguous: {listing}"
                )
            client_id = preferred[0]
        else:
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
