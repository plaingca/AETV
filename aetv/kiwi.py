"""KiwiSDR public-list lookup and live IQ capture.

V7's USB audio runs to 9 kHz. A Kiwi IQ stream is about 12 kHz complex,
so ±6 kHz around the *IQ centre*. Centre the receiver on the middle of
the AETV band (dial + fcenter) and the 8 kHz waveform fits. Resample
the IQ to the modem rate *before* heterodyning back to the transmitter
audio band; mixing first aliases the top of the waveform.

Public receivers that set `ext_api=0` grant a socket for about ten
seconds and then drop it with "Too busy now". Those are unusable for
automation even when the map shows a free channel.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import socket
import struct
import threading
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .audio_io import StreamResampler, resample_ratio

LIST_URL = "http://rx.linkfanel.net/kiwisdr_com.js"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AETV/0.1"
SND_HEADER = 7
IQ_GPS_HEADER = 10


def normalize_kiwi_host(value: str) -> str:
    """Turn a pasted Kiwi URL or host into the canonical ``host:port`` form."""
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = text if "://" in text else f"//{text}"
    parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https", "ws", "wss"}:
        raise ValueError("KiwiSDR address must be an http, https, ws, or wss URL")
    host = parsed.hostname
    if not host:
        raise ValueError("KiwiSDR address must contain a hostname or IP address")
    try:
        port = parsed.port or 8073
    except ValueError as error:
        raise ValueError("KiwiSDR URL has an invalid port") from error
    if not 1 <= port <= 65535:
        raise ValueError("KiwiSDR port must be between 1 and 65535")
    display_host = f"[{host}]" if ":" in host else host
    return f"{display_host}:{port}"


@dataclass
class KiwiReceiver:
    host: str
    name: str = ""
    loc: str = ""
    lat: float = 0.0
    lon: float = 0.0
    ext_api: int = 0
    users: int = 0
    users_max: int = 0
    free: int = 0
    km: float | None = None
    mode: str = ""
    offline: str = "?"
    predicted_snr_db: float | None = None
    predicted_uncertainty_db: float | None = None
    success_probability: float | None = None
    prediction_engine: str = ""

    @property
    def usable(self) -> bool:
        return self.ext_api > 0 and self.free > 0 and self.offline == "no"

    def label(self) -> str:
        place = self.loc or self.name or self.host
        km = "" if self.km is None else f"  {self.km:.0f} km"
        return f"{self.host}  {place}{km}"


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def fetch_public_list() -> str:
    """Fetch the canonical KiwiSDR directory.

    rx.linkfanel.net currently advertises IPv6 even where its IPv6 HTTP path
    may not answer. Prefer a resolved IPv4 address while retaining the
    canonical hostname in the Host header, then fall back to the ordinary URL.
    """
    parsed = urllib.parse.urlsplit(LIST_URL)
    headers = {"User-Agent": BROWSER_UA, "Connection": "close"}
    try:
        addresses = socket.getaddrinfo(
            parsed.hostname, parsed.port or 80, socket.AF_INET, socket.SOCK_STREAM
        )
        ipv4 = addresses[0][4][0]
        direct_url = urllib.parse.urlunsplit(
            (parsed.scheme, f"{ipv4}:{parsed.port}" if parsed.port else ipv4, parsed.path, parsed.query, "")
        )
        request = urllib.request.Request(direct_url, headers={**headers, "Host": parsed.netloc})
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        request = urllib.request.Request(LIST_URL, headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read().decode("utf-8", "replace")


def parse_hosts(blob: str) -> list[str]:
    hosts: set[str] = set()
    for match in re.finditer(r"(?:url|host)[=:]\s*\"?(?:http://)?([A-Za-z0-9_.\-]+:\d+)", blob):
        hosts.add(match.group(1))
    for match in re.finditer(r"http://([A-Za-z0-9_.\-]+:\d+)", blob):
        hosts.add(match.group(1))
    return sorted(hosts)


def parse_directory(blob: str) -> list[KiwiReceiver]:
    """Parse the canonical Kiwi directory without contacting each SDR."""
    match = re.search(r"var\s+kiwisdr_com\s*=\s*(\[.*?\])\s*;", blob, re.DOTALL)
    if match:
        # The feed is JavaScript and currently includes a trailing array comma.
        payload = re.sub(r",(\s*)\]$", r"\1]", match.group(1))
        entries = json.loads(payload)
        found: dict[str, KiwiReceiver] = {}
        for entry in entries:
            gps = re.match(
                r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
                str(entry.get("gps", "")),
            )
            if not gps:
                continue
            lat, lon = float(gps.group(1)), float(gps.group(2))
            # Kiwi directory coordinates are operator-supplied. Impossible
            # values must not wrap through great-circle trig and masquerade as
            # plausible propagation paths.
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            try:
                host = normalize_kiwi_host(str(entry.get("url", "")))
                users = int(entry.get("users", 99))
                users_max = int(entry.get("users_max", 0))
                ext_api = int(entry.get("ext_api", 0))
            except (TypeError, ValueError):
                continue
            found[host] = KiwiReceiver(
                host=host,
                name=str(entry.get("name", ""))[:60],
                loc=str(entry.get("loc", ""))[:40],
                lat=lat,
                lon=lon,
                ext_api=ext_api,
                users=users,
                users_max=users_max,
                free=max(0, users_max - users),
                mode=str(entry.get("mode", "")),
                offline=str(entry.get("offline", "?")),
            )
        return list(found.values())

    # Retain compatibility with saved responses from the former directory.
    match = re.search(r"var\s+receivers\s*=\s*(\[.*?\]);", blob, re.DOTALL)
    if not match:
        raise RuntimeError("Kiwi directory returned an unrecognized response")
    groups = json.loads(match.group(1))
    found: dict[str, KiwiReceiver] = {}
    for group in groups:
        coordinates = group.get("location", {}).get("coordinates", [])
        if len(coordinates) < 2:
            continue
        lon, lat = float(coordinates[0]), float(coordinates[1])
        for entry in group.get("receivers", []):
            if str(entry.get("type", "")).lower() != "kiwisdr":
                continue
            parsed = urllib.parse.urlparse(str(entry.get("url", "")))
            host = parsed.netloc
            if not host:
                continue
            found[host] = KiwiReceiver(
                host=host,
                name=str(entry.get("label", ""))[:60],
                loc=str(group.get("label", ""))[:40],
                lat=lat,
                lon=lon,
            )
    return list(found.values())


def probe_receiver(host: str, timeout: float = 8.0) -> KiwiReceiver | None:
    host = normalize_kiwi_host(host)
    try:
        request = urllib.request.Request(f"http://{host}/status", headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            resolved_url = getattr(response, "geturl", lambda: f"http://{host}/status")()
            text = response.read().decode("utf-8", "replace")
    except Exception:
        return None
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    gps = re.match(r"\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", fields.get("gps", ""))
    if not gps:
        return None
    lat, lon = float(gps.group(1)), float(gps.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    try:
        users = int(fields.get("users", "99"))
        users_max = int(fields.get("users_max", "0"))
        ext_api = int(fields.get("ext_api", "0"))
    except ValueError:
        return None
    return KiwiReceiver(
        # Public proxy front doors redirect /status to the generation-specific
        # host that actually supports WebSocket upgrades. WebSocket clients do
        # not reliably follow that HTTP redirect, so retain the resolved host.
        host=normalize_kiwi_host(resolved_url),
        name=fields.get("name", "")[:60],
        loc=fields.get("loc", "")[:40],
        lat=lat,
        lon=lon,
        ext_api=ext_api,
        users=users,
        users_max=users_max,
        free=users_max - users,
        mode=fields.get("mode", ""),
        offline=fields.get("offline", "?"),
    )


def find_receivers(
    lat: float,
    lon: float,
    max_km: float = 2500.0,
    timeout: float = 3.0,
    workers: int = 32,
    on_progress=None,
    max_probes: int = 0,
) -> list[KiwiReceiver]:
    """Return nearby entries from the canonical, live KiwiSDR directory.

    ``max_probes`` can request live status refreshes, but defaults to zero so
    discovery does not wait on dozens of unreachable receivers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    blob = fetch_public_list()
    candidates = parse_directory(blob)
    for item in candidates:
        item.km = great_circle_km(lat, lon, item.lat, item.lon)
    nearby = sorted(
        (item for item in candidates if item.km <= max_km),
        key=lambda item: item.km,
    )
    probe_targets = nearby[:max_probes] if max_probes > 0 else []
    found = {item.host: item for item in nearby}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_receiver, item.host, timeout): item for item in probe_targets}
        done = 0
        for future in as_completed(futures):
            done += 1
            info = future.result()
            if on_progress is not None:
                on_progress(done, len(probe_targets), info)
            if info is not None:
                info.km = great_circle_km(lat, lon, info.lat, info.lon)
                if info.km <= max_km:
                    found[info.host] = info
    result = list(found.values())
    result.sort(key=lambda item: (not item.usable, item.km if item.km is not None else 1e9))
    return result


def kiwi_center_khz(dial_mhz: float, fcenter_hz: float) -> float:
    """IQ centre that places `fcenter_hz` USB audio at DC."""
    return dial_mhz * 1000.0 + fcenter_hz / 1000.0


def iq_to_passband(
    iq: np.ndarray,
    src_rate: int,
    dst_rate: int,
    offset_hz: float,
    phase: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Resample complex IQ, mix by `-offset_hz`, return (real audio, next phase).

    `offset_hz` is `(dial_hz - center_hz)`. For a Kiwi centred on the
    AETV band middle that is `-fcenter`.
    """
    iq = np.asarray(iq)
    if iq.size == 0:
        return np.zeros(0, dtype=np.float32), phase
    if src_rate != dst_rate:
        from scipy.signal import resample_poly

        up, down = resample_ratio(src_rate, dst_rate)
        iq = resample_poly(iq.real, up, down) + 1j * resample_poly(iq.imag, up, down)
    n = np.arange(len(iq), dtype=np.float64)
    increment = 2.0 * math.pi * offset_hz / dst_rate
    phases = phase + increment * n
    audio = np.real(iq * np.exp(-1j * phases)).astype(np.float32)
    next_phase = float((phase + increment * len(iq)) % (2.0 * math.pi))
    return audio, next_phase


class _FractionalResampler:
    """Small streaming rate correction using continuous linear interpolation.

    The preceding polyphase stage does the anti-alias filtering and nearly all
    of the rate conversion. This stage only corrects the Kiwi's fractional
    crystal-rate error (normally tens of ppm), so linear interpolation has
    negligible passband loss while avoiding multi-second FIR buffering.
    """

    def __init__(self, src_rate: float, dst_rate: float):
        self.step = float(src_rate) / float(dst_rate)
        self._buf = np.zeros(0, dtype=np.float64)
        self._pos = 0.0

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate(
            [self._buf, np.asarray(chunk, dtype=np.float64).reshape(-1)]
        )
        available = (len(self._buf) - 1) - self._pos
        count = int(math.ceil(available / self.step)) if available > 0 else 0
        if count <= 0:
            return np.zeros(0, dtype=np.float64)
        positions = self._pos + self.step * np.arange(count, dtype=np.float64)
        positions = positions[positions < len(self._buf) - 1]
        if positions.size == 0:
            return np.zeros(0, dtype=np.float64)
        indices = np.arange(len(self._buf), dtype=np.float64)
        output = np.interp(positions, indices, self._buf)
        next_pos = float(positions[-1] + self.step)
        consumed = min(int(math.floor(next_pos)), len(self._buf) - 1)
        self._buf = self._buf[consumed:]
        self._pos = next_pos - consumed
        return output


class IqToPassband:
    """Streaming version of `iq_to_passband` with continuous phase and FIR state."""

    def __init__(self, src_rate: float, dst_rate: int, offset_hz: float):
        self.src_rate = float(src_rate)
        self.dst_rate = int(dst_rate)
        self.offset_hz = float(offset_hz)
        self.phase = 0.0
        if not math.isclose(src_rate, dst_rate):
            # Keep the expensive filtered ratio deliberately small. A second,
            # fractional stage below removes the remaining clock error using
            # the exact rate advertised by this particular KiwiSDR.
            coarse = Fraction(dst_rate / src_rate).limit_denominator(64)
            up, down = coarse.numerator, coarse.denominator
            self._i = StreamResampler(up, down)
            self._q = StreamResampler(up, down)
            coarse_rate = self.src_rate * up / down
            if not math.isclose(coarse_rate, self.dst_rate, rel_tol=1e-10):
                self._fine_i = _FractionalResampler(coarse_rate, self.dst_rate)
                self._fine_q = _FractionalResampler(coarse_rate, self.dst_rate)
            else:
                self._fine_i = self._fine_q = None
        else:
            self._i = self._q = None
            self._fine_i = self._fine_q = None

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq)
        if iq.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._i is None:
            i = np.asarray(iq.real, dtype=np.float64)
            q = np.asarray(iq.imag, dtype=np.float64)
        else:
            i = self._i(iq.real)
            q = self._q(iq.imag)
            if self._fine_i is not None:
                i = self._fine_i(i)
                q = self._fine_q(q)
        if len(i) == 0:
            return np.zeros(0, dtype=np.float32)
        increment = 2.0 * math.pi * self.offset_hz / self.dst_rate
        n = np.arange(len(i), dtype=np.float64)
        phases = self.phase + increment * n
        self.phase = float((self.phase + increment * len(i)) % (2.0 * math.pi))
        return np.real((i + 1j * q) * np.exp(-1j * phases)).astype(np.float32)


def _decode_snd_iq(payload: bytes) -> np.ndarray:
    """Kiwi SND body after the 10-byte header, as complex64 at the kiwi rate."""
    if len(payload) < 4:
        return np.zeros(0, dtype=np.complex64)
    if len(payload) % 2 == 0:
        # Normal Kiwi network samples are big-endian. Little-endian is only
        # used by the separate "camp" relay mode, which AETV does not use.
        samples = np.frombuffer(payload, dtype=">i2").astype(np.float32) / 32768.0
        if samples.size >= 2 and samples.size % 2 == 0:
            return samples[0::2] + 1j * samples[1::2]
    return np.zeros(0, dtype=np.complex64)


@dataclass
class KiwiStatus:
    connected: bool = False
    host: str = ""
    sample_rate: float = 12000.0
    rssi_db: float = float("nan")
    message: str = ""


class KiwiExternalApiDisabled(RuntimeError):
    """The receiver permits browser listening but no external IQ clients."""


class KiwiCapture:
    """Background IQ capture into a passband ring buffer.

    Reconnects immediately on the Kiwi's ~10 s `too_busy` drop. The
    modem will see a hole; that is better than a 15 s backoff that
    deletes the rest of the transmission.
    """

    def __init__(
        self,
        host: str,
        dial_mhz: float,
        fcenter_hz: float,
        dst_rate: int,
        ring,
        user: str = "aetv",
        password: str = "",
        on_status=None,
        on_error=None,
        on_discontinuity=None,
        on_iq=None,
    ):
        self.host = normalize_kiwi_host(host)
        self.dial_mhz = float(dial_mhz)
        self.fcenter_hz = float(fcenter_hz)
        self.dst_rate = int(dst_rate)
        self.ring = ring
        self.user = user or "aetv"
        self.password = password or ""
        self._on_status = on_status or (lambda status: None)
        self._on_error = on_error or (lambda msg: None)
        self._on_discontinuity = on_discontinuity or (lambda: None)
        self._on_iq = on_iq or (lambda _iq, _rate, _sequence: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._retry_delay = 0.05
        self.status = KiwiStatus(host=self.host)

    @property
    def center_khz(self) -> float:
        return kiwi_center_khz(self.dial_mhz, self.fcenter_hz)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kiwi-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=4.0)
        self._thread = None
        self.status.connected = False

    def _run(self) -> None:
        while not self._stop.is_set():
            self._session_had_samples = False
            try:
                asyncio.run(self._session())
            except Exception as error:
                self.status.connected = False
                self.status.message = str(error)
                self._retry_delay = (
                    30.0 if isinstance(error, KiwiExternalApiDisabled) else 1.0
                )
                self._on_error(f"Kiwi {self.host}: {error}")
                self._on_status(self.status)
            finally:
                if self._session_had_samples and not self._stop.is_set():
                    self._on_discontinuity()
            if self._stop.wait(self._retry_delay):
                break

    async def _session(self) -> None:
        try:
            import websockets
        except ImportError as error:
            raise RuntimeError("websockets is required for KiwiSDR receive") from error

        # Browser availability and external API availability are separate on a
        # KiwiSDR. A receiver may show free browser channels while refusing the
        # raw IQ socket used by AETV. Detect that explicitly instead of leaving
        # the GUI in an endless, silent reconnect loop. A supplied password is
        # allowed through because an operator may grant authenticated access.
        receiver = await asyncio.to_thread(probe_receiver, self.host, 5.0)
        if receiver is not None and receiver.ext_api <= 0 and not self.password:
            raise KiwiExternalApiDisabled(
                "external API is disabled (browser slots do not permit AETV IQ); "
                "choose an API-enabled Kiwi or enter an operator password"
            )
        connection_host = receiver.host if receiver is not None else self.host
        self.status.host = connection_host

        # Current Kiwi builds use /<token>/SND. Some receivers used in AETV's
        # original OTA trials only accepted /ws/kiwi/<timestamp>/SND, so fall
        # back when the first socket closes before it reports a sample rate.
        token = int(time.time() + os.getpid()) & 0xFFFFFFFF
        paths = [
            f"ws://{connection_host}/{token}/SND",
            f"ws://{connection_host}/ws/kiwi/{int(time.time() * 1000)}/SND",
        ]
        first_error = None
        for uri in paths:
            self.status.connected = False
            try:
                await self._session_uri(websockets, uri)
                return
            except Exception as error:
                if self.status.connected:
                    raise
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    async def _session_uri(self, websockets, uri: str) -> None:
        converter = None
        kiwi_rate = 12000.0
        last_sequence = None
        async with websockets.connect(
            uri,
            origin=f"http://{self.host}",
            user_agent_header=BROWSER_UA,
            open_timeout=12,
            close_timeout=2,
            max_size=2**22,
            ping_interval=None,
        ) as ws:
            await ws.send(f"SET auth t=kiwi p={self.password}")
            keepalive = asyncio.create_task(self._keepalive(ws))
            try:
                while not self._stop.is_set():
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(message, str):
                        raw = message.encode("utf-8", "replace")
                    else:
                        raw = bytes(message)
                    if len(raw) < 3:
                        continue
                    tag, body = raw[:3], raw[3:]
                    if tag == b"MSG":
                        text = body[1:].decode("utf-8", "replace") if body else ""
                        self._handle_msg(text)
                        audio_rate = re.search(r"(?:^|\s)audio_rate=([0-9.]+)", text)
                        if audio_rate:
                            await ws.send(
                                f"SET AR OK in={int(round(float(audio_rate.group(1))))} out=48000"
                            )
                        match = re.search(r"(?:^|\s)sample_rate=([0-9.]+)", text)
                        if match and converter is None:
                            kiwi_rate = float(match.group(1))
                            await self._tune(ws, kiwi_rate)
                            converter = IqToPassband(
                                kiwi_rate,
                                self.dst_rate,
                                offset_hz=self.dial_mhz * 1e6 - self.center_khz * 1e3,
                            )
                            self.status.connected = True
                            self._retry_delay = 0.05
                            self.status.sample_rate = float(kiwi_rate)
                            self.status.message = (
                                f"Kiwi tuned: TX dial {self.dial_mhz:.6f} MHz · "
                                f"IQ center {self.center_khz:.3f} kHz"
                            )
                            self._on_status(self.status)
                        if "too_busy" in text or "too busy" in text.lower():
                            self.status.connected = False
                            limit = re.search(r"(?:^|\s)too_busy=(\d+)", text)
                            limit_text = limit.group(1) if limit else "configured"
                            self._retry_delay = 5.0
                            self.status.message = (
                                f"external API slots full (limit {limit_text}); "
                                "retrying in 5 s or choose another Kiwi"
                            )
                            self._on_status(self.status)
                            return
                        continue
                    if tag != b"SND" or converter is None or len(body) <= SND_HEADER:
                        continue
                    sequence = struct.unpack_from("<I", body, 1)[0]
                    if last_sequence is not None and sequence != ((last_sequence + 1) & 0xFFFFFFFF):
                        # TCP preserves packets, so this means the Kiwi deleted IQ
                        # samples. Never splice those timelines into one modem GOP.
                        self._on_discontinuity()
                        converter = IqToPassband(
                            kiwi_rate,
                            self.dst_rate,
                            offset_hz=self.dial_mhz * 1e6 - self.center_khz * 1e3,
                        )
                    last_sequence = sequence
                    smeter = struct.unpack_from(">H", body, 5)[0]
                    self.status.rssi_db = smeter / 10.0 - 127.0
                    payload = body[SND_HEADER:]
                    # IQ mode prefixes every sample block with GNSS timing fields
                    # (solution flags, GPS seconds and nanoseconds).
                    if len(payload) <= IQ_GPS_HEADER:
                        continue
                    iq = _decode_snd_iq(payload[IQ_GPS_HEADER:])
                    if iq.size:
                        self._on_iq(iq, kiwi_rate, sequence)
                        audio = converter(iq)
                        if audio.size:
                            self.ring.write(audio)
                            self._session_had_samples = True
            finally:
                keepalive.cancel()
                await asyncio.gather(keepalive, return_exceptions=True)

    async def _keepalive(self, ws) -> None:
        """Keep a Kiwi allocation alive independently of SND packet flow."""
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            await ws.send("SET keepalive")

    async def _tune(self, ws, kiwi_rate: float) -> None:
        center = self.center_khz
        await ws.send(f"SET AR OK in={int(round(kiwi_rate))} out=48000")
        await ws.send("SET squelch=0 max=0")
        await ws.send("SET genattn=0")
        await ws.send("SET gen=0 mix=-1")
        await ws.send(
            f"SET mod=iq low_cut=-5500 high_cut=5500 freq={center:.3f}"
        )
        # Modem IQ must retain its amplitude envelope. Kiwi's AGC runs ahead
        # of the IQ websocket output, so use a fixed manual gain and keep both
        # AGC and ADPCM compression out of the receive path.
        await ws.send("SET agc=0 hang=0 thresh=-100 slope=6 decay=1000 manGain=50")
        await ws.send("SET compression=0")
        await ws.send(f"SET ident_user={self.user}")
        await ws.send("SET keepalive")

    def _handle_msg(self, message: str) -> None:
        badp = re.search(r"(?:^|\s)badp=(\d+)", message)
        if badp and badp.group(1) == "0":
            return
        if badp or "password" in message.lower():
            self.status.message = message.strip()[:160]
            self._on_status(self.status)


def receivers_to_json(receivers: list[KiwiReceiver]) -> str:
    return json.dumps([item.__dict__ for item in receivers], indent=2) + "\n"
