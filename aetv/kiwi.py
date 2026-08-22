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
import re
import struct
import threading
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

from .audio_io import StreamResampler, resample_ratio

LIST_URL = "http://rx.linkfanel.net/kiwisdr_com.js"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AETV/0.1"
SND_HEADER = 10


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
    request = urllib.request.Request(LIST_URL, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", "replace")


def parse_hosts(blob: str) -> list[str]:
    hosts: set[str] = set()
    for match in re.finditer(r"(?:url|host)[=:]\s*\"?(?:http://)?([A-Za-z0-9_.\-]+:\d+)", blob):
        hosts.add(match.group(1))
    for match in re.finditer(r"http://([A-Za-z0-9_.\-]+:\d+)", blob):
        hosts.add(match.group(1))
    return sorted(hosts)


def probe_receiver(host: str, timeout: float = 8.0) -> KiwiReceiver | None:
    try:
        request = urllib.request.Request(f"http://{host}/status", headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
    try:
        users = int(fields.get("users", "99"))
        users_max = int(fields.get("users_max", "0"))
        ext_api = int(fields.get("ext_api", "0"))
    except ValueError:
        return None
    return KiwiReceiver(
        host=host,
        name=fields.get("name", "")[:60],
        loc=fields.get("loc", "")[:40],
        lat=float(gps.group(1)),
        lon=float(gps.group(2)),
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
    timeout: float = 8.0,
    workers: int = 32,
    on_progress=None,
) -> list[KiwiReceiver]:
    """Probe the public list. Returns reachable receivers, nearest first."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    blob = fetch_public_list()
    hosts = parse_hosts(blob)
    found: list[KiwiReceiver] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_receiver, host, timeout): host for host in hosts}
        done = 0
        for future in as_completed(futures):
            done += 1
            info = future.result()
            if on_progress is not None:
                on_progress(done, len(hosts), info)
            if info is None:
                continue
            info.km = great_circle_km(lat, lon, info.lat, info.lon)
            if info.km <= max_km:
                found.append(info)
    found.sort(key=lambda item: (not item.usable, item.km if item.km is not None else 1e9))
    return found


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


class IqToPassband:
    """Streaming version of `iq_to_passband` with continuous phase and FIR state."""

    def __init__(self, src_rate: int, dst_rate: int, offset_hz: float):
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.offset_hz = float(offset_hz)
        self.phase = 0.0
        if src_rate != dst_rate:
            up, down = resample_ratio(src_rate, dst_rate)
            self._i = StreamResampler(up, down)
            self._q = StreamResampler(up, down)
        else:
            self._i = self._q = None

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
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
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
    ):
        self.host = host.strip()
        self.dial_mhz = float(dial_mhz)
        self.fcenter_hz = float(fcenter_hz)
        self.dst_rate = int(dst_rate)
        self.ring = ring
        self.user = user or "aetv"
        self.password = password or "#"
        self._on_status = on_status or (lambda status: None)
        self._on_error = on_error or (lambda msg: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
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
            try:
                asyncio.run(self._session())
            except Exception as error:
                self.status.connected = False
                self.status.message = str(error)
                self._on_error(f"Kiwi {self.host}: {error}")
                self._on_status(self.status)
            if self._stop.wait(0.05):
                break

    async def _session(self) -> None:
        try:
            import websockets
        except ImportError as error:
            raise RuntimeError("websockets is required for KiwiSDR receive") from error

        stamp = int(time.time() * 1000)
        uri = f"ws://{self.host}/ws/kiwi/{stamp}/SND"
        converter = None
        kiwi_rate = 12000
        async with websockets.connect(
            uri,
            origin=f"http://{self.host}",
            user_agent_header=BROWSER_UA,
            open_timeout=12,
            close_timeout=2,
            max_size=2**22,
        ) as ws:
            await ws.send(f"SET auth t=kiwi p={self.password} ipl=#")
            while not self._stop.is_set():
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=8.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(message, str):
                    self._handle_msg(message)
                    if "sample_rate" in message or message.startswith("MSG"):
                        match = re.search(r"sample_rate=([0-9.]+)", message)
                        if match:
                            kiwi_rate = int(round(float(match.group(1))))
                        if converter is None:
                            await self._tune(ws)
                            converter = IqToPassband(
                                kiwi_rate,
                                self.dst_rate,
                                offset_hz=self.dial_mhz * 1e6 - self.center_khz * 1e3,
                            )
                            self.status.connected = True
                            self.status.sample_rate = float(kiwi_rate)
                            self.status.message = f"IQ {self.center_khz:.2f} kHz"
                            self._on_status(self.status)
                    if "too_busy" in message or "too busy" in message.lower():
                        self.status.connected = False
                        self.status.message = "too busy; reconnecting"
                        self._on_status(self.status)
                        return
                    continue
                if converter is None or len(message) <= SND_HEADER:
                    continue
                flags_seq = message[:SND_HEADER]
                if len(flags_seq) >= 7:
                    smeter = struct.unpack_from(">H", flags_seq, 5)[0]
                    self.status.rssi_db = smeter / 10.0 - 127.0
                iq = _decode_snd_iq(message[SND_HEADER:])
                if iq.size:
                    audio = converter(iq)
                    if audio.size:
                        self.ring.write(audio)

    async def _tune(self, ws) -> None:
        center = self.center_khz
        await ws.send("SET AR OK in=12000 out=48000")
        await ws.send("SET squelch=0 maxdB=0 mindB=-110")
        await ws.send("SET compression=0")
        await ws.send(
            f"SET mod=iq low_cut=-5000 high_cut=5000 freq={center:.3f}"
        )
        await ws.send(f"SET ident_user={self.user}")
        await ws.send("SET OVERRIDE inactivity_timeout=0")

    def _handle_msg(self, message: str) -> None:
        if "too_busy" in message or "badp" in message or "password" in message.lower():
            self.status.message = message.strip()[:160]
            self._on_status(self.status)


def receivers_to_json(receivers: list[KiwiReceiver]) -> str:
    return json.dumps([item.__dict__ for item in receivers], indent=2) + "\n"
