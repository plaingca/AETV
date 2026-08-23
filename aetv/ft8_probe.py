"""Operator-controlled FT8 probes and PSK Reporter calibration import."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio_io import read_wav
from .flex import FlexVitaSession
from .kiwi import KiwiReceiver
from .propagation import (
    CalibrationStore,
    ProbeMeasurement,
    TARGET_SNR_DB,
    antenna_gain_db,
    initial_bearing_deg,
    record_ota_measurement,
)
from .settings import StationSettings, config_dir, normalize_callsign

DEFAULT_FT8_DIALS_MHZ = (3.573, 7.074, 10.136, 14.074, 18.100, 21.074, 24.915, 28.074)
PSK_REPORTER_QUERY = "https://retrieve.pskreporter.info/query"
_runs_lock = threading.Lock()
_query_lock = threading.Lock()
_last_query_time = 0.0


@dataclass(frozen=True)
class Ft8Spot:
    receiver_callsign: str
    receiver_locator: str
    sender_callsign: str
    frequency_hz: int
    snr_db: float
    timestamp: int


@dataclass(frozen=True)
class Ft8ProbeRun:
    timestamp: int
    callsign: str
    grid: str
    dial_mhz: float
    audio_hz: float
    power_w: float


def ft8_runtime_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", str(config_dir()))) / "AETV" / "ft8_lib"
    return root / "aetv_gen_ft8.exe"


def parse_ft8_frequencies(text: str) -> list[float]:
    found = []
    for token in str(text or "").replace(";", ",").split(","):
        try:
            value = float(token.strip())
        except ValueError:
            continue
        if 1.6 <= value <= 30.0 and value not in found:
            found.append(value)
    return found or list(DEFAULT_FT8_DIALS_MHZ)


def maidenhead_grid(lat: float, lon: float, precision: int = 4) -> str:
    """Convert WGS84 coordinates to a 4- or 6-character Maidenhead locator."""
    if precision not in {4, 6} or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("valid coordinates and a 4- or 6-character grid are required")
    x = min(359.999999, lon + 180.0)
    y = min(179.999999, lat + 90.0)
    grid = chr(ord("A") + int(x // 20)) + chr(ord("A") + int(y // 10))
    x %= 20
    y %= 10
    grid += str(int(x // 2)) + str(int(y))
    if precision == 6:
        x %= 2
        y %= 1
        grid += chr(ord("A") + int(x * 12)) + chr(ord("A") + int(y * 24))
    return grid


def maidenhead_coordinates(locator: str) -> tuple[float, float]:
    """Return the center of a 4- or 6-character Maidenhead locator."""
    text = str(locator or "").strip().upper()
    if len(text) < 4 or not (text[:2].isalpha() and text[2:4].isdigit()):
        raise ValueError("invalid Maidenhead locator")
    lon = (ord(text[0]) - ord("A")) * 20.0 - 180.0
    lat = (ord(text[1]) - ord("A")) * 10.0 - 90.0
    lon += int(text[2]) * 2.0
    lat += int(text[3])
    lon_size, lat_size = 2.0, 1.0
    if len(text) >= 6 and text[4:6].isalpha():
        lon_size, lat_size = 2.0 / 24.0, 1.0 / 24.0
        lon += (ord(text[4]) - ord("A")) * lon_size
        lat += (ord(text[5]) - ord("A")) * lat_size
    lon += lon_size / 2.0
    lat += lat_size / 2.0
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Maidenhead locator is outside the Earth")
    return lat, lon


def generate_ft8_wav(message: str, output: Path, audio_hz: float = 1000.0) -> Path:
    executable = ft8_runtime_path()
    if not executable.is_file():
        raise RuntimeError(
            "FT8 encoder is not installed; run scripts/install_ft8_lib.ps1"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(executable), message, str(output), f"{float(audio_hz):.1f}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode or not output.is_file():
        raise RuntimeError((proc.stderr or proc.stdout or "FT8 generation failed").strip())
    return output


def parse_psk_reporter_xml(payload: bytes | str) -> list[Ft8Spot]:
    root = ET.fromstring(payload)
    spots = []
    for element in root.iter():
        if not element.tag.endswith("receptionReport"):
            continue
        attrs = element.attrib
        try:
            snr = float(attrs.get("sNR", attrs.get("snr", "nan")))
            frequency = int(float(attrs["frequency"]))
            timestamp = int(float(attrs["flowStartSeconds"]))
        except (KeyError, TypeError, ValueError):
            continue
        locator = attrs.get("receiverLocator", "").strip()
        if not locator or not math.isfinite(snr):
            continue
        spots.append(
            Ft8Spot(
                receiver_callsign=attrs.get("receiverCallsign", "?").strip().upper(),
                receiver_locator=locator.upper(),
                sender_callsign=attrs.get("senderCallsign", "").strip().upper(),
                frequency_hz=frequency,
                snr_db=snr,
                timestamp=timestamp,
            )
        )
    return spots


def fetch_psk_reporter_spots(
    callsign: str, since_timestamp: int, timeout: float = 20.0
) -> list[Ft8Spot]:
    global _last_query_time
    with _query_lock:
        remaining = 300.0 - (time.time() - _last_query_time)
        if remaining > 0:
            raise RuntimeError(
                f"PSK Reporter permits one query per five minutes; retry in {math.ceil(remaining)} seconds"
            )
        _last_query_time = time.time()
    now = int(time.time())
    lookback = max(300, min(24 * 3600, now - int(since_timestamp) + 300))
    query = urllib.parse.urlencode(
        {
            "senderCallsign": normalize_callsign(callsign),
            "flowStartSeconds": -lookback,
            "mode": "FT8",
            "rptlimit": 5000,
            "rronly": 1,
            "noactive": 1,
        }
    )
    request = urllib.request.Request(
        f"{PSK_REPORTER_QUERY}?{query}", headers={"User-Agent": "AETV/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        spots = parse_psk_reporter_xml(response.read())
    call = normalize_callsign(callsign)
    return [spot for spot in spots if spot.sender_callsign == call and spot.timestamp >= since_timestamp]


def import_spot_calibration(
    spots: list[Ft8Spot], settings: StationSettings,
    runs: list[Ft8ProbeRun] | None = None,
) -> int:
    prepared = []
    for spot in spots:
        try:
            rx_lat, rx_lon = maidenhead_coordinates(spot.receiver_locator)
        except ValueError:
            continue
        receiver = KiwiReceiver(
            host=f"pskr:{spot.receiver_callsign}",
            name=spot.receiver_callsign,
            loc=spot.receiver_locator,
            lat=rx_lat,
            lon=rx_lon,
        )
        bearing = initial_bearing_deg(
            settings.kiwi_lat, settings.kiwi_lon, rx_lat, rx_lon
        )
        gain = antenna_gain_db(
            settings.prop_antenna_pattern,
            bearing,
            settings.prop_antenna_azimuth_deg,
            settings.prop_antenna_gain_dbi,
        )
        matching_runs = [
            run for run in (runs or [])
            if abs(spot.timestamp - run.timestamp) <= 120
            and abs(spot.frequency_hz - (run.dial_mhz * 1_000_000.0 + run.audio_hz)) <= 5000
        ]
        power_w = (
            min(matching_runs, key=lambda run: abs(spot.timestamp - run.timestamp)).power_w
            if matching_runs else float(settings.flex_power)
        )
        prepared.append((spot, receiver, gain, power_w))

    def save(item) -> bool:
        spot, receiver, gain, power_w = item
        record_ota_measurement(
            receiver, settings.kiwi_lat, settings.kiwi_lon,
            spot.frequency_hz / 1_000_000.0,
            max(0.1, power_w), gain, spot.snr_db,
            normalize_callsign(settings.callsign),
            when=datetime.fromtimestamp(spot.timestamp, timezone.utc),
        )
        return True

    count = 0
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(prepared)))) as pool:
        futures = [pool.submit(save, item) for item in prepared]
        for future in as_completed(futures):
            try:
                count += int(future.result())
            except Exception:
                continue
    return count


def import_no_report_calibration(
    spots: list[Ft8Spot], runs: list[Ft8ProbeRun], settings: StationSettings,
) -> int:
    """Record a conservative band-wide miss for completed probes with no reports."""
    count = 0
    store = CalibrationStore()
    heard_runs: set[int] = set()
    for spot in spots:
        compatible = [
            (index, run) for index, run in enumerate(runs)
            if abs(spot.timestamp - run.timestamp) <= 120
            and abs(
                spot.frequency_hz
                - (run.dial_mhz * 1_000_000.0 + run.audio_hz)
            ) <= 5000
        ]
        if compatible:
            heard_runs.add(
                min(compatible, key=lambda item: abs(spot.timestamp - item[1].timestamp))[0]
            )
    for index, run in enumerate(runs):
        if index in heard_runs:
            continue
        # A network-wide absence is censored rather than an invented SNR. The
        # fixed prediction makes its residual -12 dB, then normal sparse-data
        # shrinkage keeps one or two misses from becoming an absolute verdict.
        store.append(
            ProbeMeasurement(
                timestamp_utc=datetime.fromtimestamp(run.timestamp, timezone.utc).isoformat(),
                host=f"pskr:no-report:{run.dial_mhz:.3f}",
                receiver_lat=float(settings.kiwi_lat),
                receiver_lon=float(settings.kiwi_lon),
                frequency_mhz=run.dial_mhz + run.audio_hz / 1_000_000.0,
                bearing_deg=0.0,
                measured_snr_db=None,
                predicted_snr_db=TARGET_SNR_DB + 6.0,
                tx_power_w=float(run.power_w),
                callsign=run.callsign,
                decoded=False,
                directional=False,
            )
        )
        count += 1
    return count


def spots_for_probe_runs(spots: list[Ft8Spot], runs: list[Ft8ProbeRun]) -> list[Ft8Spot]:
    """Keep reports matching a recorded slot and RF frequency, excluding ordinary QSOs."""
    selected = []
    seen = set()
    for spot in spots:
        for run in runs:
            expected_hz = run.dial_mhz * 1_000_000.0 + run.audio_hz
            if abs(spot.timestamp - run.timestamp) > 120:
                continue
            if abs(spot.frequency_hz - expected_hz) > 5000:
                continue
            key = (spot.receiver_callsign, spot.frequency_hz, spot.timestamp)
            if key not in seen:
                seen.add(key)
                selected.append(spot)
            break
    return selected


def runs_path() -> Path:
    return config_dir() / "ft8_probe_runs.json"


def load_probe_runs() -> list[Ft8ProbeRun]:
    try:
        return [Ft8ProbeRun(**row) for row in json.loads(runs_path().read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        return []


def save_probe_run(run: Ft8ProbeRun) -> None:
    with _runs_lock:
        rows = load_probe_runs()
        rows.append(run)
        target = runs_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(".tmp")
        pending.write_text(
            json.dumps([asdict(row) for row in rows[-200:]], indent=2) + "\n",
            encoding="utf-8",
        )
        pending.replace(target)


def transmit_ft8_probe(
    settings: StationSettings,
    dial_mhz: float,
    message: str,
    audio_hz: float = 1000.0,
    on_status=None,
) -> Ft8ProbeRun:
    """Transmit one explicitly authorized FT8 slot through direct Flex VITA audio."""
    if settings.cat_backend != "flex" or not settings.flex_native_audio or not settings.flex_host:
        raise RuntimeError("FT8 calibration currently requires configured Flex direct VITA audio")
    status = on_status or (lambda _message: None)
    safe_name = f"{normalize_callsign(settings.callsign)}-{dial_mhz:.3f}.wav".replace("/", "_")
    wav_path = config_dir() / "ft8_probes" / safe_name
    generate_ft8_wav(message, wav_path, audio_hz)
    sample_rate, audio = read_wav(wav_path)
    audio = np.asarray(audio, dtype=np.float32) * float(settings.tx_level)
    session = FlexVitaSession(
        settings.flex_host,
        frequency_mhz=float(dial_mhz),
        mode="DIGU",
        power=int(settings.flex_power),
        filter_low=300,
        filter_high=3000,
    )
    keyed = False
    try:
        session.prepare_tx()
        target = math.ceil((time.time() + 0.25) / 15.0) * 15.0
        status(f"waiting {max(0.0, target - time.time()):.1f} s for the next UTC FT8 slot")
        while time.time() < target:
            time.sleep(min(0.1, target - time.time()))
        status(f"transmitting {message} on {dial_mhz:.3f} MHz")
        session.set_ptt(True)
        keyed = True
        session.send_audio(audio, sample_rate)
    finally:
        if keyed:
            session.set_ptt(False)
        session.close()
    run = Ft8ProbeRun(
        timestamp=int(target),
        callsign=normalize_callsign(settings.callsign),
        grid=message.split()[-1].upper(),
        dial_mhz=float(dial_mhz),
        audio_hz=float(audio_hz),
        power_w=float(settings.flex_power),
    )
    save_probe_run(run)
    return run
