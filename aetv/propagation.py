"""HF path prediction and station-specific OTA calibration.

The preferred predictor is ITU-R Study Group 3's ITURHFProp reference
implementation of P.533/P.372.  A deliberately conservative geometric
fallback keeps Kiwi ranking usable when that optional native runtime has not
yet been installed; fallback results are always labelled as such.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
import threading
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .kiwi import KiwiReceiver, great_circle_km
from .settings import config_dir

SWPC_R12_FORECAST_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/"
    "predicted-solar-cycle.json"
)
SWPC_R12_OBSERVED_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/"
    "observed-solar-cycle-indices.json"
)
SWPC_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
TARGET_SNR_DB = 9.0
MIN_AUTO_SNR_DB = 6.0
CALIBRATION_RESIDUAL_LIMIT_DB = 12.0
GENERAL_CALIBRATION_LIMIT_DB = 6.0
HOST_CALIBRATION_LIMIT_DB = 8.0

# Planning dials, not a band-plan authorization. The emitted AETV passband is
# above the suppressed-carrier dial, so users must edit this list for their
# licence, region, mode bandwidth, and a clear channel.
DEFAULT_HF_PLANNING_DIALS_MHZ = (
    3.588,
    7.088,
    14.088,
    18.098,
    21.088,
    24.928,
    28.088,
)


def parse_planning_frequencies(value: str, current_mhz: float) -> list[float]:
    """Return unique, valid HF planning dials and always include the current dial."""
    frequencies = []
    for token in str(value or "").replace(";", ",").split(","):
        try:
            frequency = float(token.strip())
        except ValueError:
            continue
        if 1.6 <= frequency <= 30.0 and frequency not in frequencies:
            frequencies.append(frequency)
    current = float(current_mhz)
    if 1.6 <= current <= 30.0 and current not in frequencies:
        frequencies.append(current)
    return frequencies or [current]


def frequency_search_radius_km(frequency_mhz: float, base_radius_km: float) -> float:
    """Expand the Kiwi search horizon as higher HF bands favour longer skip."""
    frequency = float(frequency_mhz)
    base = max(50.0, float(base_radius_km))
    if frequency < 5.0:
        floor = 2500.0
    elif frequency < 10.0:
        floor = 4000.0
    elif frequency < 14.0:
        floor = 6000.0
    elif frequency < 18.0:
        floor = 9000.0
    elif frequency < 24.0:
        floor = 12000.0
    else:
        floor = 15000.0
    return min(20000.0, max(base, floor))
_store_lock = threading.Lock()


@dataclass(frozen=True)
class SpaceWeather:
    ssn: float = 100.0
    kp: float = 2.0
    source: str = "default"


@dataclass
class PathEstimate:
    host: str
    when_utc: str
    distance_km: float
    bearing_deg: float
    predicted_snr_db: float
    calibrated_snr_db: float
    uncertainty_db: float
    success_probability: float
    muf_mhz: float
    takeoff_deg: float
    reliability_pct: float
    correction_db: float
    samples: int
    engine: str


@dataclass(frozen=True)
class ProbeMeasurement:
    timestamp_utc: str
    host: str
    receiver_lat: float
    receiver_lon: float
    frequency_mhz: float
    bearing_deg: float
    measured_snr_db: float | None
    predicted_snr_db: float
    tx_power_w: float
    callsign: str
    decoded: bool = True
    directional: bool = True

    @property
    def residual_db(self) -> float:
        if self.measured_snr_db is None or not self.decoded:
            # A non-decode is censored evidence: the path was below the usable
            # threshold, but we do not invent an exact measured SNR.
            return min(-6.0, TARGET_SNR_DB - self.predicted_snr_db - 6.0)
        return self.measured_snr_db - self.predicted_snr_db


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def _angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def calibration_band(frequency_mhz: float) -> str:
    """Return the amateur HF band used to keep OTA calibration band-local."""
    frequency = float(frequency_mhz)
    ranges = (
        (1.8, 2.1, "160m"),
        (3.4, 4.1, "80m"),
        (5.0, 5.6, "60m"),
        (6.8, 7.4, "40m"),
        (9.9, 10.3, "30m"),
        (13.8, 14.5, "20m"),
        (17.8, 18.3, "17m"),
        (20.8, 21.6, "15m"),
        (24.7, 25.1, "12m"),
        (27.8, 30.0, "10m"),
    )
    for low, high, name in ranges:
        if low <= frequency <= high:
            return name
    return f"{round(frequency, 1):.1f}MHz"


def antenna_gain_db(
    pattern: str, bearing_deg: float, azimuth_deg: float, peak_gain_dbi: float
) -> float:
    """Simple initial azimuth pattern; OTA residuals refine the real station."""
    delta = _angular_difference(bearing_deg, azimuth_deg)
    if pattern == "dipole":
        # azimuth is broadside; the opposite broadside lobe is equivalent.
        relative = 10.0 * math.log10(max(0.01, math.cos(math.radians(delta)) ** 2))
        return peak_gain_dbi + max(-20.0, relative)
    if pattern == "directional":
        # Generic 70-degree beam with a bounded rear/side response.
        return peak_gain_dbi - min(20.0, 12.0 * (delta / 70.0) ** 2)
    return peak_gain_dbi


def _runtime_root() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", str(config_dir())))
    return local / "AETV" / "iturhfprop"


def native_runtime_status(month: int | None = None) -> tuple[bool, Path]:
    root = _runtime_root()
    month = month or datetime.now(timezone.utc).month
    required = (
        root / "ITURHFProp.exe",
        root / "P533.dll",
        root / "P372.dll",
        root / "Data" / "P1239-3 Decile Factors.txt",
        root / "Data" / f"ionos{month:02d}.bin",
        root / "Data" / f"COEFF{month:02d}W.txt",
    )
    return all(path.is_file() for path in required), root


def fetch_space_weather(timeout: float = 8.0) -> SpaceWeather:
    """Read current P.533 R12 and Kp, using a six-hour disk cache.

    ITU-R P.533 requires the 12-month smoothed sunspot number R12, not the
    current raw monthly count or daily 10.7 cm solar flux. NOAA cannot publish
    a centered observed R12 for the current month until six months later, so
    use its current-month smoothed-cycle forecast for planning.
    """
    cache = config_dir() / "space_weather.json"
    now = datetime.now(timezone.utc).timestamp()
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if (
            payload.get("solar_index") == "R12"
            and now - float(payload["cached_at"]) < 6 * 3600
        ):
            return SpaceWeather(float(payload["ssn"]), float(payload["kp"]), "NOAA cache")
    except (OSError, ValueError, KeyError, TypeError):
        pass

    def read_json(url: str):
        request = urllib.request.Request(url, headers={"User-Agent": "AETV/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    solar_source = "default R12"
    ssn = 100.0
    try:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        forecast = read_json(SWPC_R12_FORECAST_URL)
        matching = [row for row in forecast if row.get("time-tag") == month]
        if not matching:
            raise ValueError(f"NOAA R12 forecast has no row for {month}")
        ssn = float(matching[-1]["predicted_ssn"])
        solar_source = "NOAA SWPC R12 forecast"
    except Exception:
        try:
            observed = read_json(SWPC_R12_OBSERVED_URL)
            usable = [
                row
                for row in observed
                if float(row.get("smoothed_ssn", -1)) >= 0
            ]
            ssn = float(usable[-1]["smoothed_ssn"])
            solar_source = "NOAA SWPC observed R12"
        except Exception:
            pass

    kp = 2.0
    try:
        kp_rows = read_json(SWPC_KP_URL)
        kp = float(kp_rows[-1][1] if isinstance(kp_rows[-1], list) else kp_rows[-1]["Kp"])
    except Exception:
        pass
    try:
        cache.write_text(
            json.dumps(
                {
                    "cached_at": now,
                    "ssn": ssn,
                    "kp": kp,
                    "solar_index": "R12",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    return SpaceWeather(ssn, kp, solar_source)


class CalibrationStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (config_dir() / "propagation_measurements.json")

    def load(self) -> list[ProbeMeasurement]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            return [ProbeMeasurement(**row) for row in rows]
        except (OSError, ValueError, TypeError):
            return []

    def append(self, measurement: ProbeMeasurement) -> None:
        with _store_lock:
            rows = self.load()
            # Repeated GOPs from one reception should not overwhelm independent
            # paths. Keep at most one observation per host/frequency/minute.
            minute = measurement.timestamp_utc[:16]
            rows = [
                row
                for row in rows
                if not (
                    row.host == measurement.host
                    and abs(row.frequency_mhz - measurement.frequency_mhz) < 0.001
                    and row.timestamp_utc[:16] == minute
                )
            ]
            rows.append(measurement)
            rows = rows[-2000:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pending = self.path.with_suffix(".tmp")
            pending.write_text(
                json.dumps([asdict(row) for row in rows], indent=2) + "\n",
                encoding="utf-8",
            )
            pending.replace(self.path)

    def correction(
        self, host: str, bearing_deg: float, frequency_mhz: float
    ) -> tuple[float, float, int]:
        rows = self.load()
        if not rows:
            return 0.0, 6.0, 0
        weighted: list[tuple[float, float]] = []
        exact: list[ProbeMeasurement] = []
        target_band = calibration_band(frequency_mhz)
        for row in rows:
            if calibration_band(row.frequency_mhz) != target_band:
                continue
            if row.host == host and not row.host.startswith("pskr:"):
                exact.append(row)
                continue
            # A non-decode at some other Kiwi says nothing reliable about this
            # receiver: it may have been tuning, capture, occupancy or decoder
            # failure. Only network-wide FT8 no-report probes are allowed to
            # contribute negative evidence across receivers.
            if not row.decoded and row.directional:
                continue
            angle = _angular_difference(bearing_deg, row.bearing_deg)
            direction_weight = (
                math.exp(-0.5 * (angle / 45.0) ** 2) if row.directional else 1.0
            )
            band_weight = math.exp(
                -0.5 * (math.log(max(frequency_mhz, 0.1) / max(row.frequency_mhz, 0.1)) / 0.08) ** 2
            )
            source_weight = 3.0 if not row.host.startswith("pskr:") else 1.0
            weight = direction_weight * band_weight * source_weight
            if weight >= 0.02:
                residual = max(
                    -CALIBRATION_RESIDUAL_LIMIT_DB,
                    min(CALIBRATION_RESIDUAL_LIMIT_DB, row.residual_db),
                )
                weighted.append((weight, residual))
        if not weighted and not exact:
            return 0.0, 6.0, 0
        general_correction = 0.0
        uncertainty = 6.0
        if weighted:
            total = sum(weight for weight, _ in weighted)
            mean = sum(weight * value for weight, value in weighted) / total
            effective = min(len(weighted), total)
            # FT8 reports come from unknown antennas and noise environments.
            # They may establish that a band/direction is open, but must never
            # manufacture a 20 dB path gain for an unrelated Kiwi.
            general_correction = mean * effective / (effective + 6.0)
            general_correction = max(
                -GENERAL_CALIBRATION_LIMIT_DB,
                min(GENERAL_CALIBRATION_LIMIT_DB, general_correction),
            )
            variance = (
                sum(weight * (value - mean) ** 2 for weight, value in weighted)
                / total
            )
            uncertainty = max(
                2.0, math.sqrt(variance + 9.0 / (effective + 1.0))
            )

        host_correction = 0.0
        if exact:
            latest = max(exact, key=lambda row: row.timestamp_utc)
            if latest.decoded and latest.measured_snr_db is not None:
                residual = max(
                    -HOST_CALIBRATION_LIMIT_DB,
                    min(HOST_CALIBRATION_LIMIT_DB, latest.residual_db),
                )
                # A decoded AETV transmission is direct, receiver-specific
                # evidence in the exact modem reference bandwidth. Trust it
                # ahead of population FT8 reports from unrelated stations.
                host_correction = residual
            else:
                # Exact-host non-decodes are useful as a temporary blacklist,
                # but are not generalized to other receivers or directions.
                host_correction = -6.0
        correction = max(
            -CALIBRATION_RESIDUAL_LIMIT_DB,
            min(
                CALIBRATION_RESIDUAL_LIMIT_DB,
                general_correction + host_correction,
            ),
        )
        return correction, uncertainty, len(weighted) + len(exact)


class PropagationPredictor:
    def __init__(self, store: CalibrationStore | None = None):
        self.store = store or CalibrationStore()

    def predict(
        self,
        receiver: KiwiReceiver,
        tx_lat: float,
        tx_lon: float,
        frequency_mhz: float,
        tx_power_w: float,
        when: datetime | None = None,
        weather: SpaceWeather | None = None,
        tx_gain_dbi: float = 0.0,
    ) -> PathEstimate:
        when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
        weather = weather or fetch_space_weather()
        distance = great_circle_km(tx_lat, tx_lon, receiver.lat, receiver.lon)
        bearing = initial_bearing_deg(tx_lat, tx_lon, receiver.lat, receiver.lon)
        native, _root = native_runtime_status(when.month)
        if native:
            try:
                raw = self._predict_native(
                    receiver,
                    tx_lat,
                    tx_lon,
                    frequency_mhz,
                    tx_power_w,
                    when,
                    weather,
                    tx_gain_dbi,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                raw = self._predict_fallback(
                    distance, frequency_mhz, tx_power_w, when, weather, tx_gain_dbi
                )
                raw["engine"] = "coarse fallback (P.533 error)"
        else:
            raw = self._predict_fallback(
                distance, frequency_mhz, tx_power_w, when, weather, tx_gain_dbi
            )
        correction, calibrated_uncertainty, samples = self.store.correction(
            receiver.host, bearing, frequency_mhz
        )
        # P.533 is a monthly median. Kp is used as a present-time disturbance
        # penalty, while calibration learns the systematic local remainder.
        storm_penalty = max(0.0, weather.kp - 3.0) * 1.5
        predicted = raw["snr"] - storm_penalty
        calibrated = predicted + correction
        uncertainty = max(raw["uncertainty"], calibrated_uncertainty)
        z = (calibrated - TARGET_SNR_DB) / max(0.5, uncertainty)
        probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return PathEstimate(
            host=receiver.host,
            when_utc=when.isoformat(),
            distance_km=distance,
            bearing_deg=bearing,
            predicted_snr_db=predicted,
            calibrated_snr_db=calibrated,
            uncertainty_db=uncertainty,
            success_probability=100.0 * probability,
            muf_mhz=raw["muf"],
            takeoff_deg=raw["takeoff"],
            reliability_pct=raw["reliability"],
            correction_db=correction,
            samples=samples,
            engine=raw["engine"],
        )

    def _predict_native(
        self, receiver, tx_lat, tx_lon, frequency, power, when, weather, tx_gain_dbi
    ):
        root = _runtime_root()
        tx_db_kw = 10.0 * math.log10(max(float(power), 0.001) / 1000.0)
        hour = when.hour + 1
        data_path = str(root / "Data") + os.sep
        report_options = (
            "RPT_D | RPT_ELE | RPT_BMUF | RPT_OPMUF | RPT_PR | RPT_SNR | "
            "RPT_SNRD | RPT_SNRXX | RPT_RSN | RPT_BCR | RPT_DOMMODE | RPT_RXLOCATION"
        )
        config = f'''PathName "AETV path"
PathTXName "AETV TX"
Path.L_tx.lat {tx_lat:.6f}
Path.L_tx.lng {tx_lon:.6f}
TXAntFilePath "ISOTROPIC"
TXGOS {tx_gain_dbi:.3f}
PathRXName "{receiver.host[:80]}"
Path.L_rx.lat {receiver.lat:.6f}
Path.L_rx.lng {receiver.lon:.6f}
RXAntFilePath "ISOTROPIC"
RXGOS 0.0
AntennaOrientation "TX2RX"
Path.year {when.year}
Path.month {when.month}
Path.hour {hour}
Path.SSN {max(1, min(311, int(round(weather.ssn))))}
Path.frequency {frequency:.6f}
Path.txpower {tx_db_kw:.6f}
Path.BW 2500.0
Path.SNRr {TARGET_SNR_DB:.1f}
Path.SNRXXp 50
Path.ManMadeNoise "RURAL"
Path.Modulation "DIGITAL"
Path.SIRr 0.0
Path.A 10.0
Path.TW 10.0
Path.FW 2.0
Path.T0 10.0
Path.F0 2.0
Path.SorL "SHORTPATH"
RptFileFormat "{report_options}"
LL.lat {receiver.lat:.6f}
LL.lng {receiver.lon:.6f}
LR.lat {receiver.lat:.6f}
LR.lng {receiver.lon:.6f}
UL.lat {receiver.lat:.6f}
UL.lng {receiver.lon:.6f}
UR.lat {receiver.lat:.6f}
UR.lng {receiver.lon:.6f}
latinc 1.0
lnginc 1.0
DataFilePath "{data_path}"
'''
        with tempfile.TemporaryDirectory(prefix="aetv-p533-") as folder:
            input_path = Path(folder) / "path.in"
            output_path = Path(folder) / "path.csv"
            input_path.write_text(config, encoding="utf-8")
            proc = subprocess.run(
                [str(root / "ITURHFProp.exe"), "-s", "-c", str(input_path), str(output_path)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode or not output_path.is_file():
                raise RuntimeError((proc.stderr or proc.stdout or "P.533 failed").strip())
            with output_path.open(newline="", encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
        upper = float(row.get("DuSN", row["SNR"]))
        lower = float(row.get("DlSN", row["SNR"]))
        spread = abs(upper - lower) / 2.56
        return {
            "snr": float(row["SNR"]),
            "uncertainty": max(2.0, spread),
            "muf": float(row.get("BMUF", 0.0)),
            "takeoff": float(row.get("ele", 0.0)),
            "reliability": float(row.get("BCR", 0.0)),
            "engine": "ITU-R P.533-14",
        }

    @staticmethod
    def _predict_fallback(distance, frequency, power, when, weather, tx_gain_dbi=0.0):
        # Coarse single-hop geometry plus day/night and inverse-distance terms.
        # Its purpose is ordering, not claiming P.533 accuracy.
        distance = max(50.0, distance)
        hop_height = 300.0
        takeoff = math.degrees(math.atan2(2.0 * hop_height, distance))
        local_phase = 2.0 * math.pi * (when.hour / 24.0)
        daytime = 0.5 + 0.5 * math.cos(local_phase - math.pi)
        muf = 5.0 + 0.025 * weather.ssn + 5.0 * daytime
        frequency_penalty = 2.5 * abs(math.log(max(frequency, 1.6) / max(muf * 0.8, 1.6)))
        snr = (
            24.0
            + 10.0 * math.log10(max(power, 0.1) / 5.0)
            + tx_gain_dbi
            - 11.0 * math.log10(distance / 250.0)
            - frequency_penalty
        )
        return {
            "snr": snr,
            "uncertainty": 8.0,
            "muf": muf,
            "takeoff": takeoff,
            "reliability": 50.0,
            "engine": "coarse fallback",
        }


def record_ota_measurement(
    receiver: KiwiReceiver,
    tx_lat: float,
    tx_lon: float,
    frequency_mhz: float,
    tx_power_w: float,
    tx_gain_dbi: float,
    measured_snr_db: float,
    callsign: str,
    when: datetime | None = None,
    store: CalibrationStore | None = None,
) -> ProbeMeasurement:
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    predictor = PropagationPredictor(store)
    estimate = predictor.predict(
        receiver,
        tx_lat,
        tx_lon,
        frequency_mhz,
        tx_power_w,
        tx_gain_dbi=tx_gain_dbi,
        when=when,
    )
    measurement = ProbeMeasurement(
        timestamp_utc=when.isoformat(),
        host=receiver.host,
        receiver_lat=receiver.lat,
        receiver_lon=receiver.lon,
        frequency_mhz=frequency_mhz,
        bearing_deg=estimate.bearing_deg,
        measured_snr_db=float(measured_snr_db),
        predicted_snr_db=estimate.predicted_snr_db,
        tx_power_w=float(tx_power_w),
        callsign=callsign,
    )
    predictor.store.append(measurement)
    return measurement


def record_ota_failure(
    receiver: KiwiReceiver,
    tx_lat: float,
    tx_lon: float,
    frequency_mhz: float,
    tx_power_w: float,
    tx_gain_dbi: float,
    callsign: str,
    when: datetime | None = None,
    store: CalibrationStore | None = None,
) -> ProbeMeasurement:
    """Record a completed local transmission that the selected Kiwi did not decode."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    predictor = PropagationPredictor(store)
    estimate = predictor.predict(
        receiver,
        tx_lat,
        tx_lon,
        frequency_mhz,
        tx_power_w,
        tx_gain_dbi=tx_gain_dbi,
        when=when,
    )
    measurement = ProbeMeasurement(
        timestamp_utc=when.isoformat(),
        host=receiver.host,
        receiver_lat=receiver.lat,
        receiver_lon=receiver.lon,
        frequency_mhz=frequency_mhz,
        bearing_deg=estimate.bearing_deg,
        measured_snr_db=None,
        predicted_snr_db=estimate.predicted_snr_db,
        tx_power_w=float(tx_power_w),
        callsign=callsign,
        decoded=False,
    )
    predictor.store.append(measurement)
    return measurement
