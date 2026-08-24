from datetime import datetime, timezone
import json

import pytest

from aetv.kiwi import KiwiReceiver
from aetv.propagation import (
    CalibrationStore,
    ProbeMeasurement,
    PropagationPredictor,
    SpaceWeather,
    fetch_space_weather,
    frequency_search_radius_km,
    initial_bearing_deg,
    parse_planning_frequencies,
)


def _measurement(**overrides):
    values = dict(
        timestamp_utc="2026-08-23T19:45:00+00:00",
        host="kiwi.example:8073",
        receiver_lat=50.0,
        receiver_lon=-120.0,
        frequency_mhz=7.088,
        bearing_deg=60.0,
        measured_snr_db=14.0,
        predicted_snr_db=10.0,
        tx_power_w=5.0,
        callsign="VE7ABC",
    )
    values.update(overrides)
    return ProbeMeasurement(**values)


def test_initial_bearing_uses_great_circle_direction():
    assert initial_bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0)
    assert initial_bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0)


def test_frequency_search_radius_expands_for_higher_hf_bands():
    assert frequency_search_radius_km(3.588, 2500.0) == 2500.0
    assert frequency_search_radius_km(7.088, 2500.0) == 4000.0
    assert frequency_search_radius_km(14.088, 2500.0) == 9000.0
    assert frequency_search_radius_km(28.088, 2500.0) == 15000.0
    assert frequency_search_radius_km(28.088, 18000.0) == 18000.0


def test_planning_frequencies_are_sanitized_and_include_current_dial():
    assert parse_planning_frequencies("7.088, nope; 14.088, 31", 21.088) == [
        7.088,
        14.088,
        21.088,
    ]


def test_calibration_is_shrunk_and_direction_local(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(_measurement())

    correction, uncertainty, samples = store.correction(
        "kiwi.example:8073", 60.0, 7.088
    )
    assert correction == pytest.approx(4.0)
    assert uncertainty >= 2.0
    assert samples == 1

    opposite, _, opposite_samples = store.correction(
        "other.example:8073", 240.0, 7.088
    )
    assert opposite == 0.0
    assert opposite_samples == 0


def test_calibration_does_not_leak_between_ham_bands(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(_measurement(frequency_mhz=21.075, measured_snr_db=35.0))
    correction, _uncertainty, samples = store.correction(
        "other.example:8073", 60.0, 28.088
    )
    assert correction == 0.0
    assert samples == 0


def test_network_wide_miss_applies_in_every_direction(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(
        _measurement(
            host="pskr:no-report:28.074",
            frequency_mhz=28.075,
            measured_snr_db=None,
            predicted_snr_db=15.0,
            decoded=False,
            directional=False,
        )
    )
    north = store.correction("other.example:8073", 0.0, 28.088)
    south = store.correction("other.example:8073", 180.0, 28.088)
    assert north[0] < 0.0
    assert north == south


def test_repeated_gops_in_one_minute_do_not_overweight_calibration(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(_measurement(measured_snr_db=12.0))
    store.append(_measurement(timestamp_utc="2026-08-23T19:45:55+00:00", measured_snr_db=15.0))
    assert len(store.load()) == 1
    assert store.load()[0].measured_snr_db == 15.0


def test_failed_probe_is_censored_negative_evidence(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(
        _measurement(
            measured_snr_db=None,
            predicted_snr_db=15.0,
            decoded=False,
        )
    )
    row = store.load()[0]
    assert row.residual_db == pytest.approx(-12.0)
    correction, _uncertainty, samples = store.correction(
        "kiwi.example:8073", 60.0, 7.088
    )
    assert correction < -5.0
    assert samples == 1


def test_failed_receiver_does_not_poison_other_paths(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(
        _measurement(
            host="failed.example:8073",
            measured_snr_db=None,
            decoded=False,
        )
    )
    correction, _uncertainty, samples = store.correction(
        "other.example:8073", 60.0, 7.088
    )
    assert correction == 0.0
    assert samples == 0


def test_ft8_population_correction_cannot_invent_twenty_db_path(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    for index in range(100):
        store.append(
            _measurement(
                timestamp_utc=f"2026-08-23T19:{index // 2:02d}:{(index % 2) * 30:02d}+00:00",
                host=f"pskr:SPOT{index}",
                measured_snr_db=20.0,
                predicted_snr_db=-20.0,
            )
        )
    correction, _uncertainty, samples = store.correction(
        "unmeasured-kiwi.example:8073", 60.0, 7.088
    )
    assert correction == pytest.approx(6.0)
    assert samples == 100


def test_latest_exact_receiver_success_supersedes_older_miss(tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    store.append(
        _measurement(
            timestamp_utc="2026-08-23T19:45:00+00:00",
            measured_snr_db=None,
            decoded=False,
        )
    )
    store.append(
        _measurement(
            timestamp_utc="2026-08-23T19:46:00+00:00",
            measured_snr_db=14.0,
            predicted_snr_db=10.0,
        )
    )
    correction, _uncertainty, samples = store.correction(
        "kiwi.example:8073", 60.0, 7.088
    )
    assert correction > 2.0
    assert samples == 2


def test_space_weather_uses_current_month_r12_forecast(monkeypatch, tmp_path):
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    responses = {
        "predicted-solar-cycle.json": [
            {"time-tag": month, "predicted_ssn": 95.4}
        ],
        "noaa-planetary-k-index.json": [["time_tag", "Kp"], ["now", "1.33"]],
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(request, timeout=0):
        key = next(name for name in responses if name in request.full_url)
        return Response(responses[key])

    monkeypatch.setattr("aetv.propagation.config_dir", lambda: tmp_path)
    monkeypatch.setattr("aetv.propagation.urllib.request.urlopen", urlopen)
    # A pre-fix raw-SSN cache must be invalidated despite being recent.
    (tmp_path / "space_weather.json").write_text(
        json.dumps({"cached_at": datetime.now(timezone.utc).timestamp(), "ssn": 1, "kp": 9})
    )
    weather = fetch_space_weather()
    assert weather.ssn == pytest.approx(95.4)
    assert weather.kp == pytest.approx(1.33)
    assert weather.source == "NOAA SWPC R12 forecast"
    cache = json.loads((tmp_path / "space_weather.json").read_text())
    assert cache["solar_index"] == "R12"


def test_fallback_prediction_is_explicitly_labelled(monkeypatch, tmp_path):
    monkeypatch.setattr("aetv.propagation.native_runtime_status", lambda month=None: (False, tmp_path))
    receiver = KiwiReceiver("kiwi.example:8073", lat=50.7, lon=-120.38)
    estimate = PropagationPredictor(CalibrationStore(tmp_path / "empty.json")).predict(
        receiver,
        49.26,
        -123.11,
        7.088,
        5.0,
        when=datetime(2026, 8, 23, 20, tzinfo=timezone.utc),
        weather=SpaceWeather(80.0, 2.0, "test"),
    )
    assert estimate.engine == "coarse fallback"
    assert estimate.distance_km == pytest.approx(252.45, abs=0.1)
    assert 0.0 <= estimate.success_probability <= 100.0
