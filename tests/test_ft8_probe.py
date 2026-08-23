from pathlib import Path

import pytest
from scipy.io import wavfile

from aetv.ft8_probe import (
    Ft8ProbeRun,
    Ft8Spot,
    generate_ft8_wav,
    ft8_runtime_path,
    import_no_report_calibration,
    maidenhead_coordinates,
    maidenhead_grid,
    parse_ft8_frequencies,
    parse_psk_reporter_xml,
    spots_for_probe_runs,
)
from aetv.propagation import CalibrationStore
from aetv.settings import StationSettings


def test_maidenhead_round_trip_contains_station():
    grid = maidenhead_grid(49.268, -124.78, 4)
    assert grid == "CN79"
    lat, lon = maidenhead_coordinates(grid)
    assert 49.0 <= lat <= 50.0
    assert -126.0 <= lon <= -124.0


def test_ft8_frequency_list_is_sanitized():
    assert parse_ft8_frequencies("7.074, bad; 14.074, 31, 7.074") == [7.074, 14.074]


def test_psk_reporter_xml_and_probe_slot_filtering():
    payload = b'''<receptionReports>
      <receptionReport receiverCallsign="K7ABC" receiverLocator="CN87"
        senderCallsign="VA7EET" frequency="7075002" flowStartSeconds="1700000000"
        sNR="-11" mode="FT8" />
      <receptionReport receiverCallsign="F1ABC" receiverLocator="JN18"
        senderCallsign="VA7EET" frequency="14075000" flowStartSeconds="1700000900"
        sNR="-18" mode="FT8" />
    </receptionReports>'''
    spots = parse_psk_reporter_xml(payload)
    assert len(spots) == 2
    assert spots[0].snr_db == -11
    run = Ft8ProbeRun(1700000000, "VA7EET", "CN79", 7.074, 1000.0, 100.0)
    assert spots_for_probe_runs(spots, [run]) == [spots[0]]


def test_no_report_probe_is_recorded_once_and_heard_probe_is_not(monkeypatch, tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    monkeypatch.setattr("aetv.ft8_probe.CalibrationStore", lambda: store)
    runs = [
        Ft8ProbeRun(1700000000, "VA7EET", "CN79", 7.074, 1000.0, 100.0),
        Ft8ProbeRun(1700000090, "VA7EET", "CN79", 28.074, 1000.0, 100.0),
    ]
    spots = [Ft8Spot("K7ABC", "CN87", "VA7EET", 7075000, -10.0, 1700000000)]
    settings = StationSettings(callsign="VA7EET", kiwi_lat=49.268, kiwi_lon=-124.78)
    assert import_no_report_calibration(spots, runs, settings) == 1
    assert import_no_report_calibration(spots, runs, settings) == 1
    rows = store.load()
    assert len(rows) == 1
    assert rows[0].frequency_mhz == pytest.approx(28.075)
    assert rows[0].decoded is False
    assert rows[0].directional is False


def test_repeated_same_band_probes_match_spot_to_nearest_slot(monkeypatch, tmp_path):
    store = CalibrationStore(tmp_path / "measurements.json")
    monkeypatch.setattr("aetv.ft8_probe.CalibrationStore", lambda: store)
    runs = [
        Ft8ProbeRun(1700000000, "VA7EET", "CN79", 28.074, 1000.0, 100.0),
        Ft8ProbeRun(1700000060, "VA7EET", "CN79", 28.074, 1000.0, 100.0),
    ]
    spots = [Ft8Spot("K7ABC", "CN87", "VA7EET", 28075000, -10.0, 1700000002)]
    settings = StationSettings(callsign="VA7EET", kiwi_lat=49.268, kiwi_lon=-124.78)
    assert import_no_report_calibration(spots, runs, settings) == 1
    assert store.load()[0].timestamp_utc.startswith("2023-11-14T22:14:20")


@pytest.mark.skipif(not ft8_runtime_path().is_file(), reason="ft8_lib runtime not installed")
def test_pinned_ft8_lib_generates_standard_slot(tmp_path: Path):
    output = generate_ft8_wav("CQ VA7EET CN79", tmp_path / "probe.wav")
    sample_rate, audio = wavfile.read(output)
    assert sample_rate == 12000
    assert len(audio) == 15 * sample_rate
    assert max(abs(audio.min()), abs(audio.max())) > 30000
