"""QRL? spectrum-check waveform and keyed transmit."""

import numpy as np
import pytest

from aetv.qrl import (
    MORSE,
    SLOT_HZ,
    TONE_HZ,
    keying_envelope,
    qrl_band,
    qrl_message,
    qrl_offsets,
    qrl_waveform,
)
from aetv.settings import StationSettings
from aetv.station import Station, TxEngine, TxPhase


@pytest.mark.parametrize(
    ("mode", "waveform_mode", "slots"),
    [
        ("V8", "video", 1),
        ("V7", "video", 4),
        ("AC16", "video", 6),
        ("V8", "analog_av", 2),
        ("AC16", "analog_av", 8),
    ],
)
def test_qrl_fills_selected_bandwidth_with_2k5_slots(mode, waveform_mode, slots):
    low, high, fs = qrl_band(StationSettings(mode=mode, waveform_mode=waveform_mode))
    offsets = qrl_offsets(low, high)

    assert offsets == [k * SLOT_HZ for k in range(slots)]
    assert all(low <= offset + TONE_HZ <= high for offset in offsets)
    assert max(offsets) + TONE_HZ < fs / 2


def test_twenty_khz_mode_matches_issue_example():
    low, high, _fs = qrl_band(StationSettings(mode="AC16", waveform_mode="analog_av"))
    assert qrl_offsets(low, high) == [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500]


def test_waveform_has_identical_tone_in_every_slot_and_nothing_else():
    fs = 48000
    offsets = qrl_offsets(0.0, 20000.0)
    wave = qrl_waveform("VE3XYZ", fs, offsets, peak=0.4)
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), 1 / fs)
    tones = [TONE_HZ + offset for offset in offsets]
    levels = [spectrum[np.argmin(np.abs(freqs - tone))] for tone in tones]

    assert np.max(np.abs(wave)) == pytest.approx(0.4, rel=1e-5)
    assert max(levels) / min(levels) < 1.05
    off_slot = np.all([np.abs(freqs - tone) > 200 for tone in tones], axis=0)
    assert spectrum[off_slot].max() < 0.01 * min(levels)


def test_keying_follows_morse_timing_with_soft_edges():
    fs, wpm = 8000, 20
    unit = round(fs * 1.2 / wpm)
    envelope = keying_envelope("E E", fs, wpm)
    keyed = envelope > 0.5

    assert np.count_nonzero(np.diff(keyed.astype(int)) == 1) == 2
    assert np.count_nonzero(keyed) == pytest.approx(2 * unit, abs=4)
    assert 0 < envelope[np.argmax(envelope > 0.01)] < 0.5
    assert envelope[0] == envelope[-1] == 0


def test_message_identifies_the_station():
    assert qrl_message(" ve3xyz/p ") == "QRL? DE VE3XYZ/P"
    assert set(qrl_message("VE3XYZ/P").replace(" ", "")) <= set(MORSE)


def test_empty_band_is_rejected():
    with pytest.raises(ValueError):
        qrl_waveform("N0CALL", 8000, [])


class _Ptt:
    def __init__(self, events):
        self.events = events

    def set_ptt(self, on):
        self.events.append(f"ptt:{on}")

    def describe(self):
        return "test PTT"

    def close(self):
        self.events.append("close")


def test_qrl_keys_ptt_around_leveled_audio_at_composite_rate(monkeypatch):
    events, played, states = [], [], []

    def play(stream, rate, **kwargs):
        chunks = list(stream)
        played.append((rate, np.concatenate(chunks)))
        events.append("play")
        kwargs["on_chunk"](len(chunks))
        return True

    monkeypatch.setattr("aetv.station.play_chunk_stream", play)
    settings = StationSettings(
        mode="AC16", waveform_mode="analog_av", callsign="VE3XYZ",
        tx_level=0.5, ptt_lead_s=0, ptt_tail_s=0,
    )
    engine = TxEngine(Station(settings), on_state=states.append, ptt=_Ptt(events))

    assert engine.transmit_qrl()
    assert events == ["ptt:True", "play", "ptt:False", "close"]
    rate, audio = played[0]
    assert rate == 48000
    assert np.max(np.abs(audio)) == pytest.approx(0.5, rel=1e-5)
    assert states[-1].phase == TxPhase.DONE
    assert "8 × 2.5 kHz slots" in states[-1].message
    assert "waterfall" in states[-1].message


def test_qrl_refuses_loopback_routes_without_keying(monkeypatch):
    events, errors = [], []
    monkeypatch.setattr(
        "aetv.station.play_chunk_stream",
        lambda *_args, **_kwargs: pytest.fail("loopback QRL must not play"),
    )
    settings = StationSettings(tx_channel_profile="mpp12")
    engine = TxEngine(Station(settings), on_error=errors.append, ptt=_Ptt(events))

    assert not engine.transmit_qrl()
    assert events == []
    assert errors == ["QRL? needs the Radio route"]
    assert engine.state.phase == TxPhase.FAILED


def test_qrl_cancelled_during_ptt_lead_unkeys_and_reports(monkeypatch):
    events = []
    monkeypatch.setattr(
        "aetv.station.play_chunk_stream",
        lambda *_args, **_kwargs: pytest.fail("cancelled QRL must not play"),
    )
    engine = TxEngine(Station(StationSettings(ptt_lead_s=5, ptt_tail_s=0)), ptt=_Ptt(events))
    engine._cancel.set()
    monkeypatch.setattr(engine._cancel, "clear", lambda: None)

    assert not engine.transmit_qrl()
    assert events == ["ptt:True", "ptt:False", "close"]
    assert engine.state.phase == TxPhase.CANCELLED


def test_qrl_uses_sdr_transport_for_pluto(monkeypatch):
    captured = {}

    def transport(chunks, fs, settings, cancel, on_progress, *, max_seconds, diagnostics):
        captured.update(fs=fs, audio=np.concatenate(list(chunks)), max_seconds=max_seconds)
        return True

    monkeypatch.setattr("aetv.sdr.transmit_pluto", transport)
    settings = StationSettings(mode="V7", tx_backend="pluto")
    engine = TxEngine(Station(settings))

    assert engine.transmit_qrl()
    assert captured["fs"] == 24000
    assert captured["max_seconds"] == pytest.approx(len(captured["audio"]) / 24000)
    assert engine.state.phase == TxPhase.DONE
    assert "4 × 2.5 kHz slots" in engine.state.message
