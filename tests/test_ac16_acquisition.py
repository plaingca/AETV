"""AC16 payload phase must survive a repeated-preamble peak ambiguity."""

import numpy as np
import pytest

import aetv.modem as modem
from aetv.config import AETV_MODES
from aetv.sync import Acquisition
from aetv.hfchannel import freq_shift


@pytest.mark.parametrize("shift", [-1, 1])
@pytest.mark.parametrize("block_size", [4800, 250000])
def test_repeated_preamble_alias_preserves_payload_chronology(
    monkeypatch, shift, block_size
):
    mode = AETV_MODES["AC16"]
    sent = np.random.default_rng(44).standard_normal((3, 19200)).astype(np.float32)
    waveform = np.concatenate(list(modem.modulate_continuous_chunks(sent, "AC16")))
    # Also exercise a callback with a long pre-transmission noise interval:
    # the initial search prefix can end before the complete first GOP.
    audio = np.concatenate([np.zeros(60000), waveform])
    acquire = modem.acquire
    useful_symbol_samples = modem._band_params("A")[2]

    def ambiguous_peak(*args, **kwargs):
        result = acquire(*args, **kwargs)
        return Acquisition(
            result.preamble_start + shift * useful_symbol_samples,
            result.freq_offset,
            result.metric,
        )

    monkeypatch.setattr(modem, "acquire", ambiguous_peak)
    receiver = modem.StreamingDemodulator("A", continuous=True, mode_name="AC16")
    received = []
    for start in range(0, len(audio), block_size):
        for result in receiver.feed(audio[start : start + block_size]):
            received.extend(result.gops_latents)
    assert np.asarray(received).shape == sent.shape
    for tx, rx in zip(sent, received):
        assert np.corrcoef(tx, rx)[0, 1] > 0.9


@pytest.mark.parametrize("block_size", [4800, 96000])
def test_ac16_late_entry_at_quarter_carrier_offset(block_size):
    sent = np.random.default_rng(439).standard_normal((18, 19200)).astype(np.float32)
    waveform = np.concatenate(list(modem.modulate_continuous_chunks(sent, "AC16", "VE7TEST")))
    audio = freq_shift(waveform, 12.5, fs=48000)[150000:]
    events = []
    receiver = modem.StreamingDemodulator(
        "A", continuous=True, mode_name="AC16", boundary_tracking=True,
        on_debug=events.append,
    )
    received = []
    for start in range(0, len(audio), block_size):
        for result in receiver.feed(audio[start:start + block_size]):
            assert result.callsign == "VE7TEST"
            received.extend(result.gops_latents)
    assert len(received) >= 3
    assert any(event["event"] == "blind_acquired" for event in events)
    correlation = np.asarray(received) @ sent.T
    correlation /= np.linalg.norm(received, axis=1)[:, None] * np.linalg.norm(sent, axis=1)
    assert np.all(correlation.max(axis=1) > 0.90)
    assert np.all(np.diff(correlation.argmax(axis=1)) == 1)


def test_ac16_startup_history_is_bounded_and_keeps_future_preamble():
    receiver = modem.StreamingDemodulator("A", continuous=True, mode_name="AC16")
    for _ in range(16):
        assert receiver.feed(np.zeros(48000)) == []
    assert len(receiver.buffer) <= 12 * 48000
    sent = np.random.default_rng(101).standard_normal((2, 19200)).astype(np.float32)
    waveform = np.concatenate(list(modem.modulate_continuous_chunks(sent, "AC16")))
    received = []
    for start in range(0, len(waveform), 96000):
        for result in receiver.feed(waveform[start:start + 96000]):
            received.extend(result.gops_latents)
    assert len(received) == len(sent)
    assert all(np.corrcoef(a, b)[0, 1] > 0.9 for a, b in zip(sent, received))
