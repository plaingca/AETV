"""AC16 payload phase must survive a repeated-preamble peak ambiguity."""

import numpy as np
import pytest

import aetv.modem as modem
from aetv.config import AETV_MODES
from aetv.sync import Acquisition


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
