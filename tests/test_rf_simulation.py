"""Bounded regression tests for actual waveform RF simulation."""
import numpy as np
import pytest
from aetv.hfchannel import FadingPreset, fading
from scripts.validate_rf_simulation import Scenario, impair, run_trial

@pytest.mark.parametrize('count', [0, 1, 8, 16, 17, 100])
def test_multipath_preserves_short_and_empty_waveform_lengths(count):
    audio = np.sin(np.arange(count, dtype=float))
    result = fading(audio, 'mpp', seed=7, fs=8000)
    assert result.shape == audio.shape
    assert np.all(np.isfinite(result))


def test_zero_delay_multipath_is_supported():
    result = fading(np.ones(8), FadingPreset('zero', 1., 0.), seed=3)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_impairments_are_seeded_and_sample_clock_changes_duration():
    audio = np.sin(2*np.pi*700*np.arange(16000)/8000)
    scenario = Scenario('combined', 12, 'mpp', 25, 100)
    first = impair(audio, scenario, 8000, 1500, 2)
    second = impair(audio, scenario, 8000, 1500, 2)
    np.testing.assert_array_equal(first, second)
    assert len(first) > len(audio)
    assert np.all(np.isfinite(first))


@pytest.mark.parametrize('mode', ['V8', 'V9', 'V7', 'AC16'])
def test_release_modes_recover_distinct_clean_gops(mode):
    row = run_trial(mode, Scenario('clean'), seed=0, gops=2)
    assert row['matched_unique_gops'] == 2
    assert row['unmatched_outputs'] == 0
    assert row['mean_correlation'] > .9
    assert row['mean_nmse'] < .2
