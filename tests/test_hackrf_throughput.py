"""HackRF conversion must retain spectral protection while reducing host work."""
import numpy as np
import pytest
from scipy.signal import freqz

from aetv.hackrf import encode_iq
from aetv.sdr_dsp import IQDecimator, ModemToIQ


@pytest.mark.parametrize('rate,factor', [(9600000, 10), (8064000, 8)])
def test_hackrf_filters_protect_complete_tuning_range_and_reject_aliases(rate, factor):
    decimator = IQDecimator(factor, protected_fraction=150000 / (rate / factor))
    f, response = freqz(decimator.taps, worN=131072, fs=rate)
    db = 20 * np.log10(np.maximum(abs(response), 1e-15))
    assert np.ptp(db[f <= 150000]) < .002
    assert db[f >= rate / factor - 150000].max() < -80
    tx = ModemToIQ(rate, center_hz=10000, narrowband=True)
    f, response = freqz(tx.interpolation_taps / tx.factor, worN=262144, fs=rate)
    db = 20 * np.log10(np.maximum(abs(response), 1e-15))
    assert np.ptp(db[f <= 10000]) < .002
    assert db[f >= 38000].max() < -80


@pytest.mark.parametrize('offset', [100000, -100000, 100050, 100050.125])
def test_periodic_mixer_matches_direct_oscillator_across_uneven_chunks(offset):
    fast = ModemToIQ(9600000, offset_hz=offset, narrowband=True)
    reference = ModemToIQ(9600000, offset_hz=offset, narrowband=True)
    reference.radio_cycle = None
    rng = np.random.default_rng(82)
    for length in [1, 17, 239, 400, 71]:
        audio = rng.normal(0, .01, length)
        np.testing.assert_allclose(fast.feed(audio), reference.feed(audio), atol=1e-7, rtol=2e-6)


def test_narrowband_decimation_is_independent_of_usb_transfer_boundaries():
    args = dict(protected_fraction=150000 / 960000)
    full, streaming = IQDecimator(10, **args), IQDecimator(10, **args)
    t = np.arange(200003) / 9600000
    iq = (.2 * np.exp(-2j*np.pi*125000*t) + .5*np.exp(2j*np.pi*850000*t)).astype(np.complex64)
    expected = full.feed(iq)
    chunks = np.split(iq, [1, 97, 131071, 199992])
    np.testing.assert_allclose(np.concatenate([streaming.feed(x) for x in chunks]), expected,
                               atol=1e-7, rtol=2e-6)


def test_packed_iq_retains_original_rounding_at_8bit_boundaries():
    values = np.linspace(-.99999, .99999, 10000, dtype=np.float32)
    iq = (values + 1j*values[::-1]).astype(np.complex64)
    reference = np.clip(np.rint(iq.view(np.float32)*128), -128, 127).astype(np.int8).tobytes()
    assert encode_iq(iq) == reference
    for value in [np.nan, np.inf, -np.inf, 1., -1.]:
        with pytest.raises(ValueError):
            encode_iq(np.array([complex(value, 0)], np.complex64))
