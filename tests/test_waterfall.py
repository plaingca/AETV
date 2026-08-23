import numpy as np

from aetv.gui.waterfall import automatic_levels, spectrum_dbfs


def test_spectrum_dbfs_is_fft_size_independent_for_a_tone():
    peaks = []
    for size in (512, 1024, 2048):
        phase = np.arange(size) * (2.0 * np.pi * 32.0 / size)
        dbfs, _ = spectrum_dbfs(np.sin(phase).astype(np.float32))
        peaks.append(float(dbfs.max()))
    assert max(abs(peak) for peak in peaks) < 0.05
    assert max(peaks) - min(peaks) < 0.01


def test_automatic_levels_keeps_flat_noise_out_of_highlights():
    rng = np.random.default_rng(7)
    noise = rng.normal(-57.0, 2.0, 512)
    floor, ceiling = automatic_levels(noise)
    normalized_noise = np.clip((noise - floor) / (ceiling - floor), 0.0, 1.0)
    assert ceiling - floor >= 32.0
    assert np.percentile(normalized_noise, 95) < 0.25


def test_automatic_levels_preserves_strong_carrier_contrast():
    values = np.full(512, -70.0, dtype=np.float32)
    values[200:208] = -28.0
    floor, ceiling = automatic_levels(values)
    assert floor <= -69.0
    assert ceiling >= -29.0
