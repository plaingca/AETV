import torch

from aetv.audio_metrics import AudioPerceptualLoss


def test_audio_metrics_are_finite_and_zero_for_identical_waveforms():
    time = torch.arange(8_000) / 8_000
    audio = torch.sin(2 * torch.pi * 700 * time)[None]
    components = AudioPerceptualLoss(si_sdr_weight=0.0).components(audio, audio)

    assert all(torch.isfinite(value) for value in components.values())
    assert components["mr_stft"] < 1e-6
    assert components["mel"] < 1e-6
    assert components["waveform"] == 0
    assert components["si_sdr"] < -60
