import torch

from aetv.receiver_postprocess import (
    LinearFrameInterpolator,
    SNRConditionedDenoiser,
    SNRConditionedResidualDiffusion,
    interpolate_video,
)


def test_interpolation_retains_originals_and_doubles_rate():
    video = torch.tensor([0.0, 0.25, 1.0]).reshape(1, 1, 3, 1, 1)
    output = interpolate_video(
        video, LinearFrameInterpolator(), factor=2, scene_cut_threshold=1.0
    )
    assert output.shape == (1, 1, 5, 1, 1)
    torch.testing.assert_close(output[:, :, ::2], video)
    torch.testing.assert_close(output.flatten(), torch.tensor([0.0, 0.125, 0.25, 0.625, 1.0]))


def test_scene_cut_does_not_blend_unrelated_frames():
    video = torch.stack((torch.zeros(3, 2, 2), torch.ones(3, 2, 2)), dim=1).unsqueeze(0)
    output = interpolate_video(
        video, LinearFrameInterpolator(), factor=2, scene_cut_threshold=0.2
    )
    torch.testing.assert_close(output[:, :, 1], video[:, :, 0])


def test_snr_conditioned_diffusion_loss_and_restore_shapes():
    torch.manual_seed(4)
    denoiser = SNRConditionedDenoiser(width=8, condition_width=16)
    diffusion = SNRConditionedResidualDiffusion(denoiser, timesteps=8)
    degraded = torch.rand(2, 3, 3, 8, 8)
    clean = (degraded + 0.02 * torch.randn_like(degraded)).clamp(0, 1)
    snr = torch.tensor([6.0, 18.0])
    confidence = torch.tensor([0.4, 0.9])
    loss = diffusion.training_loss(clean, degraded, snr, confidence)
    assert loss.isfinite() and loss.item() > 0
    restored = diffusion.restore(degraded, snr, confidence, steps=2, seed=9)
    assert restored.shape == degraded.shape
    assert restored.min() >= 0 and restored.max() <= 1


def test_untrained_zero_output_denoiser_has_bounded_correction():
    diffusion = SNRConditionedResidualDiffusion(
        SNRConditionedDenoiser(width=8, condition_width=16),
        timesteps=4,
        max_correction=0.1,
    )
    degraded = torch.full((1, 3, 2, 8, 8), 0.5)
    restored = diffusion.restore(degraded, torch.tensor([6.0]), steps=2)
    assert (restored - degraded).abs().max() <= 0.100001
