import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_frame_overlap import (  # noqa: E402
    fixed_rate_confidence,
    fuse_overlapping_gops,
    overlap_seams,
    overlapping_windows,
    pad_for_overlap_coverage,
    transition_metrics,
)


def test_overlapping_windows_share_exactly_one_frame():
    video = torch.arange(16).reshape(1, 1, 16, 1, 1)
    windows = overlapping_windows(video, window_frames=6, stride_frames=5)
    assert windows.shape == (1, 3, 1, 6, 1, 1)
    assert windows[0, 0, 0, -1].item() == windows[0, 1, 0, 0].item()
    assert windows[0, 1, 0, -1].item() == windows[0, 2, 0, 0].item()


def test_terminal_padding_covers_noncongruent_clip_without_changing_body():
    video = torch.arange(360).reshape(1, 1, 360, 1, 1)
    padded = pad_for_overlap_coverage(video)
    assert padded.shape[2] == 361
    assert torch.equal(padded[:, :, :360], video)
    assert padded[0, 0, -1].item() == video[0, 0, -1].item()
    windows = overlapping_windows(padded)
    assert fuse_overlapping_gops(windows).shape[2] == 361


def test_fusion_deduplicates_and_averages_transition_frame():
    decoded = torch.zeros(1, 2, 1, 6, 1, 1)
    decoded[:, 0, :, -1] = 2
    decoded[:, 1, :, 0] = 4
    decoded[:, 1, :, 1:] = 5
    output = fuse_overlapping_gops(decoded, theta=0.5)
    assert output.shape == (1, 1, 11, 1, 1)
    assert output[0, 0, 5].item() == 3
    assert torch.equal(output[0, 0, 6:], torch.full((5, 1, 1), 5.0))


def test_fixed_rate_mask_preserves_long_run_symbol_rate():
    latents = torch.ones(1, 12, 2816)
    confidence = fixed_rate_confidence(latents, pattern="uniform")
    assert confidence.sum().item() == 10 * 2816
    assert confidence.shape == latents.shape
    counts = confidence.sum(dim=-1).unique().tolist()
    assert counts == [2346.0, 2347.0]


def test_exact_reconstruction_has_zero_transition_errors():
    target = torch.rand(1, 3, 16, 8, 8)
    metrics = transition_metrics(
        target.clone(),
        target,
        overlap_seams(16),
        torch.device("cpu"),
        include_lpips=False,
    )
    for name in (
        "l1",
        "temporal_delta",
        "seam_in_delta",
        "seam_out_delta",
        "seam_two_sided_delta",
        "seam_lowpass_delta",
        "seam_acceleration",
        "within_delta",
    ):
        assert metrics[name] == 0
