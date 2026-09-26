"""Shared scoring protocol: split hygiene, receiver scaling and paired statistics."""

import math

import torch

from aetv.shared_eval import (
    EVAL_CLIPS,
    eval_split,
    masked_psnr,
    paired,
    scale_confidence,
    summarize,
    training_pool,
)


def test_flat_confidence_is_not_scaled():
    flat = torch.full((2, 2816), 0.9)
    assert torch.equal(scale_confidence(flat, 1.75), flat)


def test_varying_confidence_is_scaled_and_clamped():
    weights = torch.tensor([[0.1, 0.4, 0.8]])
    scaled = scale_confidence(weights, 1.5)
    assert torch.allclose(scaled, torch.tensor([[0.15, 0.6, 1.0]]))


def test_scaling_is_per_row():
    weights = torch.tensor([[0.5, 0.5, 0.5], [0.2, 0.5, 0.9]])
    scaled = scale_confidence(weights, 2.0)
    assert torch.equal(scaled[0], weights[0])
    assert torch.allclose(scaled[1], torch.tensor([0.4, 1.0, 1.0]))


def test_training_pool_never_contains_eval_clips(tmp_path):
    for index in range(EVAL_CLIPS + 20):
        (tmp_path / f"clip{index:03d}.pt").touch()
    evaluation = set(eval_split(tmp_path))
    pool = set(training_pool(tmp_path))
    assert len(evaluation) == EVAL_CLIPS
    assert len(pool) == 20
    assert not evaluation & pool


def test_paired_standard_error():
    stats = paired([2.0, 3.0, 5.0, None], [1.0, 1.0, 1.0, 4.0])
    assert stats["n"] == 3
    assert math.isclose(stats["mean"], 7.0 / 3.0)
    assert math.isclose(stats["se"], torch.tensor([1.0, 2.0, 4.0]).std().item() / math.sqrt(3), rel_tol=1e-6)


def test_summarize_ignores_missing_values():
    assert summarize([None, None])["n"] == 0
    assert summarize([1.0, 3.0])["mean"] == 2.0


def test_masked_psnr_uses_only_masked_pixels():
    reference = torch.zeros(3, 2, 4, 4)
    reconstruction = reference.clone()
    reconstruction[:, :, :2] = 0.1
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, 2:] = True
    assert masked_psnr(reference, reconstruction, mask) == 100.0
    mask[:, :2] = True
    assert math.isclose(masked_psnr(reference, reconstruction, mask), 10 * math.log10(1 / 0.005), rel_tol=1e-5)
    assert masked_psnr(reference, reconstruction, torch.zeros(2, 4, 4, dtype=torch.bool)) is None
