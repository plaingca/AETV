import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_boundaries import (  # noqa: E402
    boundary_indices,
    boundary_losses,
    decode_independent_gops,
    encode_independent_gops,
    join_gops,
    rgb_to_ycbcr_delta,
    runtime_retry_seed,
    split_gops,
)


def test_split_and_join_gops_round_trip():
    sequence = torch.arange(2 * 3 * 12 * 4 * 5).reshape(2, 3, 12, 4, 5)
    separated = split_gops(sequence, 6)
    assert separated.shape == (4, 3, 6, 4, 5)
    assert torch.equal(join_gops(separated, batch=2, count=2), sequence)


def test_boundary_indices_exclude_within_gop_transitions():
    assert boundary_indices(18, 6) == [6, 12]


def test_boundary_losses_are_zero_for_exact_reconstruction():
    target = torch.rand(2, 3, 12, 8, 8)
    losses = boundary_losses(target.clone(), target, frames_per_gop=6)
    error_terms = {
        name: value
        for name, value in losses.items()
        if not name.endswith("_ratio")
    }
    assert all(value.item() == 0 for value in error_terms.values())
    assert torch.isclose(losses["within_motion_ratio"], torch.tensor(1.0))
    assert torch.isclose(losses["spatial_detail_ratio"], torch.tensor(1.0))


def test_boundary_delta_loss_scores_only_excess_reconstruction_jump():
    target = torch.zeros(1, 3, 12, 8, 8)
    recon = target.clone()
    recon[:, :, 6:] = 0.25
    losses = boundary_losses(recon, target, frames_per_gop=6)
    assert torch.isclose(losses["boundary_rgb_delta"], torch.tensor(0.25))
    assert torch.isclose(losses["boundary_excess"], torch.tensor(0.25))
    assert losses["within_gop_temporal_error"].item() == 0


def test_boundary_acceleration_scores_both_triplets_touching_join():
    target = torch.zeros(1, 3, 12, 8, 8)
    recon = target.clone()
    recon[:, :, 6:] = 0.25
    losses = boundary_losses(recon, target, frames_per_gop=6)
    # The step contributes +0.25 before the join and -0.25 after it.
    assert torch.isclose(losses["boundary_acceleration"], torch.tensor(0.25))


def test_lowpass_y_and_chroma_are_explicit_source_referenced_terms():
    target = torch.zeros(1, 3, 12, 16, 16)
    recon = target.clone()
    recon[:, 0, 6:] = 0.2
    losses = boundary_losses(recon, target, frames_per_gop=6)
    assert losses["boundary_lowpass_y"].item() > 0
    assert losses["boundary_lowpass_chroma"].item() > 0


def test_gradient_delta_ignores_spatially_constant_boundary_bias():
    target = torch.zeros(1, 3, 12, 8, 8)
    recon = target.clone()
    recon[:, :, 6:] = 0.25
    losses = boundary_losses(recon, target, frames_per_gop=6)
    assert losses["boundary_gradient_delta"].item() == 0


def test_ycbcr_delta_has_no_constant_offsets():
    black_delta = torch.zeros(1, 3, 1, 2, 2)
    assert torch.equal(rgb_to_ycbcr_delta(black_delta), black_delta)


def test_runtime_retry_seed_is_deterministic_and_distinct():
    assert runtime_retry_seed(42, 0) == 42
    assert runtime_retry_seed(42, 1) == 104771


class _CountingCodec:
    def __init__(self):
        self.encoder_calls = 0
        self.decoder_calls = 0

    def encoder(self, value):
        self.encoder_calls += 1
        return value.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)

    def decoder(self, value, weights, output_shape):
        self.decoder_calls += 1
        frames, height, width = output_shape
        return (value * weights)[:, :1, None, None, None].expand(-1, 3, frames, height, width)


def test_codec_uses_two_explicit_encoder_and_decoder_calls():
    codec = _CountingCodec()
    separated = torch.ones(2, 3, 6, 4, 5)
    encoded = encode_independent_gops(codec, separated)
    assert encoded.shape == (2, 1)
    assert codec.encoder_calls == 2

    class _Mode:
        gop_frames = 6
        height = 4
        width = 5

    received = torch.ones(2, 2, 1)
    decoded = decode_independent_gops(codec, received, torch.ones_like(received), _Mode())
    assert decoded.shape == (4, 3, 6, 4, 5)
    assert codec.decoder_calls == 2
