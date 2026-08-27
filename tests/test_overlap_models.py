import pytest
import torch

from aetv.config import AETV_MODES
from aetv.overlap_models import (
    OverlappingGOPAutoencoder,
    OverlappingGOPDecoder,
    OverlappingGOPEncoder,
    join_video_gops,
    split_video_gops,
)


def test_split_join_gops_is_exact_inverse():
    video = torch.rand(2, 3, 30, 16, 24)
    gops = split_video_gops(video, 6)
    assert gops.shape == (2, 5, 3, 6, 16, 24)
    assert torch.equal(join_video_gops(gops), video)


def test_invalid_window_size_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        OverlappingGOPDecoder(mode="V8", width=32, window_gops=4)


def test_fixed_rate_and_center_output_contract():
    mode = AETV_MODES["V8"]
    model = OverlappingGOPAutoencoder(
        mode=mode, width=32, latent_channels=3, window_gops=3
    )
    # Use the native spatial size because decoder geometry is part of the wire
    # contract; five GOPs should produce three emitted center GOPs.
    video = torch.rand(1, 3, 30, mode.height, mode.width)
    with torch.inference_mode():
        latents = model.encode_sequence(video)
        output = model.decode_sequence(latents)
    assert latents.shape == (1, 5, 2816)
    assert output.shape == (1, 3, 18, mode.height, mode.width)
    assert model.target_for_sequence(video).shape == output.shape
    assert model.lookahead_gops == 1
    assert model.config()["wire_values_per_second"] == 2816


def test_five_gop_window_emits_center_three_in_one_pass():
    mode = AETV_MODES["V8"]
    model = OverlappingGOPAutoencoder(
        mode=mode, width=32, latent_channels=3, window_gops=5, emit_gops=3
    )
    video = torch.rand(1, 3, 30, mode.height, mode.width)
    with torch.inference_mode():
        latents = model.encode_sequence(video)
        output = model.decode_sequence(latents)
    assert latents.shape == (1, 5, 2816)
    assert output.shape == (1, 3, 18, mode.height, mode.width)
    assert model.target_for_sequence(video).shape == output.shape
    assert model.lookahead_gops == 1
    assert model.config()["emit_gops"] == 3


def test_two_overlapping_decode_calls_emit_six_contiguous_gops():
    mode = AETV_MODES["V8"]
    model = OverlappingGOPAutoencoder(
        mode=mode, width=32, latent_channels=3, window_gops=5, emit_gops=3
    )
    latents = torch.rand(1, 8, mode.latents_per_gop)
    source = torch.rand(1, 3, 48, mode.height, mode.width)
    with torch.inference_mode():
        output = model.decode_sequence(latents)
    assert output.shape == (1, 3, 36, mode.height, mode.width)
    assert model.target_for_sequence(source).shape == output.shape


def test_every_transmitted_coordinate_reaches_learned_unpacker():
    model = OverlappingGOPDecoder(mode="V8", width=32, latent_channels=3)
    assert model.latent_unpack.in_features == 2816
    assert model.latent_unpack.out_features == 2808


def test_wide_encoder_learns_full_grid_to_fixed_budget_pack():
    model = OverlappingGOPEncoder(mode="V8", width=32, latent_channels=8)
    assert model.latent_pack.in_features == 8 * 3 * 14 * 24
    assert model.latent_pack.out_features == 2816
    video = torch.rand(1, 3, 6, 108, 192)
    with torch.inference_mode():
        latent = model(video)
    assert latent.shape == (1, 2816)
    assert torch.allclose(latent.pow(2).mean(), torch.tensor(1.0), atol=2e-5)


def test_missing_context_weight_masks_its_latent_values():
    model = OverlappingGOPDecoder(mode="V8", width=32, latent_channels=3)
    latents = torch.rand(1, 3, 2816)
    weights = torch.ones_like(latents)
    weights[:, 0] = 0
    changed = latents.clone()
    changed[:, 0] = 1000
    with torch.inference_mode():
        first = model(latents, weights)
        second = model(changed, weights)
    assert torch.equal(first, second)
