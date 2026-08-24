import torch

from aetv.attention import (
    face_crop_grid,
    landmark_face_mask,
    normalized_region_weights,
    region_contrast_loss,
    region_detail_loss,
    region_gradient_loss,
    region_reconstruction_loss,
    sample_face_crops,
)


def test_region_weights_preserve_per_clip_loss_scale():
    mask = torch.zeros(2, 1, 3, 8, 8)
    mask[0, :, :, 2:5, 2:5] = 1
    mask[1, :, :, 1:7, 1:7] = 1

    weights = normalized_region_weights(mask, boost=3.0)

    assert torch.allclose(weights.mean(dim=(1, 2, 3, 4)), torch.ones(2))
    assert weights[0, 0, 0, 3, 3] > weights[0, 0, 0, 0, 0]


def test_region_losses_are_zero_for_exact_reconstruction():
    target = torch.rand(2, 3, 4, 12, 16)
    mask = torch.rand(2, 1, 4, 12, 16)

    assert region_reconstruction_loss(target, target, mask, 3.0) == 0
    assert region_gradient_loss(target, target, mask, 3.0) == 0
    assert region_detail_loss(target, target, mask, 3.0) == 0
    assert region_contrast_loss(target, target, mask, 3.0) == 0


def test_region_reconstruction_penalizes_masked_error_more():
    target = torch.zeros(1, 3, 2, 8, 8)
    mask = torch.zeros(1, 1, 2, 8, 8)
    mask[..., 2:6, 2:6] = 1
    inside = target.clone()
    outside = target.clone()
    inside[..., 3, 3] = 1
    outside[..., 0, 0] = 1

    assert region_reconstruction_loss(inside, target, mask, 3.0) > (
        region_reconstruction_loss(outside, target, mask, 3.0)
    )


def test_landmark_face_mask_focuses_features_without_box_expansion():
    # x, y, w, h; eyes, nose, mouth corners; score
    detection = torch.tensor(
        [[60, 20, 48, 60, 74, 42, 94, 42, 84, 54, 76, 66, 92, 66, 0.99]],
        dtype=torch.float32,
    ).numpy()

    mask = landmark_face_mask(detection, height=108, width=192)

    assert mask.shape == (108, 192)
    assert mask[42, 74] > 0.95  # eye hotspot
    assert mask[66, 76] > 0.9  # mouth-corner hotspot
    assert mask[50, 84] > mask[10, 84]  # face, not expanded forehead/background
    assert mask.mean() < 0.08  # far tighter than the old 15-44% rectangles


def test_face_crop_is_face_only_and_differentiable():
    video = torch.zeros(2, 3, 3, 32, 48, requires_grad=True)
    video.data[0, :, :, 10:22, 17:29] = 1.0
    mask = torch.zeros(2, 1, 3, 32, 48)
    mask[0, :, :, 11:21, 18:28] = 1.0
    grid, indices = face_crop_grid(mask, torch.tensor([True, False]), crop_size=24)
    crops = sample_face_crops(video, grid, indices)

    assert indices.tolist() == [0]
    assert crops.shape == (1, 3, 3, 24, 24)
    # Crop includes deliberate cheek/forehead context, but remains centered on
    # the synthetic face rather than the empty frame.
    assert crops.mean() > 0.25
    crops.mean().backward()
    assert video.grad is not None and video.grad[0].abs().sum() > 0
    assert video.grad[1].abs().sum() == 0
