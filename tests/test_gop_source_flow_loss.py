import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_source_flow_loss import (  # noqa: E402
    forward_backward_validity,
    motion_compensated_residual_loss,
    weight_token,
)


def test_forward_backward_zero_flow_is_valid():
    forward = torch.zeros(2, 2, 8, 10)
    backward = torch.zeros_like(forward)
    valid = forward_backward_validity(forward, backward)
    assert valid.shape == (2, 1, 8, 10)
    assert torch.equal(valid, torch.ones_like(valid))


def test_motion_compensated_residual_is_zero_for_source_reconstruction():
    source = torch.rand(1, 3, 12, 8, 10)
    flow = torch.zeros(1, 2, 8, 10)
    valid = torch.ones(1, 1, 8, 10)
    source_residual = source[:, :, 6] - source[:, :, 5]
    loss = motion_compensated_residual_loss(
        source, source_residual, flow, valid, 6
    )
    assert float(loss) < 1e-7


def test_motion_compensated_residual_penalizes_frozen_boundary():
    reconstruction = torch.zeros(1, 3, 12, 8, 10)
    source_residual = torch.full((1, 3, 8, 10), 0.25)
    flow = torch.zeros(1, 2, 8, 10)
    valid = torch.ones(1, 1, 8, 10)
    loss = motion_compensated_residual_loss(
        reconstruction, source_residual, flow, valid, 6
    )
    assert float(loss) > 0.24


def test_weight_token_is_filename_safe():
    assert weight_token(0.1) == "0p1"
    assert weight_token(0.25) == "0p25"
