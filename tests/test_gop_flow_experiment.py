import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_flow import (  # noqa: E402
    FlowContextConfig,
    _raft_padding,
    apply_flow_sequence,
    sample_padded_reference,
)


class IdentityAligner(nn.Module):
    def forward(self, reference, target):
        return reference


def test_raft_padding_satisfies_minimum_and_multiple_of_eight():
    left, right, top, bottom = _raft_padding(108, 192)
    assert (left, right, top, bottom) == (0, 0, 10, 10)
    for height, width in ((32, 45), (129, 193), (256, 256)):
        left, right, top, bottom = _raft_padding(height, width)
        assert height + top + bottom >= 128
        assert width + left + right >= 128
        assert (height + top + bottom) % 8 == 0
        assert (width + left + right) % 8 == 0


def test_zero_flow_samples_unpadded_region_exactly():
    reference = torch.rand(2, 3, 8, 11)
    padded = torch.nn.functional.pad(reference, (2, 3, 4, 5), mode="replicate")
    flow = torch.zeros(2, 2, 8, 11)
    output = sample_padded_reference(
        padded, flow, left=2, top=4, output_height=8, output_width=11
    )
    assert torch.allclose(output, reference, atol=1e-6)


def test_first_gop_is_exact_bypass_and_later_gop_is_fused():
    base = torch.zeros(1, 2, 3, 6, 8, 8)
    base[:, 0] = 1.0
    config = FlowContextConfig(
        strength=0.5,
        photometric_threshold=2.0,
        photometric_softness=0.01,
        taper=(1, 1, 1, 1, 1, 1),
    )
    output = apply_flow_sequence(base, IdentityAligner(), config)
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.allclose(output[:, 1], torch.full_like(output[:, 1], 0.5))


def test_zero_boundary_confidence_is_exact_bypass():
    base = torch.rand(1, 3, 3, 6, 8, 8)
    confidence = torch.tensor([[1.0, 0.0, 1.0]])
    config = FlowContextConfig(
        strength=1.0,
        photometric_threshold=2.0,
        taper=(1, 1, 1, 1, 1, 1),
    )
    output = apply_flow_sequence(
        base, IdentityAligner(), config, confidences=confidence
    )
    assert torch.equal(output, base)


def test_invalid_strength_is_rejected():
    with pytest.raises(ValueError, match="strength"):
        FlowContextConfig(strength=1.1).validate()
