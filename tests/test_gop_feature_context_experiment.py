import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_feature_context import (  # noqa: E402
    FeatureContextRefiner,
    apply_feature_sequence,
)


def test_fresh_feature_refiner_is_exact_noop():
    refiner = FeatureContextRefiner(feature_channels=8, width=8, blocks=1)
    current = torch.rand(2, 8, 6, 16, 16)
    previous = torch.rand_like(current)
    reliability = torch.rand(2, 1, 6, 16, 16)
    assert torch.equal(refiner(current, previous, reliability), current)


def test_missing_feature_state_is_exact_bypass_after_training_changes():
    refiner = FeatureContextRefiner(feature_channels=8, width=8, blocks=1)
    with torch.no_grad():
        refiner.output.bias.fill_(1.0)
    current = torch.rand(1, 8, 6, 16, 16)
    assert torch.equal(refiner(current, None), current)


def test_zero_confidence_is_exact_feature_bypass():
    refiner = FeatureContextRefiner(feature_channels=8, width=8, blocks=1)
    with torch.no_grad():
        refiner.output.bias.fill_(1.0)
    current = torch.rand(1, 8, 6, 16, 16)
    previous = torch.rand_like(current)
    output = refiner(current, previous, confidence=0.0)
    assert torch.equal(output, current)


def test_final_feature_frame_is_exact_bypass():
    refiner = FeatureContextRefiner(feature_channels=8, width=8, blocks=1)
    with torch.no_grad():
        refiner.output.bias.fill_(1.0)
    current = torch.rand(1, 8, 6, 16, 16)
    previous = torch.rand_like(current)
    output = refiner(current, previous)
    assert torch.equal(output[:, :, -1], current[:, :, -1])


def test_output_flow_strength_contract_is_bounded():
    base = torch.zeros(1, 1, 3, 6, 8, 8)
    features = torch.zeros(1, 1, 8, 6, 8, 8)
    skips = torch.zeros_like(base)
    with pytest.raises(ValueError, match="output flow strength"):
        apply_feature_sequence(
            None,
            None,
            None,
            base,
            features,
            skips,
            output_flow_strength=1.1,
        )
