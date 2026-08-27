import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_reliability_gate import (  # noqa: E402
    GATE_INPUT_CHANNELS,
    ReliabilityGate,
    boundary_safety_mask,
    event_cells,
    join_many_gops,
    multi_boundary_losses,
    split_many_gops,
)


def gate_inputs(batch=2, frames=6, height=8, width=12):
    return torch.rand(batch, GATE_INPUT_CHANNELS, frames, height, width)


def test_many_gop_split_join_round_trip():
    sequence = torch.arange(2 * 3 * 30 * 4 * 5).reshape(2, 3, 30, 4, 5)
    separated = split_many_gops(sequence, 6)
    grouped = separated.reshape(2, 5, 3, 6, 4, 5)
    assert torch.equal(join_many_gops(grouped), sequence)


def test_scalar_gate_uses_boundary_confidence_and_hard_validity():
    gate = ReliabilityGate("scalar", feature_channels=4)
    current = torch.rand(2, 4, 6, 8, 12)
    previous = torch.rand_like(current)
    feature, spatial = gate(
        gate_inputs(), current, previous, torch.tensor([0.25, 0.75]), torch.tensor([True, False])
    )
    assert torch.allclose(feature[0], torch.full_like(feature[0], 0.25))
    assert torch.count_nonzero(feature[1]) == 0
    assert spatial.shape == (2, 1, 6, 8, 12)


@pytest.mark.parametrize("mode", ("spatial", "spatial_channel"))
def test_learned_gate_starts_at_scalar_control(mode):
    gate = ReliabilityGate(mode, feature_channels=4)
    current = torch.rand(2, 4, 6, 8, 12)
    previous = torch.rand_like(current)
    confidence = torch.tensor([0.25, 0.75])
    feature, spatial = gate(
        gate_inputs(), current, previous, confidence, torch.tensor([True, True])
    )
    assert torch.allclose(spatial[0], torch.full_like(spatial[0], 0.25), atol=1e-6)
    assert torch.allclose(spatial[1], torch.full_like(spatial[1], 0.75), atol=1e-6)
    assert torch.allclose(feature[:, 0], spatial[:, 0], atol=1e-6)


@pytest.mark.parametrize("mode", ("scalar", "spatial", "spatial_channel"))
def test_zero_confidence_is_exact_gate_bypass(mode):
    gate = ReliabilityGate(mode, feature_channels=4)
    current = torch.rand(1, 4, 6, 8, 12)
    feature, spatial = gate(
        gate_inputs(batch=1), current, torch.rand_like(current), torch.zeros(1), torch.ones(1, dtype=torch.bool)
    )
    assert torch.count_nonzero(feature) == 0
    assert torch.count_nonzero(spatial) == 0


def test_multi_boundary_loss_zero_for_exact_five_gop_reconstruction():
    target = torch.rand(2, 3, 30, 8, 8)
    losses = multi_boundary_losses(target.clone(), target, 6)
    errors = {name: value for name, value in losses.items() if not name.endswith("_ratio")}
    assert all(value.item() == 0 for value in errors.values())


def test_curriculum_includes_good_fade_good_and_rejects_short_runs():
    assert event_cells("good_fade_good", 5) == [
        "clean",
        "clean",
        "measured_hf",
        "mpp_12db",
        "clean",
    ]
    with pytest.raises(ValueError, match="at least five GOPs"):
        event_cells("steady_clean", 4)


def test_boundary_safety_hard_bypasses_cut_bad_previous_path_and_reset():
    safe = boundary_safety_mask(
        torch.tensor([True, True, True, True]),
        torch.tensor([True, True, True, True]),
        torch.tensor([False, False, False, True]),
        torch.tensor([0.05, 0.20, 0.05, 0.05]),
        torch.tensor([10.0, 10.0, -5.0, 10.0]),
        torch.tensor([0.9, 0.9, 0.1, 0.9]),
        scene_cut_threshold=0.15,
        min_previous_snr_db=-2.0,
        min_previous_pilot_coherence=0.25,
    )
    assert torch.equal(safe, torch.tensor([True, False, False, False]))
