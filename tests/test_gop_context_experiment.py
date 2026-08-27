import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_context import (  # noqa: E402
    StatefulGOPCorrector,
    apply_adapter_sequence,
)


def test_fresh_adapter_is_exact_noop_with_context():
    adapter = StatefulGOPCorrector(width=8, blocks=1)
    previous = torch.rand(2, 3, 6, 16, 16)
    current = torch.rand(2, 3, 6, 16, 16)
    assert torch.equal(adapter(current, previous), current)


def test_missing_context_is_exact_bypass_after_training_changes():
    adapter = StatefulGOPCorrector(width=8, blocks=1)
    with torch.no_grad():
        adapter.output.bias.fill_(1.0)
    current = torch.rand(2, 3, 6, 16, 16)
    assert torch.equal(adapter(current, None), current)


def test_zero_confidence_is_exact_bypass():
    adapter = StatefulGOPCorrector(width=8, blocks=1)
    with torch.no_grad():
        adapter.output.bias.fill_(1.0)
    previous = torch.rand(2, 3, 6, 16, 16)
    current = torch.rand(2, 3, 6, 16, 16)
    assert torch.equal(adapter(current, previous, confidence=0.0), current)


def test_sequence_reset_probability_one_bypasses_all_corrections():
    adapter = StatefulGOPCorrector(width=8, blocks=1)
    with torch.no_grad():
        adapter.output.bias.fill_(1.0)
    base = torch.rand(2, 3, 3, 6, 16, 16)
    output = apply_adapter_sequence(adapter, base, reset_probability=1.0)
    assert torch.equal(output, base)


def test_bad_previous_confidence_bypasses_next_correction():
    adapter = StatefulGOPCorrector(width=8, blocks=1)
    with torch.no_grad():
        adapter.output.bias.fill_(1.0)
    base = torch.rand(1, 3, 3, 6, 16, 16)
    confidence = torch.tensor([[1.0, 0.0, 1.0]])
    output = apply_adapter_sequence(adapter, base, confidences=confidence)
    assert torch.equal(output, base)


def test_full_context_adapter_uses_previous_gop_trajectory():
    adapter = StatefulGOPCorrector(width=8, blocks=1, context_mode="full")
    previous = torch.rand(2, 3, 6, 16, 16)
    current = torch.rand(2, 3, 6, 16, 16)
    assert torch.equal(adapter(current, previous), current)


def test_nonzero_taper_floor_can_correct_final_frame():
    adapter = StatefulGOPCorrector(width=8, blocks=1, taper_floor=1.0)
    with torch.no_grad():
        adapter.output.bias.fill_(1.0)
    previous = torch.rand(1, 3, 6, 16, 16)
    current = torch.full_like(previous, 0.5)
    output = adapter(current, previous)
    assert not torch.equal(output[:, :, -1], current[:, :, -1])
