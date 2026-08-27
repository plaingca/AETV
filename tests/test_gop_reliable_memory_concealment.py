import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_gop_reliable_memory_concealment import (  # noqa: E402
    ConcealmentConfig,
    apply_bounded_memory_concealment,
    classify_reliable_gops,
    hold_concealment,
    triggered_quality_reasons,
)


class ShiftAligner:
    def estimate_flow(self, reference, target):
        return torch.ones(reference.shape[0], 2, *reference.shape[-2:])

    def warp_with_flow(self, reference, flow):
        return reference + flow[:, :1] * 0.01


def test_false_high_confidence_is_vetoed_by_bad_pilot_path():
    weights = torch.full((1, 2, 20), 0.95)
    snr = torch.tensor([[10.0, -8.0]])
    coherence = torch.tensor([[0.9, 0.1]])
    reliable, _ = classify_reliable_gops(weights, snr, coherence, ConcealmentConfig())
    assert torch.equal(reliable, torch.tensor([[True, False]]))


def test_false_low_confidence_is_overridden_by_strong_pilot_path():
    weights = torch.full((1, 2, 20), 0.05)
    snr = torch.tensor([[40.0, 20.0]])
    coherence = torch.tensor([[0.99, 0.90]])
    reliable, _ = classify_reliable_gops(weights, snr, coherence, ConcealmentConfig())
    assert torch.equal(reliable, torch.tensor([[True, True]]))


def test_hold_concealment_repeats_last_reliable_frame():
    gop = torch.rand(2, 3, 6, 4, 5)
    held = hold_concealment(gop)
    assert torch.equal(held[:, :, 0], gop[:, :, -1])
    assert torch.equal(held[:, :, -1], gop[:, :, -1])


def test_bounded_memory_does_not_update_on_erasure_and_expires():
    gops = torch.stack(
        [torch.full((3, 6, 4, 5), float(index)) for index in range(4)], dim=0
    ).unsqueeze(0)
    reliable = torch.tensor([[True, False, False, False]])
    output, events = apply_bounded_memory_concealment(
        gops,
        reliable,
        ShiftAligner(),
        ConcealmentConfig(max_memory_gops=2),
        mode="hold",
    )
    assert torch.equal(output[0, 1], torch.zeros_like(output[0, 1]))
    assert torch.equal(output[0, 2], torch.zeros_like(output[0, 2]))
    assert torch.equal(output[0, 3], gops[0, 3])
    assert [row["action"] for row in events[0]] == [
        "reliable_update",
        "conceal_hold",
        "conceal_hold",
        "memory_expired_bypass",
    ]


def test_flow_concealment_path_preserves_shape_and_changes_erased_gop():
    gops = torch.zeros(1, 2, 3, 6, 4, 5)
    reliable = torch.tensor([[True, False]])
    output, events = apply_bounded_memory_concealment(
        gops,
        reliable,
        ShiftAligner(),
        ConcealmentConfig(),
        mode="flow",
    )
    assert output.shape == gops.shape
    assert torch.count_nonzero(output[:, 1]) > 0
    assert events[0][1]["action"] == "conceal_flow"


def test_triggered_sequence_quality_guard_detects_motion_blur():
    reference = {
        "sequences": {
            "mpp": [
                {"within_motion_ratio": 0.50, "spatial_detail_ratio": 0.60, "lpips": 0.30}
            ]
        }
    }
    candidate = {
        "sequences": {
            "mpp": [
                {"within_motion_ratio": 0.20, "spatial_detail_ratio": 0.40, "lpips": 0.29}
            ]
        }
    }
    detection = {
        "mpp": [
            {
                "sequence": 0,
                "actions": [{"action": "reliable_update"}, {"action": "conceal_flow"}],
            }
        ]
    }
    reasons = triggered_quality_reasons(
        "memory_flow", candidate, reference, detection
    )
    assert any("motion ratio fell 0.500->0.200" in reason for reason in reasons)
    assert any("spatial detail ratio fell 0.600->0.400" in reason for reason in reasons)
