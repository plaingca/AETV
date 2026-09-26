"""GOP-boundary metrics and the receiver-side refiner."""

import torch

from aetv.gop_boundary import BoundaryRefiner, boundary_record, transition_error


def test_boundary_jump_is_detected():
    source = torch.full((3, 12, 8, 8), 0.5)
    recon = source.clone()
    recon[:, 6:] += 0.1
    record = boundary_record(source, recon, gop=6)
    errors = transition_error(source, recon)
    assert errors[5] > 0 and sum(errors) == errors[5]
    assert record["boundary_tpsnr"] < 25 and record["interior_tpsnr"] == 100.0
    assert abs(record["boundary_jump"] - 25.5) < 1e-3


def test_untrained_refiner_is_identity():
    video = torch.rand(1, 3, 12, 16, 24)
    out = BoundaryRefiner(width=16, blocks=1)(video, torch.rand(1, 12))
    assert torch.equal(out, video)


def test_per_gop_refiner_has_no_cross_gop_view():
    torch.manual_seed(0)
    model = BoundaryRefiner(width=16, blocks=1, per_gop=True)
    torch.nn.init.normal_(model.tail.weight, std=0.01)
    video, conf = torch.rand(2, 3, 12, 16, 24), torch.rand(2, 12)
    changed = video.clone()
    changed[:, :, 6:] = 0
    assert torch.equal(model(video, conf)[:, :, :6], model(changed, conf)[:, :, :6])


def test_healthy_gate_scales_only_healthy_gops():
    torch.manual_seed(0)
    video, conf = torch.full((1, 3, 12, 16, 24), 0.5), torch.tensor([[0.1] * 6 + [0.7] * 6])
    full = BoundaryRefiner(width=16, blocks=1)
    torch.nn.init.normal_(full.tail.weight, std=0.01)
    gated = BoundaryRefiner(width=16, blocks=1, healthy_gate=0.0)
    gated.load_state_dict(full.state_dict())
    out_full, out_gated = full(video, conf), gated(video, conf)
    assert torch.equal(out_gated[:, :, :6], out_full[:, :, :6])
    assert torch.equal(out_gated[:, :, 6:], video[:, :, 6:])
    learned = BoundaryRefiner(width=16, blocks=1, healthy_gate="learned", gate_init=0.0)
    assert torch.allclose(learned.gate(conf, 12), torch.tensor([[1.0] * 6 + [0.5] * 6]))


def test_causal_refiner_ignores_future_frames():
    torch.manual_seed(0)
    model = BoundaryRefiner(width=16, blocks=1, causal=True)
    torch.nn.init.normal_(model.tail.weight, std=0.01)
    video, conf = torch.rand(1, 3, 12, 16, 24), torch.rand(1, 12)
    changed = video.clone()
    changed[:, :, 8:] = 0
    assert torch.equal(model(video, conf)[:, :, :8], model(changed, conf)[:, :, :8])
    assert not torch.equal(model(video, conf)[:, :, 8:], model(changed, conf)[:, :, 8:])
