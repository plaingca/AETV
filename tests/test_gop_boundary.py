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


def test_causal_refiner_ignores_future_frames():
    torch.manual_seed(0)
    model = BoundaryRefiner(width=16, blocks=1, causal=True)
    torch.nn.init.normal_(model.tail.weight, std=0.01)
    video, conf = torch.rand(1, 3, 12, 16, 24), torch.rand(1, 12)
    changed = video.clone()
    changed[:, :, 8:] = 0
    assert torch.equal(model(video, conf)[:, :, :8], model(changed, conf)[:, :, :8])
    assert not torch.equal(model(video, conf)[:, :, 8:], model(changed, conf)[:, :, 8:])
