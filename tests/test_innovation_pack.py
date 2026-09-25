"""Spare-coordinate innovation keeps the 2,816-coordinate wire."""

import torch

from aetv.innovation_pack import GOP_VALUES, InnovationPack, embed_innovation
from aetv.narrowband_jscc import unit_rms as wire_rms


def test_embed_keeps_other_coordinates_and_unit_rms():
    torch.manual_seed(0)
    raw = wire_rms(torch.randn(2, GOP_VALUES))
    index = torch.arange(1280, 1280 + 384)
    code = torch.randn(2, 384, requires_grad=True)
    wire = embed_innovation(raw, code, index)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    keep = torch.ones(GOP_VALUES, dtype=torch.bool)
    keep[index] = False
    assert torch.allclose(wire[:, keep], raw[:, keep], atol=1e-5)
    recovered = wire_rms(wire[:, index])
    assert torch.allclose(recovered, wire_rms(code), atol=1e-4)
    recovered.square().mean().backward()
    assert code.grad is not None and code.grad.abs().sum() > 0


def test_untrained_correction_is_zero_and_wire_is_exact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InnovationPack().to(device).train()
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (1, GOP_VALUES)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    model.reset()
    recon, outage = model.decode_gop(wire, retain_state=False)
    assert recon.shape == video.shape
    assert outage.shape == (1, 4)
    damaged = wire.clone()
    damaged[:, model.index] = 0
    base, _ = model.base.decode_gop(damaged, retain_state=False)
    assert torch.allclose(recon, base.clamp(0, 1), atol=1e-5)
    loss = (recon - video).square().mean()
    loss.backward()
    # The correction starts at zero, so only that layer has a gradient on the first step.
    assert model.pack.correct.weight.grad is not None
    assert model.pack.correct.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())
