"""Wire budgets stay on the existing narrowband channels."""

import torch

from aetv.config import BAND_A, LATENTS_PER_GOP_U, LATENTS_PER_GOP_W
from aetv.narrowband_jscc import BANDWIDTHS, NarrowbandJSCC, impair_wire, resolve_bandwidth


def test_budgets_match_existing_modem_gops():
    assert resolve_bandwidth(2.2) == (2.2, LATENTS_PER_GOP_W, "V8")
    assert resolve_bandwidth(8) == (8.0, LATENTS_PER_GOP_U, "V7")
    assert resolve_bandwidth(16) == (16.0, BAND_A.latents_per_gop, "AC16")
    assert [item[1] for item in BANDWIDTHS] == [
        LATENTS_PER_GOP_W,
        LATENTS_PER_GOP_U,
        BAND_A.latents_per_gop,
    ]


def test_forward_uses_exact_budget_and_reconstructs_shape():
    torch.manual_seed(0)
    model = NarrowbandJSCC(2.2, width=32)
    video = torch.rand(2, 3, 6, 108, 192)
    wire = model.encode(video)
    assert wire.shape == (2, LATENTS_PER_GOP_W)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    noisy, confidence = impair_wire(wire, 15.0)
    recon = model.decode(noisy, confidence)
    assert recon.shape == video.shape
    assert recon.min() >= 0 and recon.max() <= 1
    loss = (recon - video).square().mean()
    loss.backward()
    assert model.to_wire.local.blocks[0].weight.grad is not None


def test_wider_budgets_stay_on_their_channels():
    for khz, budget, _mode in BANDWIDTHS:
        model = NarrowbandJSCC(khz, width=48)
        video = torch.rand(1, 3, 6, 108, 192)
        wire = model.encode(video)
        assert wire.shape[-1] == budget
        recon = model(video, snr_db=18.0)
        assert recon.shape == video.shape
