"""Carrier fill keeps the 2,816-real AC2K wire and repairs faded tones."""

import torch

from aetv.multipath_common import (
    COORDS_PER_CARRIER,
    GOP_VALUES,
    LATENT_CARRIERS,
    CarrierFill,
    MultipathFill,
    apply_carrier_fade,
    carrier_index,
    carrier_layout,
)


def test_each_latent_carrier_owns_64_coordinates():
    layout = carrier_layout()
    assert layout.shape == (LATENT_CARRIERS, COORDS_PER_CARRIER)
    flat = layout.reshape(-1).sort().values
    assert torch.equal(flat, torch.arange(GOP_VALUES))
    carriers = carrier_index()
    assert int(carriers.min()) == 0 and int(carriers.max()) == LATENT_CARRIERS - 1


def test_fade_keeps_shape_and_confidence_in_unit_interval():
    wire = torch.randn(2, GOP_VALUES)
    wire = wire / wire.square().mean(-1, keepdim=True).sqrt()
    received, confidence = apply_carrier_fade(wire, sigma=0.4)
    assert received.shape == wire.shape
    assert confidence.shape == wire.shape
    assert float(confidence.min()) >= 0
    assert float(confidence.max()) <= 1


def test_untrained_fill_matches_ac2k_on_a_clean_wire():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultipathFill().to(device).eval()
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (1, GOP_VALUES)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    model.reset()
    recon, outage = model.decode_gop(wire, retain_state=False)
    model.reset()
    baseline, _outage = model.base.decode_gop(wire, retain_state=False)
    assert recon.shape == video.shape
    assert outage.shape == (1, 4)
    assert torch.allclose(recon, baseline, atol=1e-5)
    assert float(recon.detach().min()) >= -1e-5 and float(recon.detach().max()) <= 1 + 1e-5


def test_fill_repairs_a_zeroed_carrier_without_touching_confident_ones():
    torch.manual_seed(0)
    fill = CarrierFill(hidden=32, layers=1, heads=4, dropout=0.0)
    # Open the last layer so the gate is observable, then force a constant shift.
    torch.nn.init.constant_(fill.output.bias, 0.5)
    wire = torch.randn(2, GOP_VALUES)
    confidence = torch.ones(2, GOP_VALUES)
    confidence[:, fill.layout[0]] = 0
    restored = fill(wire, confidence)
    confident = fill.layout[1:].reshape(-1)
    erased = fill.layout[0]
    assert torch.allclose(restored[:, confident], wire[:, confident], atol=1e-5)
    erased_in = wire[:, erased].clamp(-6, 6)
    # The skip path is the raw coordinate; the bias rides on the low-confidence gate.
    assert torch.allclose(restored[:, erased], wire[:, erased] + 0.5, atol=1e-5)
    assert torch.allclose(erased_in, wire[:, erased].clamp(-6, 6))
