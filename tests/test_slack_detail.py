"""Quantization-slack detail stays inside the 2,816-coordinate wire."""

import torch

from aetv.slack_detail import (
    CODE,
    GOP_VALUES,
    LIMIT,
    SlackDetail,
    quantize_wire,
)


def test_clean_roundtrip_recovers_the_detail_and_stays_in_bin():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis = torch.zeros(48, 2)
    basis[0, 0] = 1
    basis[1, 1] = 1
    model = SlackDetail(basis=basis, patch_mean=torch.zeros(1, 48)).to(device).eval()
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (1, GOP_VALUES)
    quantized = quantize_wire(wire)
    dither = wire - quantized
    assert float(dither.abs().max()) <= LIMIT + 1e-5
    model.reset()
    recon, outage = model.decode_gop(wire, retain_state=False)
    assert recon.shape == video.shape
    assert outage.shape == (1, 4)
    assert float(recon.min()) >= 0 and float(recon.max()) <= 1
    # The dither occupies the code slots and one gain slot; other slots match the bin center.
    untouched = torch.ones(GOP_VALUES, dtype=torch.bool)
    untouched[:CODE] = False
    untouched[CODE] = False
    assert torch.allclose(wire[:, untouched], quantized[:, untouched], atol=1e-6)


def test_noisy_confidence_ignores_the_dither():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlackDetail().to(device).eval()
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    wire = model.encode_gop(video, retain_state=False)
    confidence = torch.full_like(wire, 0.97)
    model.reset()
    recon, _ = model.decode_gop(wire, confidence=confidence, retain_state=False)
    model.reset()
    base, _ = model.base.decode_gop(wire, confidence=confidence, retain_state=False)
    assert torch.allclose(recon, base, atol=1e-5)
