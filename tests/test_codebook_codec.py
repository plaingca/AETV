"""The codebook codec fills the 2,816-real budget and round-trips a GOP."""

import torch

from aetv.codebook_codec import (
    CODES,
    GOP_VALUES,
    TOKENS,
    Codebook,
    CodebookCodec,
    apply_carrier_fade,
)


def test_a_token_on_its_code_is_selected():
    book = Codebook()
    tokens = book.book[:8].detach().unsqueeze(0)
    _chosen, weights = book.assign(tokens)
    assert float(weights.detach().max(dim=-1).values.min()) > 0.5


def test_layout_is_the_modem_budget():
    model = CodebookCodec()
    assert model.codebook.book.shape == (CODES, 32)
    assert TOKENS * 32 == GOP_VALUES


def test_fade_matches_the_wire():
    wire = torch.randn(2, GOP_VALUES)
    wire = wire / wire.square().mean(-1, keepdim=True).sqrt()
    received, confidence = apply_carrier_fade(wire, sigma=0.4)
    assert received.shape == wire.shape
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0


def test_encode_is_unit_rms_and_decode_has_video_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodebookCodec().to(device).eval()
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (1, GOP_VALUES)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    model.reset()
    recon, outage = model.decode_gop(wire, retain_state=False)
    assert recon.shape == video.shape
    assert outage.shape == (1, 4)
    assert float(recon.detach().min()) >= -1e-5
    assert float(recon.detach().max()) <= 1 + 1e-5
