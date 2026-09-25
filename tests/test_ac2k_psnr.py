"""The PSNR tail does not change the 2,816-coordinate AC2K wire."""

import torch

from aetv.ac2k import AC2KCodec
from aetv.ac2k_psnr import AC2KPSNR, ResidualTail
from aetv.config import LATENTS_PER_GOP_W


def test_residual_tail_is_identity_at_init():
    tail = ResidualTail()
    video = torch.rand(1, 3, 4, 108, 192)
    assert torch.allclose(tail(video), video)


def test_psnr_wrapper_keeps_ac2k_budget():
    torch.manual_seed(0)
    model = AC2KPSNR(AC2KCodec(width=16, features=8, motion_width=16, refine_blocks=1))
    video = torch.rand(1, 3, 4, 108, 192)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (1, LATENTS_PER_GOP_W)
    recon, outage = model.decode_gop(wire, retain_state=False)
    assert recon.shape == video.shape
    assert outage.shape[0] == 1
    base_recon, _ = model.base.decode_gop(wire, retain_state=False)
    assert torch.allclose(recon, base_recon)
