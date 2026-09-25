"""RVDJSCC narrowband port keeps the existing coordinate budgets."""

import torch

from aetv.config import BAND_A, LATENTS_PER_GOP_U, LATENTS_PER_GOP_W
from aetv.rvdjscc_narrowband import (
    GRID,
    RVDJSCCNarrowband,
    grid_channels,
    packet_split,
    read_checkpoint,
    save_fp16_shards,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_packet_split_matches_paper_ratio_on_existing_budgets():
    assert packet_split(LATENTS_PER_GOP_W) == (1412, 468)
    assert packet_split(LATENTS_PER_GOP_U) == (5060, 1684)
    assert packet_split(BAND_A.latents_per_gop) == (9600, 3200)
    for budget in (LATENTS_PER_GOP_W, LATENTS_PER_GOP_U, BAND_A.latents_per_gop):
        key, interp = packet_split(budget)
        assert key + 3 * interp == budget
        assert key == 3 * interp or key == 3 * interp + (budget % 12)
        assert grid_channels(key) * GRID >= key
        assert grid_channels(interp) * GRID >= interp


def test_fp16_shards_roundtrip(tmp_path):
    payload = {
        "architecture": "rvdjscc-narrowband-v2",
        "model_state_dict": {
            "a": torch.arange(100, dtype=torch.float32),
            "b": torch.arange(100, dtype=torch.float32) + 0.5,
        },
        "step": 3,
    }
    directory = save_fp16_shards(payload, tmp_path / "ckpt", limit_bytes=300)
    loaded = read_checkpoint(directory)
    assert loaded["step"] == 3
    assert set(loaded["model_state_dict"]) == {"a", "b"}
    assert loaded["model_state_dict"]["a"].dtype == torch.float16
    assert len(list(directory.glob("weights-*.pt"))) >= 2


def test_forward_uses_exact_2_2_khz_budget_and_identity_denoiser():
    torch.manual_seed(0)
    device = _device()
    model = RVDJSCCNarrowband(2.2, width=64, context_channels=8).to(device)
    assert model.architecture == "rvdjscc-narrowband-v2"
    assert model.key_codec.latent_channels == 5
    assert model.interp_codec.latent_channels == 2
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    clean = torch.rand(1, 64, device=device)
    assert torch.allclose(model.denoiser(clean, torch.full((1, 1), 15.0, device=device)), clean, atol=1e-6)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (1, LATENTS_PER_GOP_W)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    recon, outage = model.decode_gop(wire, retain_state=False)
    assert recon.shape == video.shape
    assert outage.shape == (1, 4)
    assert recon.min() >= 0 and recon.max() <= 1
    snr = torch.full((1, 1), 15.0, device=device)
    loss = (recon - video).square().mean() + model.denoising_loss(wire, wire, snr)
    loss.backward()
    latent_grad = model.key_codec.encoder.net[12].conv.weight.grad
    assert latent_grad is not None and latent_grad.abs().sum() > 0
    assert model.ssf.down[0].weight.grad is not None
    assert model.denoiser.exit.weight.grad is not None


def test_state_carries_the_previous_key_across_gops():
    torch.manual_seed(1)
    device = _device()
    model = RVDJSCCNarrowband(2.2, width=64, context_channels=8).to(device)
    video = torch.rand(1, 3, 4, 108, 192, device=device)
    assert model.prev_key is None
    model.encode_gop(video, retain_state=True)
    assert model.prev_key is not None
    assert model.prev_key.shape == (1, 3, 108, 192)
    model.reset()
    assert model.prev_key is None
