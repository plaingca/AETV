from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aetv.models import AETVAutoencoder
from aetv.multigop_channel import MultiGOPChannelCurriculum
from aetv.optical_flow import sample_padded_reference
from aetv.recurrent_feature_tcm import V8RecurrentFeatureTCM


class ZeroFlowAligner(nn.Module):
    def estimate_flow(self, reference, target):
        return torch.zeros(
            reference.shape[0], 2, *reference.shape[-2:],
            device=reference.device, dtype=reference.dtype,
        )

    def warp_with_flow(self, reference, flow):
        height, width = flow.shape[-2:]
        return sample_padded_reference(
            reference, flow, left=0, top=0, output_height=height, output_width=width
        )


class ShiftAligner(ZeroFlowAligner):
    def estimate_flow(self, reference, target):
        flow = super().estimate_flow(reference, target)
        flow[:, 0] = 2.0
        return flow


def small_tcm(aligner=None) -> V8RecurrentFeatureTCM:
    base = AETVAutoencoder(mode="V8", width=32, latent_channels=3, compact=False)
    return V8RecurrentFeatureTCM(base, aligner=aligner or ZeroFlowAligner())


def test_zero_init_feature_tcm_matches_independent_decode():
    model = small_tcm().eval()
    video = torch.rand(1, 3, 12, 108, 192)
    with torch.inference_mode():
        latents = model.encode_gops(video)
        recurrent = model.decode_gops(latents)
        independent = torch.cat(
            [model.decoder(latents[:, index]) for index in range(2)], dim=2
        )
    assert latents.shape == (1, 2, 2816)
    assert torch.allclose(recurrent, independent, atol=1e-5, rtol=1e-5)


def test_encoder_stays_frozen_and_fuser_gets_gradients():
    model = small_tcm()
    counts = model.set_trainable_contract()
    assert counts["encoder"] == 0
    assert counts["encoder_context"] == 0
    assert counts["state"] > 0
    assert counts["decoder_tail"] > 0
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    video = torch.rand(1, 3, 12, 108, 192)
    reconstruction, events = model(video, MultiGOPChannelCurriculum(), progress=1.0)
    assert events["received"].shape[-1] == 2816
    reconstruction.mean().backward()
    assert model.encoder.encoder.net[0].conv.weight.grad is None
    assert model.fuser.output.weight.grad is not None
    assert torch.isfinite(model.fuser.output.weight.grad).all()


def test_scene_reset_is_exact_independent_decode():
    model = small_tcm().eval()
    model.photometric_threshold = 2.0
    torch.nn.init.normal_(model.fuser.output.weight, std=0.05)
    torch.nn.init.normal_(model.fuser.output.bias, std=0.05)
    video = torch.rand(1, 3, 12, 108, 192)
    reset = torch.tensor([[False, True]])
    with torch.inference_mode():
        latents = model.encode_gops(video)
        independent = model.decoder(latents[:, 1])
        reset_decoded = model.decode_gops(latents, reset=reset)
        live_decoded = model.decode_gops(latents)
    assert torch.allclose(reset_decoded[:, :, 6:], independent, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(live_decoded[:, :, 6:], independent, atol=1e-4)


def test_nonzero_flow_changes_fused_features():
    zero = small_tcm(ZeroFlowAligner()).eval()
    shifted = small_tcm(ShiftAligner()).eval()
    zero.photometric_threshold = 2.0
    shifted.photometric_threshold = 2.0
    shifted.load_state_dict(zero.state_dict(), strict=False)
    torch.nn.init.normal_(shifted.fuser.output.weight, std=0.05)
    torch.nn.init.normal_(shifted.fuser.output.bias, std=0.05)
    zero.fuser.load_state_dict(shifted.fuser.state_dict())
    video = torch.rand(1, 3, 12, 108, 192)
    with torch.inference_mode():
        latents = zero.encode_gops(video)
        unaligned = zero.decode_gops(latents)
        aligned = shifted.decode_gops(latents)
    assert not torch.allclose(aligned, unaligned, atol=1e-5)


def test_released_feature_tcm_keeps_wire_and_frozen_encoder():
    checkpoint = Path("models/v8-hf3k-face-gan.pt")
    if not checkpoint.exists():
        pytest.skip("released V8 checkpoint unavailable")
    model = V8RecurrentFeatureTCM.from_released(checkpoint, aligner=ZeroFlowAligner())
    counts = model.set_trainable_contract()
    assert model.latent_budget == 2816
    assert counts["encoder"] == 0
    assert counts["state"] > 0
    assert counts["decoder_tail"] > 0
    video = torch.rand(1, 3, 12, 108, 192)
    with torch.inference_mode():
        latents = model.encode_gops(video)
        independent = torch.stack(
            (model.encoder(video[:, :, :6]), model.encoder(video[:, :, 6:12])),
            dim=1,
        )
    assert latents.shape[-1] == 2816
    assert torch.allclose(latents, independent, atol=1e-6, rtol=1e-5)
