from pathlib import Path
import sys

import pytest
import torch

from aetv.decoder_context_adapter import V8DecoderContextAdapter, warp_nchw
from aetv.models import AETVAutoencoder
from aetv.multigop_channel import MultiGOPChannelCurriculum
from aetv.recurrent_joint_codec import V8RecurrentJointCodec
from aetv.transition_anchor_codec import V8TransitionAnchorCodec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_v8_recurrent_joint import CHANNEL_WEIGHTS, QUALITY_WEIGHTS  # noqa: E402


def small_joint() -> V8RecurrentJointCodec:
    base = AETVAutoencoder(mode="V8", width=32, latent_channels=3, compact=False)
    adapter = V8DecoderContextAdapter(
        base,
        adapter_width=16,
        attention_dim=8,
        attention_heads=2,
        adapter_blocks=1,
        freeze_base=False,
        temporal_taper=(1.0, 0.5, 0.0),
        preserve_highpass=True,
    )
    return V8RecurrentJointCodec(adapter)


def test_multigop_channel_keeps_wire_shape_and_encoder_gradient():
    transmitted = torch.randn(2, 8, 2816, requires_grad=True)
    received, weights, events = MultiGOPChannelCurriculum()(transmitted, progress=1.0)
    assert received.shape == weights.shape == transmitted.shape
    assert events["missing"].shape == (2, 8)
    assert events["fade_end"].shape == (2, 8)
    received.sum().backward()
    assert transmitted.grad is not None
    assert torch.isfinite(transmitted.grad).all()


def test_multigop_channel_can_emit_good_fade_good():
    torch.manual_seed(0)
    transmitted = torch.randn(4, 8, 2816)
    _, _, events = MultiGOPChannelCurriculum(fade_probability=1.0)(transmitted, progress=1.0)
    fade = events["fade"]
    assert fade.any()
    assert not fade[:, 0].any()
    expected_end = torch.zeros_like(fade)
    expected_end[:, 1:] = fade[:, :-1] & ~fade[:, 1:]
    assert torch.equal(events["fade_end"], expected_end)
    assert events["fade_end"].any()


def test_released_joint_model_has_all_three_trainable_groups():
    checkpoint = Path("models/v8-hf3k-face-gan.pt")
    if not checkpoint.exists():
        pytest.skip("released V8 checkpoint unavailable")
    model = V8RecurrentJointCodec.from_released(checkpoint)
    counts = model.set_trainable_contract()
    assert model.latent_budget == 2816
    assert counts["encoder"] > 0
    assert counts["encoder_context"] > 0
    assert counts["state"] > 0
    assert counts["decoder_tail"] > 0
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder_context.parameters())


def test_unroll_contract_rejects_non_v8_wire_shape():
    channel = MultiGOPChannelCurriculum()
    with pytest.raises(ValueError, match="B,G,2816"):
        channel(torch.randn(1, 5, 2815), progress=0.5)


def test_transition_anchor_is_fixed_rate_and_trainable():
    checkpoint = Path("models/v8-hf3k-face-gan.pt")
    if not checkpoint.exists():
        pytest.skip("released V8 checkpoint unavailable")
    model = V8TransitionAnchorCodec.from_released(checkpoint, anchor_values=64)
    counts = model.set_trainable_contract()
    assert model.anchor_values == 64
    assert model.latent_budget == 2816
    assert counts["transition_anchor"] > 0


def test_loss_does_not_optimize_raw_boundary_rgb():
    assert "boundary_rgb" not in QUALITY_WEIGHTS
    assert "boundary_rgb" not in CHANNEL_WEIGHTS
    required = {
        "spatial_l1",
        "spatial_lpips",
        "detail",
        "spatial_highpass",
        "boundary_delta",
        "boundary_lowpass_y",
        "boundary_acceleration",
        "boundary_delta_lpips",
    }
    assert required <= set(QUALITY_WEIGHTS)
    assert required <= set(CHANNEL_WEIGHTS)


def test_zero_flow_warp_is_identity():
    values = torch.rand(2, 4, 13, 24)
    flow = torch.zeros(2, 2, 13, 24)
    warped = warp_nchw(values, flow)
    assert torch.allclose(warped, values, atol=1e-5, rtol=1e-5)


def test_nonzero_bottleneck_flow_changes_aligned_context():
    model = small_joint()
    torch.nn.init.normal_(model.context_adapter.flow.weight, std=0.05)
    torch.nn.init.normal_(model.context_adapter.flow.bias, std=0.05)
    torch.nn.init.normal_(model.context_adapter.output.weight, std=0.01)
    current = torch.randn(1, model.context_adapter.feature_channels, 3, 13, 24)
    previous = torch.randn_like(current)
    confidence = torch.ones(1)
    aligned, _ = model.context_adapter(current, previous, confidence)
    with torch.no_grad():
        model.context_adapter.flow.weight.zero_()
        model.context_adapter.flow.bias.zero_()
    unaligned, _ = model.context_adapter(current, previous, confidence)
    assert not torch.allclose(aligned, unaligned, atol=1e-5)


def test_zero_init_encoder_context_keeps_independent_current_gop_latents():
    model = small_joint().eval()
    video = torch.rand(1, 3, 12, 108, 192)
    with torch.inference_mode():
        joint = model.encode_gops(video)
        independent = torch.stack(
            (model.encoder(video[:, :, :6]), model.encoder(video[:, :, 6:12])),
            dim=1,
        )
    assert joint.shape == (1, 2, 2816)
    assert torch.allclose(joint, independent, atol=1e-6, rtol=1e-5)


def test_scene_reset_disables_encoder_context():
    model = small_joint().eval()
    torch.nn.init.normal_(model.encoder_context.project.weight, std=0.05)
    torch.nn.init.normal_(model.encoder_context.project.bias, std=0.05)
    video = torch.rand(1, 3, 12, 108, 192)
    reset = torch.tensor([[False, True]])
    with torch.inference_mode():
        independent = model.encoder(video[:, :, 6:12])
        reset_encoded = model.encode_gops(video, reset=reset)
        live_encoded = model.encode_gops(video)
    assert torch.allclose(reset_encoded[:, 1], independent, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(live_encoded[:, 1], independent, atol=1e-5)


def test_encoder_and_context_receive_unroll_gradients():
    model = small_joint()
    counts = model.set_trainable_contract()
    assert counts["encoder"] > 0 and counts["encoder_context"] > 0
    video = torch.rand(1, 3, 12, 108, 192)
    channel = MultiGOPChannelCurriculum()
    reconstruction, events = model(video, channel, progress=1.0)
    assert events["received"].shape[-1] == 2816
    reconstruction.mean().backward()
    stem_weight = model.encoder.encoder.net[0].conv.weight
    assert stem_weight.grad is not None
    assert torch.isfinite(stem_weight.grad).all()
    assert model.encoder_context.project.weight.grad is not None
    assert torch.isfinite(model.encoder_context.project.weight.grad).all()
    assert model.encoder_context.flow.weight.grad is not None
    assert model.context_adapter.output.weight.grad is not None
    assert model.context_adapter.flow.weight.grad is not None
