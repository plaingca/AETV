import torch

from aetv.decoder_context_adapter import V8DecoderContextAdapter
from aetv.models import AETVAutoencoder
from aetv.overlap_models import join_video_gops


def small_model() -> tuple[AETVAutoencoder, V8DecoderContextAdapter]:
    base = AETVAutoencoder(mode="V8", width=32, latent_channels=3, compact=False)
    model = V8DecoderContextAdapter(
        base,
        adapter_width=16,
        attention_dim=8,
        attention_heads=2,
        adapter_blocks=1,
        freeze_base=True,
    )
    return base, model


def stock_sequence(base: AETVAutoencoder, latents: torch.Tensor) -> torch.Tensor:
    batch, count, budget = latents.shape
    decoded = base.decoder(latents.reshape(batch * count, budget))
    return join_video_gops(decoded.reshape(batch, count, *decoded.shape[1:]))


def test_zero_initialized_internal_adapter_is_exact_stock_decoder():
    base, model = small_model()
    latents = torch.randn(1, 3, 2816)
    with torch.inference_mode():
        context = model.decode_sequence(latents)
        stock = stock_sequence(base, latents)
    assert torch.equal(context, stock)


def test_internal_adapter_is_only_trainable_component():
    _, model = small_model()
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("context_adapter.") for name in trainable)


def test_previous_gop_can_change_current_reconstruction_after_learning():
    _, model = small_model()
    torch.nn.init.normal_(model.context_adapter.output.weight, std=0.01)
    latents = torch.randn(1, 2, 2816)
    changed = latents.clone()
    changed[:, 0].add_(1.0)
    with torch.inference_mode():
        original = model.decode_sequence(latents)
        modified = model.decode_sequence(changed)
    # The current latent is identical, but its six reconstructed frames can use
    # the previous latent through the internal bottleneck adapter.
    assert not torch.equal(original[:, :, 6:], modified[:, :, 6:])


def test_scene_reset_is_exact_bypass_and_rate_is_fixed():
    base, model = small_model()
    torch.nn.init.normal_(model.context_adapter.output.weight, std=0.01)
    latents = torch.randn(1, 3, 2816)
    reset = torch.ones(1, 2, dtype=torch.bool)
    with torch.inference_mode():
        output = model.decode_sequence(latents, context_reset=reset)
        stock = stock_sequence(base, latents)
    assert torch.equal(output, stock)
    assert output.shape == (1, 3, 18, 108, 192)
    assert model.config()["latents_per_gop"] == 2816
    assert model.config()["lookahead_gops"] == 0
