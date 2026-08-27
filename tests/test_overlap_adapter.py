import torch

from aetv.models import AETVAutoencoder
from aetv.overlap_adapter import V8OverlapAdapter


def small_model() -> tuple[AETVAutoencoder, V8OverlapAdapter]:
    base = AETVAutoencoder(mode="V8", width=32, latent_channels=3, compact=False)
    model = V8OverlapAdapter(
        base, window_gops=5, emit_gops=3, adapter_width=8, freeze_base=True
    )
    return base, model


def test_zero_initialized_adapter_is_exact_released_function():
    base, model = small_model()
    latents = torch.rand(1, 5, 2816)
    weights = torch.ones_like(latents)
    with torch.inference_mode():
        overlap = model.decode_sequence(latents, weights)
        direct = torch.cat(
            [base.decoder(latents[:, index], weights[:, index]) for index in range(1, 4)],
            dim=2,
        )
    # Batched and one-at-a-time convolution kernels can differ by a few ULPs.
    assert torch.allclose(overlap, direct, atol=1e-6, rtol=0)


def test_adapter_is_only_trainable_component():
    _, model = small_model()
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("adapter.") for name in trainable)


def test_context_can_influence_center_after_adapter_learns():
    _, model = small_model()
    torch.nn.init.normal_(model.adapter.output.weight, std=0.01)
    latents = torch.rand(1, 5, 2816)
    changed = latents.clone()
    changed[:, 0] += 1
    with torch.inference_mode():
        first = model.decode_sequence(latents)
        second = model.decode_sequence(changed)
    assert not torch.equal(first, second)


def test_two_windows_emit_six_gops_at_fixed_rate():
    _, model = small_model()
    latents = torch.rand(1, 8, 2816)
    with torch.inference_mode():
        output = model.decode_sequence(latents)
    assert output.shape == (1, 3, 36, 108, 192)
    assert model.config()["latents_per_gop"] == 2816
