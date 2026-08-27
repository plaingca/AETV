import pytest
import torch

from aetv.ltx_channel import (
    LTX_LATENT_SHAPE,
    V8_CHANNEL_VALUES,
    LTXV8ChannelAdapter,
    finish_ltx_video,
    prepare_ltx_video,
)


def test_ltx_v8_video_padding_and_crop():
    video = torch.rand(2, 3, 6, 108, 192)
    padded = prepare_ltx_video(video)
    assert padded.shape == (2, 3, 9, 108, 192)
    assert torch.allclose(padded[:, :, -1], video[:, :, -1] * 2 - 1)
    decoded = torch.rand(2, 3, 9, 128, 192) * 2 - 1
    assert finish_ltx_video(decoded).shape == video.shape


def test_ltx_adapter_exact_budget_and_gradients():
    adapter = LTXV8ChannelAdapter()
    latent = torch.randn(2, *LTX_LATENT_SHAPE, requires_grad=True)
    symbols = adapter.encode(latent)
    assert symbols.shape == (2, V8_CHANNEL_VALUES)
    assert torch.allclose(symbols.square().mean(1), torch.ones(2), atol=2e-5)
    restored = adapter.decode(symbols, torch.ones_like(symbols))
    assert restored.shape == latent.shape
    restored.square().mean().backward()
    assert latent.grad is not None


def test_ltx_adapter_rejects_wrong_latent_shape():
    with pytest.raises(ValueError):
        LTXV8ChannelAdapter().encode(torch.randn(1, 128, 1, 4, 6))
