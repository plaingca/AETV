"""PSNR fine-tune wrapper around the AC2K 2.2 kHz codec.

The wire budget stays 2,816 real coordinates. A zero-initialized pixel residual
can correct blur without changing the on-air latent layout, and the base codec
remains the loaded AC2K checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .ac2k import AC2KCodec


class ResidualTail(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return (video + self.net(video)).clamp(0, 1)


class AC2KPSNR(nn.Module):
    architecture = "ac2k-psnr-tail-v1"

    def __init__(self, base: AC2KCodec):
        super().__init__()
        self.base = base
        self.tail = ResidualTail()

    def reset(self) -> None:
        self.base.reset()

    def encode_gop(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base.encode_gop(video, **kwargs)

    def decode_gop(self, payload: torch.Tensor, confidence: torch.Tensor | None = None, **kwargs):
        recon, outage = self.base.decode_gop(payload, confidence, **kwargs)
        return self.tail(recon), outage


def load_psnr_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[AC2KPSNR, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("base_config") or {}
    base = AC2KCodec(
        width=config.get("width", 48),
        features=config.get("features", 20),
        motion_width=config.get("motion_width", 40),
        refine_blocks=config.get("refine_blocks", 2),
        num_offsets=config.get("num_offsets", 2),
        confidence_threshold=config.get("confidence_threshold", 0.02),
        diagonal_gdn=config.get("diagonal_gdn", True),
        iframe_mem=config.get("iframe_mem", True),
        anchor_gate_max=config.get("anchor_gate_max", 0.35),
        anchor_style_max=config.get("anchor_style_max", 0.39),
        deep_tail=bool(config.get("deep_tail", False)),
    )
    model = AC2KPSNR(base)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def load_ac2k(path: str | Path, device: torch.device | str = "cpu") -> AC2KCodec:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("model_config") or {}
    model = AC2KCodec(
        width=config.get("width", 48),
        features=config.get("features", 20),
        motion_width=config.get("motion_width", 40),
        refine_blocks=config.get("refine_blocks", 2),
        num_offsets=config.get("num_offsets", 2),
        confidence_threshold=config.get("confidence_threshold", 0.02),
        diagonal_gdn=config.get("diagonal_gdn", True),
        iframe_mem=config.get("iframe_mem", True),
        anchor_gate_max=config.get("anchor_gate_max", 0.35),
        anchor_style_max=config.get("anchor_style_max", 0.39),
        deep_tail=bool(config.get("deep_tail", False)),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model
