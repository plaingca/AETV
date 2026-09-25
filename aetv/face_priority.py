"""Face-priority input for the V8 encoder.

The transmitter runs YuNet on its own source frames and feeds the encoder a
fourth input plane: 1 inside the (slightly expanded) face boxes, 0 elsewhere.
Nothing extra is sent: the wire is still the 2,816-real V8 GOP, so the receiver
needs no mask. The extra stem input starts at zero weight, so a converted
checkpoint encodes exactly as before until it is trained.
"""

from __future__ import annotations

import torch
from torch import nn

EXPAND = 1.2


class MaskStem(nn.Module):
    """The pretrained RGB stem plus a zero-initialized convolution of the face-mask plane."""

    def __init__(self, rgb: nn.Conv3d):
        super().__init__()
        self.rgb = rgb
        self.mask = nn.Conv3d(1, rgb.out_channels, rgb.kernel_size, stride=rgb.stride, padding=rgb.padding,
                              bias=False).to(rgb.weight.device, rgb.weight.dtype)
        nn.init.zeros_(self.mask.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rgb(x[:, :3]) + self.mask(x[:, 3:])


def add_mask_input(autoencoder) -> None:
    """Give the encoder stem an extra face-mask input plane (its weights start at zero)."""
    stem = autoencoder.encoder.encoder.net[0]
    if isinstance(stem.conv, MaskStem):
        return
    stem.conv = MaskStem(stem.conv)


def mask_parameters(autoencoder) -> list[nn.Parameter]:
    return list(autoencoder.encoder.encoder.net[0].conv.mask.parameters())


def boxes_to_mask(boxes: torch.Tensor, height: int, width: int, expand: float = EXPAND) -> torch.Tensor:
    """(B, T, 4) xywh boxes (NaN = no face) -> (B, 1, T, H, W) float mask of the expanded boxes."""
    x, y, w, h = boxes.unbind(-1)
    cx, cy = x + 0.5 * w, y + 0.5 * h
    w, h = w * expand, h * expand
    ys = torch.arange(height, device=boxes.device).view(1, 1, height, 1) + 0.5
    xs = torch.arange(width, device=boxes.device).view(1, 1, 1, width) + 0.5
    x0, x1 = (cx - 0.5 * w)[..., None, None], (cx + 0.5 * w)[..., None, None]
    y0, y1 = (cy - 0.5 * h)[..., None, None], (cy + 0.5 * h)[..., None, None]
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return inside.float().unsqueeze(1)


def encode_with_mask(autoencoder, video: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """``video`` (B, 3, T, H, W) in [0, 1]; ``boxes`` (B, T, 4) from the transmitter's detector."""
    mask = boxes_to_mask(boxes.to(video.device), video.shape[-2], video.shape[-1]).to(video.dtype)
    return autoencoder.encoder(torch.cat([video, mask], 1))
