"""Frozen RAFT-Small alignment used by receiver-side feature TCM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def raft_padding(height: int, width: int) -> tuple[int, int, int, int]:
    """Symmetric padding to RAFT's >=128 and multiple-of-eight contract."""
    padded_height = max(128, math.ceil(height / 8) * 8)
    padded_width = max(128, math.ceil(width / 8) * 8)
    vertical = padded_height - height
    horizontal = padded_width - width
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    return left, right, top, bottom


def sample_padded_reference(
    reference: torch.Tensor,
    flow: torch.Tensor,
    *,
    left: int,
    top: int,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    """Backward-warp a padded NCHW reference with target-to-reference flow."""
    batch, _, padded_height, padded_width = reference.shape
    if flow.shape != (batch, 2, output_height, output_width):
        raise ValueError(
            f"flow shape {tuple(flow.shape)} does not match "
            f"{(batch, 2, output_height, output_width)}"
        )
    y, x = torch.meshgrid(
        torch.arange(output_height, device=reference.device, dtype=flow.dtype),
        torch.arange(output_width, device=reference.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = x.unsqueeze(0) + left + flow[:, 0]
    sample_y = y.unsqueeze(0) + top + flow[:, 1]
    grid_x = 2.0 * sample_x / max(1, padded_width - 1) - 1.0
    grid_y = 2.0 * sample_y / max(1, padded_height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        reference,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class RAFTAligner(nn.Module):
    """Frozen RAFT-Small target-to-reference aligner."""

    def __init__(self, device: torch.device | str = "cpu"):
        super().__init__()
        device = torch.device(device)
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transforms = weights.transforms()

    def estimate_flow(self, reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if reference.shape != target.shape or reference.ndim != 4:
            raise ValueError("reference and target must have the same NCHW shape")
        _, _, height, width = target.shape
        left, right, top, bottom = raft_padding(height, width)
        padding = (left, right, top, bottom)
        padded_reference = F.pad(reference, padding, mode="replicate")
        padded_target = F.pad(target, padding, mode="replicate")
        normalized_target, normalized_reference = self.transforms(
            padded_target, padded_reference
        )
        # RAFT(image1, image2) predicts image1-to-image2 flow.  Running target
        # first produces the backward sampling field required to warp the
        # previous reference into the current target coordinates.
        flow = self.model(normalized_target, normalized_reference)[-1]
        return flow[:, :, top : top + height, left : left + width]

    def warp_with_flow(self, reference: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        height, width = flow.shape[-2:]
        return sample_padded_reference(
            reference,
            flow,
            left=0,
            top=0,
            output_height=height,
            output_width=width,
        )

    def forward(self, reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.warp_with_flow(reference, self.estimate_flow(reference, target))
