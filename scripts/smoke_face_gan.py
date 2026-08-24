#!/usr/bin/env python3
"""Small CUDA backward check for the differentiable face-crop GAN path."""

import torch

from aetv.attention import face_crop_grid, sample_face_crops
from aetv.models import SpatioTemporalPatchGAN3D


device = torch.device("cuda")
video = torch.rand(2, 3, 12, 108, 192, device=device, requires_grad=True)
mask = torch.zeros(2, 1, 12, 108, 192, device=device)
mask[0, :, :, 25:80, 65:125] = 1.0
grid, indices = face_crop_grid(
    mask, torch.tensor([True, False], device=device), crop_size=64
)
crops = sample_face_crops(video, grid, indices)
critic = SpatioTemporalPatchGAN3D(base_channels=32).to(device)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits, _ = critic(crops)
    loss = -logits.mean()
loss.backward()
print(crops.shape, float(loss), float(video.grad.abs().sum()))
