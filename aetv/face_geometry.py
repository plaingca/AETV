"""Facial landmark geometry through the channel.

A frozen 2D-FAN (``face_alignment``, 68 landmarks) runs on the same face crop of
the source and of a reconstruction. Crops follow the FAN convention around the
largest YuNet box per source frame: centre shifted up by 0.12 h, side
1.03 (w + h), resampled to 256x256. Landmark error is the mean L2 distance
between the two landmark sets as a percentage of the crop side (box-normalized
NME), which stays stable on faces a few tens of pixels wide.

The same crops and heatmaps give a differentiable consistency loss: the
reconstruction's FAN heatmaps are pulled toward the source's. FAN is a
reference network on the source, so the loss cannot be met by inventing
texture that is not in the source.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CROP = 256
HEATMAP = 64


def fan_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """YuNet (..., 4) x, y, w, h -> (..., 3) centre x, centre y, side in pixels (NaN kept)."""
    x, y, w, h = boxes.unbind(-1)
    cx = x + 0.5 * w
    cy = y + 0.5 * h - 0.12 * h
    side = 1.03 * (w + h)
    return torch.stack([cx, cy, side], -1)


def crop_faces(frames: torch.Tensor, boxes: torch.Tensor, size: int = CROP) -> torch.Tensor:
    """Resample square crops. ``frames`` (N, 3, H, W) in [0, 1], ``boxes`` (N, 3) centre/side in pixels."""
    n, _, h, w = frames.shape
    cx, cy, side = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    theta = torch.zeros(n, 2, 3, device=frames.device, dtype=frames.dtype)
    theta[:, 0, 0] = side / w
    theta[:, 1, 1] = side / h
    theta[:, 0, 2] = 2 * cx / w - 1
    theta[:, 1, 2] = 2 * cy / h - 1
    grid = F.affine_grid(theta, (n, 3, size, size), align_corners=False)
    return F.grid_sample(frames, grid, mode="bilinear", padding_mode="border", align_corners=False)


def heatmap_landmarks(heatmaps: torch.Tensor) -> torch.Tensor:
    """FAN decoding: argmax plus a quarter-pixel step toward the larger neighbour. (N, 68, 2) in heatmap pixels."""
    n, k, hh, ww = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx = flat.argmax(-1)
    ys, xs = (idx // ww).float(), (idx % ww).float()
    xi, yi = idx % ww, idx // ww
    pad = F.pad(heatmaps, (1, 1, 1, 1))
    b = torch.arange(n, device=heatmaps.device)[:, None].expand(n, k)
    c = torch.arange(k, device=heatmaps.device)[None, :].expand(n, k)
    dx = pad[b, c, yi + 1, xi + 2] - pad[b, c, yi + 1, xi]
    dy = pad[b, c, yi + 2, xi + 1] - pad[b, c, yi, xi + 1]
    return torch.stack([xs + 0.25 * torch.sign(dx) + 0.5, ys + 0.25 * torch.sign(dy) + 0.5], -1)


class FaceGeometry:
    """Frozen 2D-FAN landmarks on YuNet-boxed face crops."""

    def __init__(self, device):
        import face_alignment

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=str(device),
                                          face_detector="blazeface")
        self.net = fa.face_alignment_net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def heatmaps(self, crops: torch.Tensor) -> torch.Tensor:
        return self.net(crops)[-1]

    def frame_crops(self, video: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``video`` (3, T, H, W) or (B, 3, T, H, W); ``boxes`` (T, 4) or (B, T, 4) YuNet xywh.

        Returns crops for the frames that have a face and their flat (b*T + t) indices.
        """
        if video.ndim == 4:
            video, boxes = video[None], boxes[None]
        b, c, t, h, w = video.shape
        flat_boxes = fan_boxes(boxes.reshape(b * t, 4).to(video.device, video.dtype))
        keep = torch.isfinite(flat_boxes).all(-1)
        index = torch.nonzero(keep).squeeze(-1)
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)[index]
        return crop_faces(frames, flat_boxes[index]), index

    @torch.no_grad()
    def nme(self, source: torch.Tensor, recon: torch.Tensor, boxes: torch.Tensor) -> float | None:
        """Mean box-normalized landmark error (% of crop side) over frames with a face, or None."""
        src_crops, index = self.frame_crops(source.float(), boxes)
        if index.numel() == 0:
            return None
        rec_crops, _ = self.frame_crops(recon.float(), boxes)
        a = heatmap_landmarks(self.heatmaps(src_crops))
        b = heatmap_landmarks(self.heatmaps(rec_crops))
        return float((a - b).norm(dim=-1).mean() / HEATMAP * 100)

    def heatmap_loss(self, recon: torch.Tensor, source: torch.Tensor, boxes: torch.Tensor,
                     max_crops: int | None = None) -> torch.Tensor:
        """MSE between FAN heatmaps of reconstruction and source crops (source detached)."""
        src_crops, index = self.frame_crops(source, boxes)
        if index.numel() == 0:
            return recon.new_zeros(())
        if max_crops is not None and index.numel() > max_crops:
            pick = torch.randperm(index.numel(), device=index.device)[:max_crops]
            src_crops, index = src_crops[pick], index[pick]
        b, c, t, h, w = recon.shape
        frames = recon.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)[index]
        flat_boxes = fan_boxes(boxes.reshape(b * t, 4).to(recon.device, recon.dtype))[index]
        rec_crops = crop_faces(frames, flat_boxes)
        with torch.no_grad():
            target = self.heatmaps(src_crops.float())
        return F.mse_loss(self.heatmaps(rec_crops.float()), target)
