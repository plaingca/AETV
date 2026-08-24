"""Training-time region attention for preserving faces and foreground detail."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def face_crop_grid(
    mask: torch.Tensor,
    face_clips: torch.Tensor,
    crop_size: int = 64,
    context: float = 1.30,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build stabilized sampling grids around the landmark face masks.

    The mask's weighted moments avoid a CPU round-trip and remain robust to the
    soft facial oval.  Centers and scale are smoothed over time so the face
    critic cannot mistake detector/crop jitter for high-frequency detail.
    """
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("face mask must have shape (B, 1, T, H, W)")
    indices = face_clips.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return None, indices

    selected = mask.index_select(0, indices).float().squeeze(1).clamp_min(0.0)
    n, frames, height, width = selected.shape
    # Squaring suppresses the low-weight oval outskirts while retaining the
    # landmark hotspots that define the useful face crop.
    weights = selected.square()
    mass = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
    xs = torch.linspace(0.0, float(width - 1), width, device=mask.device)
    ys = torch.linspace(0.0, float(height - 1), height, device=mask.device)
    center_x = (weights * xs.view(1, 1, 1, width)).sum(dim=(-2, -1)) / mass
    center_y = (weights * ys.view(1, 1, height, 1)).sum(dim=(-2, -1)) / mass
    var_x = (
        weights * (xs.view(1, 1, 1, width) - center_x[..., None, None]).square()
    ).sum(dim=(-2, -1)) / mass
    var_y = (
        weights * (ys.view(1, 1, height, 1) - center_y[..., None, None]).square()
    ).sum(dim=(-2, -1)) / mass
    # About +/- 2.7 sigma contains the facial oval; a little context prevents
    # the critic from exploiting an artificial crop boundary.
    half_size = context * 2.7 * torch.maximum(var_x.sqrt(), var_y.sqrt())
    half_size = half_size.clamp(min=6.0, max=0.48 * min(height, width))

    def smooth(values: torch.Tensor) -> torch.Tensor:
        if frames < 3:
            return values
        padded = F.pad(values[:, None], (1, 1), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=3, stride=1).squeeze(1)

    center_x = smooth(center_x)
    center_y = smooth(center_y)
    half_size = smooth(half_size)

    axis = torch.linspace(-1.0, 1.0, crop_size, device=mask.device)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    sample_x = center_x[..., None, None] + half_size[..., None, None] * gx
    sample_y = center_y[..., None, None] + half_size[..., None, None] * gy
    grid = torch.stack(
        (
            2.0 * sample_x / max(1, width - 1) - 1.0,
            2.0 * sample_y / max(1, height - 1) - 1.0,
        ),
        dim=-1,
    )
    return grid.reshape(n * frames, crop_size, crop_size, 2), indices


def sample_face_crops(
    video: torch.Tensor,
    grid: torch.Tensor | None,
    face_indices: torch.Tensor,
) -> torch.Tensor | None:
    """Sample Bx3xTxHxW video with a grid returned by :func:`face_crop_grid`."""
    if grid is None or face_indices.numel() == 0:
        return None
    selected = video.index_select(0, face_indices)
    n, channels, frames, height, width = selected.shape
    images = selected.permute(0, 2, 1, 3, 4).reshape(
        n * frames, channels, height, width
    )
    crops = F.grid_sample(
        images, grid.to(images.dtype), mode="bilinear", padding_mode="border",
        align_corners=True,
    )
    return crops.reshape(n, frames, channels, crops.shape[-2], crops.shape[-1]).permute(
        0, 2, 1, 3, 4
    )


def normalized_region_weights(mask: torch.Tensor, boost: float) -> torch.Tensor:
    """Turn a soft Bx1xTxHxW mask into unit-mean spatial weights."""
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("region mask must have shape (B, 1, T, H, W)")
    weights = 1.0 + float(boost) * mask.float().clamp(0.0, 1.0)
    return weights / weights.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)


def region_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """Reference reconstruction loss biased toward the selected region."""
    weights = normalized_region_weights(mask, boost).to(recon.dtype)
    error = recon - target
    # Keep a small quadratic component for convergence without letting MSE's
    # conditional-mean optimum dominate the localized objective and soften
    # ambiguous facial texture.
    return (weights * error.abs()).mean() + 0.1 * (weights * error.square()).mean()


def region_gradient_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """Match horizontal/vertical edges, emphasizing edges inside the mask."""
    weights = normalized_region_weights(mask, boost).to(recon.dtype)
    dx = (recon[..., 1:] - recon[..., :-1]) - (
        target[..., 1:] - target[..., :-1]
    )
    dy = (recon[..., 1:, :] - recon[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    wx = 0.5 * (weights[..., 1:] + weights[..., :-1])
    wy = 0.5 * (weights[..., 1:, :] + weights[..., :-1, :])
    return (wx * dx.abs()).mean() + (wy * dy.abs()).mean()


def region_detail_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """Match two spatial high-pass bands to directly penalize soft renders."""
    total = recon.new_zeros(())
    pred = recon.float()
    truth = target.float()
    region = mask.float()
    for scale in (3, 5):
        padding = scale // 2
        pred_high = pred - F.avg_pool3d(
            pred, (1, scale, scale), stride=1, padding=(0, padding, padding)
        )
        truth_high = truth - F.avg_pool3d(
            truth, (1, scale, scale), stride=1, padding=(0, padding, padding)
        )
        weights = normalized_region_weights(region, boost)
        total = total + (weights * (pred_high - truth_high).abs()).mean()
    return total / 2.0


def region_contrast_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """Match local spatial contrast at two scales inside the selected region."""
    weights = normalized_region_weights(mask, boost)
    total = recon.new_zeros(())
    pred = recon.float()
    truth = target.float()
    for scale in (3, 7):
        padding = scale // 2

        def local_std(x: torch.Tensor) -> torch.Tensor:
            mean = F.avg_pool3d(
                x, (1, scale, scale), stride=1, padding=(0, padding, padding)
            )
            mean_square = F.avg_pool3d(
                x.square(), (1, scale, scale), stride=1, padding=(0, padding, padding)
            )
            return (mean_square - mean.square()).clamp_min(1e-6).sqrt()

        total = total + (weights * (local_std(pred) - local_std(truth)).abs()).mean()
    return total / 2.0


def landmark_face_mask(
    detections: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Make a soft face-and-features mask from YuNet boxes and landmarks.

    YuNet returns five landmarks after each box: two eyes, nose, and two mouth
    corners. A low-weight facial oval retains cheeks/forehead while tight
    Gaussian hotspots put most of the objective on eyes, nose and mouth. This
    deliberately avoids the hair, torso and background included by expanded
    rectangular boxes.
    """
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros((height, width), dtype=torch.float32)
    for detection in np.asarray(detections):
        x, y, box_width, box_height = map(float, detection[:4])
        if box_width <= 1.0 or box_height <= 1.0:
            continue
        landmarks = np.asarray(detection[4:14], dtype=np.float32).reshape(5, 2)
        center_x = x + 0.50 * box_width
        center_y = y + 0.49 * box_height
        radius_x = max(2.0, 0.47 * box_width)
        radius_y = max(2.0, 0.50 * box_height)
        distance = ((xx - center_x) / radius_x).square() + (
            (yy - center_y) / radius_y
        ).square()
        oval = torch.exp(-2.2 * distance) * (distance <= 2.0)

        sigma = max(1.25, 0.075 * max(box_width, box_height))
        feature = torch.zeros_like(mask)
        # Eyes and mouth carry more identity/detail than the nose, but retaining
        # all five landmarks prevents the learned result from becoming a set of
        # disconnected sharpened dots.
        strengths = (1.0, 1.0, 0.75, 0.95, 0.95)
        for (point_x, point_y), strength in zip(landmarks, strengths):
            gaussian = torch.exp(
                -((xx - float(point_x)).square() + (yy - float(point_y)).square())
                / (2.0 * sigma * sigma)
            )
            feature = torch.maximum(feature, float(strength) * gaussian)
        mask = torch.maximum(mask, torch.maximum(0.32 * oval, feature))
    return mask.clamp(0.0, 1.0)


class RegionAttentionTeacher:
    """Build face masks with YuNet, falling back to an LRASPP object mask.

    Face detection runs on CPU because the source clips are tiny and this avoids
    competing with the codec for GPU kernels. LRASPP handles only clips with no
    detected face and runs one center frame per clip on the selected device.
    """

    def __init__(
        self,
        face_model: str | Path,
        device: torch.device,
        *,
        face_score_threshold: float = 0.72,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "face-weighted training requires opencv-python-headless"
            ) from exc

        model_path = Path(face_model)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet face model not found: {model_path}")
        self.cv2 = cv2
        self.face_detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), face_score_threshold, 0.3, 5000
        )
        self.device = device

        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )

        self.saliency = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        ).to(device).eval()
        self.saliency.requires_grad_(False)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    def _face_masks(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, frames, height, width = video.shape
        masks = torch.zeros((batch, 1, frames, height, width), dtype=torch.float32)
        has_face = torch.zeros(batch, dtype=torch.bool)
        images = (
            video.detach().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 4, 1).cpu().numpy()
        )
        self.face_detector.setInputSize((width, height))
        for batch_index in range(batch):
            for frame_index in range(frames):
                bgr = np.ascontiguousarray(images[batch_index, frame_index, :, :, ::-1])
                _, detections = self.face_detector.detect(bgr)
                if detections is None:
                    continue
                has_face[batch_index] = True
                masks[batch_index, 0, frame_index] = landmark_face_mask(
                    detections, height, width
                )

        # Fill detector misses from the nearest successfully detected frame,
        # without max-pooling separate face positions together or widening the
        # masks that were already valid.
        for batch_index in range(batch):
            valid = masks[batch_index, 0].flatten(1).amax(dim=1) > 0
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            for frame_index in (~valid).nonzero(as_tuple=False).flatten():
                if len(valid_indices):
                    nearest = valid_indices[(valid_indices - frame_index).abs().argmin()]
                    masks[batch_index, 0, frame_index] = masks[
                        batch_index, 0, nearest
                    ]
        return masks, has_face

    @torch.inference_mode()
    def _object_masks(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        inputs = (frames.float() - self.mean) / self.std
        logits = self.saliency(inputs)["out"]
        logits = F.interpolate(logits, (height, width), mode="bilinear", align_corners=False)
        probabilities = logits.softmax(dim=1)

        # Choose the non-background class with the strongest spatial evidence,
        # then retain its confident core. This keeps one dominant semantic
        # object category instead of weighting every labelled foreground pixel.
        class_scores = probabilities[:, 1:].flatten(2).topk(
            k=max(1, height * width // 20), dim=2
        ).values.mean(dim=2)
        class_index = class_scores.argmax(dim=1) + 1
        selected = probabilities.gather(
            1, class_index[:, None, None, None].expand(-1, 1, height, width)
        )
        # A relative threshold still produces a useful model-guided focus mask
        # when the scene is outside the segmenter's 20 foreground classes.
        # An absolute confidence floor left landscapes and machinery with an
        # all-zero mask, defeating the non-face fallback entirely.
        peak = selected.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        floor = peak * 0.38
        masks = ((selected - floor) / (peak - floor).clamp_min(1e-4)).clamp(0, 1)
        masks = F.max_pool2d(masks, 9, stride=1, padding=4)
        return F.avg_pool2d(masks, 15, stride=1, padding=7)

    @torch.inference_mode()
    def __call__(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        face_masks, has_face = self._face_masks(video)
        face_masks = face_masks.to(self.device, non_blocking=True)
        no_face = ~has_face
        if bool(no_face.any()):
            no_face_indices = no_face.nonzero(as_tuple=False).flatten().to(video.device)
            center = video.index_select(0, no_face_indices)[:, :, video.shape[2] // 2]
            object_masks = self._object_masks(center)
            object_masks = object_masks[:, :, None].expand(-1, -1, video.shape[2], -1, -1)
            face_masks.index_copy_(0, no_face_indices.to(self.device), object_masks)
        return face_masks, has_face.to(self.device)
