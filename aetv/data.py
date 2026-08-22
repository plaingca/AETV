"""Dataset loaders and OpenVid-1M live streaming pipeline for AETV.

Streams clips on-demand from OpenVid-1M on Hugging Face (`lance-format/Openvid-1M`),
resizing and extracting the exact GOP frame count and spatial dimensions
for the target AETV mode (V0-V7).
"""

from __future__ import annotations

import math
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .video_data import (
    HFDatasetsVideoDataset,
    HFViewerVideoDataset,
    SyntheticVideoDataset,
    VideoClipSpec,
)
from .config import AETV_MODES, AETVModeSpec


class AETVOpenVidStreamDataset(IterableDataset):
    """Live streaming dataset from Hugging Face OpenVid-1M for AETV."""

    def __init__(
        self,
        mode_spec: AETVModeSpec | str = "V1",
        dataset_name: str = "lance-format/Openvid-1M",
        loader_kind: str = "datasets",  # "datasets" or "viewer"
        epoch_size: int = 1000,
        seed: int = 0,
        cache_dir: str | None = None,
    ):
        super().__init__()
        if isinstance(mode_spec, str):
            self.mode_spec = AETV_MODES[mode_spec]
        else:
            self.mode_spec = mode_spec

        self.clip_spec = VideoClipSpec(
            frames=self.mode_spec.gop_frames,
            fps=self.mode_spec.fps,
            height=self.mode_spec.height,
            width=self.mode_spec.width,
        )
        self.dataset_name = dataset_name
        self.loader_kind = loader_kind
        self.epoch_size = epoch_size
        self.seed = seed
        self.cache_dir = cache_dir

    def __iter__(self) -> Iterator[torch.Tensor]:
        try:
            if self.loader_kind == "viewer":
                stream = HFViewerVideoDataset(
                    dataset=self.dataset_name,
                    spec=self.clip_spec,
                    epoch_size=self.epoch_size,
                    seed=self.seed,
                    cache_dir=self.cache_dir,
                )
            else:
                stream = HFDatasetsVideoDataset(
                    dataset=self.dataset_name,
                    spec=self.clip_spec,
                    epoch_size=self.epoch_size,
                    seed=self.seed,
                )
            for clip in stream:
                yield clip
        except Exception as error:
            # Fallback to procedural synthetic video if offline or streaming unavailable
            synthetic = AETVSyntheticVideoDataset(
                mode_spec=self.mode_spec,
                count=self.epoch_size,
                seed=self.seed,
            )
            for clip in synthetic:
                yield clip


class AETVSyntheticVideoDataset(Dataset):
    """Procedural synthetic video dataset for testing and offline execution."""

    def __init__(
        self,
        mode_spec: AETVModeSpec | str = "V1",
        count: int = 100,
        seed: int = 0,
    ):
        super().__init__()
        if isinstance(mode_spec, str):
            self.mode_spec = AETV_MODES[mode_spec]
        else:
            self.mode_spec = mode_spec
        self.count = count
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed + index)
        frames = self.mode_spec.gop_frames
        h, w = self.mode_spec.height, self.mode_spec.width

        # Generate moving synthetic shapes and color gradients
        video = np.zeros((3, frames, h, w), dtype=np.float32)
        base_color = rng.uniform(0.1, 0.9, size=(3, 1, 1, 1))

        # Grid coordinate maps
        y, x = np.mgrid[0:h, 0:w]
        x_norm = x / max(1, w - 1)
        y_norm = y / max(1, h - 1)

        speed_x = rng.uniform(-0.1, 0.1)
        speed_y = rng.uniform(-0.1, 0.1)

        for t in range(frames):
            shift_x = (x_norm + t * speed_x) % 1.0
            shift_y = (y_norm + t * speed_y) % 1.0
            grad = (np.sin(2 * np.pi * shift_x) * np.cos(2 * np.pi * shift_y) + 1.0) * 0.5
            video[0, t] = base_color[0, 0, 0, 0] * grad
            video[1, t] = base_color[1, 0, 0, 0] * (1.0 - grad)
            video[2, t] = base_color[2, 0, 0, 0] * (shift_x + shift_y) * 0.5

        return torch.from_numpy(video).clamp(0.0, 1.0)
