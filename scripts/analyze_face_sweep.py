#!/usr/bin/env python3
"""Compare facial high-frequency fidelity in LR-sweep evaluation sheets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aetv.attention import landmark_face_mask


ROOT = Path("runs/v8-hf3k-face-detail-lr-sweep-20260824")
RUNS = {
    "source12k": Path(
        "runs/v8-hf3k-detail-face-saliency-20260824/eval_step_012000"
    ),
    "lr1e-6": ROOT / "lr-1e-6/eval_step_000400",
    "lr3e-6": ROOT / "lr-3e-6/eval_step_000400",
    "lr8e-6": ROOT / "lr-8e-6/eval_step_000400",
    "lr2e-5": ROOT / "lr-2e-5/eval_step_000400",
    "v2-step0": Path("runs/v8-hf3k-face-detail-v2-20260824/eval_step_000000"),
    "v2-step1000": Path("runs/v8-hf3k-face-detail-v2-20260824/eval_step_001000"),
    "v2-step2000": Path("runs/v8-hf3k-face-detail-v2-20260824/eval_step_002000"),
    "v2-step3000": Path("runs/v8-hf3k-face-detail-v2-20260824/eval_step_003000"),
    "v2-step4000": Path("runs/v8-hf3k-face-detail-v2-20260824/eval_step_004000"),
    "gan-base": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.005/eval_step_000000"
    ),
    "gan005-300": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.005/eval_step_000300"
    ),
    "gan005-600": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.005/eval_step_000600"
    ),
    "gan010-300": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.010/eval_step_000300"
    ),
    "gan010-600": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.010/eval_step_000600"
    ),
    "gan020-300": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.020/eval_step_000300"
    ),
    "gan020-600": Path(
        "runs/v8-hf3k-face-gan-sweep-20260824/adv-0.020/eval_step_000600"
    ),
}
CONDITIONS = {
    "clean": (0, 1),
    "18db": (0, 2),
    "12db": (0, 3),
    "6db": (1, 0),
    "0db": (1, 1),
    "ota": (1, 3),
    "mpp": (2, 0),
}


def laplacian(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return np.abs(
        4 * image[1:-1, 1:-1]
        - image[:-2, 1:-1]
        - image[2:, 1:-1]
        - image[1:-1, :-2]
        - image[1:-1, 2:]
    )


def score(folder: Path, detector) -> tuple[int, dict[str, tuple[float, float]]]:
    errors = {key: [] for key in CONDITIONS}
    retention = {key: [] for key in CONDITIONS}
    found = 0
    for path in sorted(folder.glob("*.png")):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        cell_height, cell_width = height // 4, width // 4
        source = image[:cell_height, :cell_width]
        detector.setInputSize((cell_width, cell_height))
        _, faces = detector.detect(source)
        if faces is None:
            continue
        found += 1
        mask = landmark_face_mask(faces, cell_height, cell_width).numpy()[1:-1, 1:-1]
        weights = mask / (mask.sum() + 1e-8)
        source_laplacian = laplacian(source).mean(axis=2)
        source_energy = float((weights * source_laplacian).sum())
        for name, (row, column) in CONDITIONS.items():
            tile = image[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
            ]
            tile_laplacian = laplacian(tile).mean(axis=2)
            errors[name].append(float((weights * np.abs(tile_laplacian - source_laplacian)).sum()))
            retention[name].append(
                float((weights * tile_laplacian).sum() / (source_energy + 1e-8))
            )
    return found, {
        key: (float(np.mean(errors[key])), float(np.mean(retention[key])))
        for key in CONDITIONS
    }


def main() -> None:
    detector = cv2.FaceDetectorYN.create(
        "data/teachers/face_detection_yunet_2023mar.onnx",
        "",
        (130, 74),
        0.55,
        0.3,
        5000,
    )
    print(
        "run faces clean_err clean_ret 12db_err 12db_ret "
        "0db_err 0db_ret ota_err ota_ret mpp_err mpp_ret"
    )
    for name, path in RUNS.items():
        count, scores = score(path, detector)
        values = [
            value
            for condition in ("clean", "12db", "0db", "ota", "mpp")
            for value in scores[condition]
        ]
        print(name, count, *(f"{value:.5f}" for value in values))


if __name__ == "__main__":
    main()
