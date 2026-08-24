#!/usr/bin/env python3
"""Summarize face-critic TensorBoard scores for completed sweep runs."""

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


root = Path("runs/v8-hf3k-face-gan-sweep-20260824")
tags = (
    "train/loss_face_disc",
    "train/face_disc_real_score",
    "train/face_disc_fake_score",
    "train/loss_face_adv",
)
for label in ("0.005", "0.010", "0.020"):
    accumulator = EventAccumulator(str(root / f"adv-{label}" / "tensorboard"))
    accumulator.Reload()
    print(label)
    for tag in tags:
        values = [event.value for event in accumulator.Scalars(tag) if event.step >= 100]
        if values:
            print(
                f"  {tag.rsplit('/', 1)[-1]}: mean={sum(values) / len(values):.5f} "
                f"last={values[-1]:.5f} min={min(values):.5f} max={max(values):.5f}"
            )
