"""Adaptive key-frame scheduling for narrowband-HF video experiments.

This module adapts the temporal stage of VQ-DeepVSC to AETV's fixed-rate
analog latent transport.  It deliberately does not quantize AETV latents or
pretend that repeated source frames are independent FEC symbols.  Instead it
selects source frames by interpolation residual, packs them into the six V8
codec slots, and reconstructs their original timeline after decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeyframeSchedule:
    """Selected source positions and repetitions in codec-slot order."""

    positions: tuple[int, ...]
    repeats: tuple[int, ...]
    importance: tuple[float, ...]

    def validate(self, source_frames: int, codec_slots: int) -> None:
        if len(self.positions) != len(self.repeats) or len(self.positions) != len(self.importance):
            raise ValueError("schedule fields must have equal lengths")
        if not self.positions or self.positions[0] != 0 or self.positions[-1] != source_frames - 1:
            raise ValueError("the first and last source frames must be selected")
        if tuple(sorted(set(self.positions))) != self.positions:
            raise ValueError("positions must be strictly increasing")
        if any(value < 1 for value in self.repeats) or sum(self.repeats) != codec_slots:
            raise ValueError("positive repeat counts must fill every codec slot")


def _gradient_error(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    dx_ref = reference[..., :, 1:] - reference[..., :, :-1]
    dx_est = estimate[..., :, 1:] - estimate[..., :, :-1]
    dy_ref = reference[..., 1:, :] - reference[..., :-1, :]
    dy_est = estimate[..., 1:, :] - estimate[..., :-1, :]
    return 0.5 * (
        (dx_ref - dx_est).abs().mean(dim=(-3, -2, -1))
        + (dy_ref - dy_est).abs().mean(dim=(-3, -2, -1))
    )


def _segment_estimates(video: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Linearly estimate all frames strictly between two C,H,W endpoints."""
    steps = right - left
    if steps <= 1:
        return video.new_empty((video.shape[0], 0, *video.shape[2:]))
    alpha = torch.arange(1, steps, device=video.device, dtype=video.dtype) / steps
    return (
        video[:, left : left + 1] * (1.0 - alpha[None, :, None, None, None])
        + video[:, right : right + 1] * alpha[None, :, None, None, None]
    )


def interpolation_importance(
    video: torch.Tensor,
    positions: tuple[int, ...],
    *,
    edge_weight: float = 0.25,
) -> torch.Tensor:
    """Return a per-frame interpolation residual for a B,T,C,H,W video."""
    if video.ndim != 5:
        raise ValueError(f"expected B,T,C,H,W video, got {tuple(video.shape)}")
    frames = video.shape[1]
    if frames < 2:
        raise ValueError("at least two source frames are required")
    scores = video.new_zeros((video.shape[0], frames))
    for left, right in zip(positions[:-1], positions[1:]):
        estimate = _segment_estimates(video, left, right)
        if estimate.shape[1] == 0:
            continue
        target = video[:, left + 1 : right]
        pixel = (target - estimate).abs().mean(dim=(-3, -2, -1))
        edge = _gradient_error(target, estimate)
        scores[:, left + 1 : right] = pixel + edge_weight * edge
    return scores


def _repeat_counts(importance: list[float], slots: int) -> tuple[int, ...]:
    if len(importance) > slots:
        raise ValueError("key-frame count cannot exceed codec slots")
    repeats = [1] * len(importance)
    # Diminishing returns prevent every spare slot from going to one frame.
    for _ in range(slots - len(importance)):
        winner = max(
            range(len(repeats)),
            key=lambda index: (importance[index] + 1e-9) / repeats[index],
        )
        repeats[winner] += 1
    return tuple(repeats)


def adaptive_schedule(
    video: torch.Tensor,
    keyframes: int,
    *,
    codec_slots: int = 6,
    edge_weight: float = 0.25,
) -> KeyframeSchedule:
    """Greedily add the least-interpolable frame until ``keyframes`` remain.

    The score is transmitter-side and source referenced, matching the paper's
    frame-importance calculation.  Batch size one is intentional: a schedule
    is metadata for one over-the-air GOP.
    """
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("adaptive scheduling expects one B,T,C,H,W GOP")
    frames = video.shape[1]
    if not 2 <= keyframes <= min(frames, codec_slots):
        raise ValueError("keyframes must fit between the two endpoints and codec slots")
    selected = [0, frames - 1]
    selection_scores = {0: 0.0, frames - 1: 0.0}
    while len(selected) < keyframes:
        ordered = tuple(sorted(selected))
        scores = interpolation_importance(video, ordered, edge_weight=edge_weight)[0]
        scores[selected] = -1.0
        winner = int(scores.argmax().item())
        selected.append(winner)
        selection_scores[winner] = float(scores[winner].item())
    positions = tuple(sorted(selected))

    # Endpoints can still be important at a cut entering/leaving the GOP.  Use
    # their adjacent-frame change as a finite retransmission priority.
    endpoint_change = (video[:, 1:] - video[:, :-1]).abs().mean(dim=(-3, -2, -1))[0]
    selection_scores[0] = float(endpoint_change[0].item())
    selection_scores[frames - 1] = float(endpoint_change[-1].item())
    importance = tuple(selection_scores[position] for position in positions)
    schedule = KeyframeSchedule(positions, _repeat_counts(list(importance), codec_slots), importance)
    schedule.validate(frames, codec_slots)
    return schedule


def uniform_schedule(source_frames: int = 11, codec_slots: int = 6) -> KeyframeSchedule:
    """The current V8 6-to-12 fps sampling pattern, represented explicitly."""
    if source_frames < codec_slots:
        raise ValueError("source timeline must contain at least as many frames as codec slots")
    positions = tuple(round(index * (source_frames - 1) / (codec_slots - 1)) for index in range(codec_slots))
    result = KeyframeSchedule(positions, (1,) * codec_slots, (0.0,) * codec_slots)
    result.validate(source_frames, codec_slots)
    return result


def pack_keyframes(video: torch.Tensor, schedule: KeyframeSchedule) -> torch.Tensor:
    """Pack B,T,C,H,W source frames into repeated B,C,S,H,W codec slots."""
    schedule.validate(video.shape[1], sum(schedule.repeats))
    pieces = [
        video[:, position : position + 1].expand(-1, repeats, -1, -1, -1)
        for position, repeats in zip(schedule.positions, schedule.repeats)
    ]
    return torch.cat(pieces, dim=1).permute(0, 2, 1, 3, 4).contiguous()


def collapse_repetitions(decoded: torch.Tensor, schedule: KeyframeSchedule) -> torch.Tensor:
    """Average repeated decoded slots into B,K,C,H,W key frames."""
    if decoded.ndim != 5:
        raise ValueError(f"expected B,C,S,H,W decoded video, got {tuple(decoded.shape)}")
    if decoded.shape[2] != sum(schedule.repeats):
        raise ValueError("decoded slot count does not match schedule")
    temporal = decoded.permute(0, 2, 1, 3, 4)
    keys = []
    offset = 0
    for repeats in schedule.repeats:
        keys.append(temporal[:, offset : offset + repeats].mean(dim=1))
        offset += repeats
    return torch.stack(keys, dim=1)


def reconstruct_timeline(
    keys: torch.Tensor,
    schedule: KeyframeSchedule,
    source_frames: int,
    *,
    scene_cut_threshold: float = 0.35,
) -> torch.Tensor:
    """Reconstruct B,C,T,H,W at the original integer source positions."""
    if keys.ndim != 5 or keys.shape[1] != len(schedule.positions):
        raise ValueError("keys must be B,K,C,H,W and match the schedule")
    output: list[torch.Tensor | None] = [None] * source_frames
    for index, position in enumerate(schedule.positions):
        output[position] = keys[:, index]
    for index, (left, right) in enumerate(zip(schedule.positions[:-1], schedule.positions[1:])):
        left_frame = keys[:, index]
        right_frame = keys[:, index + 1]
        cut = (left_frame - right_frame).abs().mean(dim=(1, 2, 3)) > scene_cut_threshold
        span = right - left
        for position in range(left + 1, right):
            alpha = (position - left) / span
            estimate = left_frame * (1.0 - alpha) + right_frame * alpha
            held = left_frame if alpha <= 0.5 else right_frame
            output[position] = torch.where(cut[:, None, None, None], held, estimate)
    if any(frame is None for frame in output):
        raise RuntimeError("schedule did not cover the source timeline")
    return torch.stack([frame for frame in output if frame is not None], dim=2)
