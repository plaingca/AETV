import torch

from aetv.adaptive_keyframes import (
    adaptive_schedule,
    collapse_repetitions,
    interpolation_importance,
    pack_keyframes,
    reconstruct_timeline,
    uniform_schedule,
)


def ramp(frames: int = 11) -> torch.Tensor:
    values = torch.linspace(0, 1, frames).view(1, frames, 1, 1, 1)
    return values.expand(1, frames, 3, 4, 5).clone()


def test_linear_video_has_zero_interpolation_importance():
    scores = interpolation_importance(ramp(), (0, 10))
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-6)


def test_adaptive_schedule_selects_noninterpolable_frame_and_fills_slots():
    video = ramp()
    video[:, 5] = 1.0
    schedule = adaptive_schedule(video, 3, codec_slots=6)
    assert schedule.positions == (0, 5, 10)
    assert sum(schedule.repeats) == 6
    assert all(value >= 1 for value in schedule.repeats)


def test_pack_collapse_and_reconstruct_round_trip_selected_frames():
    video = ramp()
    schedule = adaptive_schedule(video, 4, codec_slots=6)
    packed = pack_keyframes(video, schedule)
    assert packed.shape == (1, 3, 6, 4, 5)
    keys = collapse_repetitions(packed, schedule)
    reconstructed = reconstruct_timeline(keys, schedule, source_frames=11)
    assert reconstructed.shape == (1, 3, 11, 4, 5)
    for index, position in enumerate(schedule.positions):
        assert torch.equal(reconstructed[:, :, position], keys[:, index])


def test_uniform_schedule_matches_even_12fps_sampling():
    schedule = uniform_schedule(11, 6)
    assert schedule.positions == (0, 2, 4, 6, 8, 10)
    assert schedule.repeats == (1, 1, 1, 1, 1, 1)


def test_six_key_adaptive_schedule_is_not_forced_to_uniform_positions():
    video = ramp()
    video[:, 1] = 1.0
    schedule = adaptive_schedule(video, 6, codec_slots=6)
    assert 1 in schedule.positions
    assert schedule.repeats == (1, 1, 1, 1, 1, 1)
