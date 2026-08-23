import numpy as np

from aetv.gui.widgets import blend_gop_boundary


def test_gop_boundary_blend_reduces_jump_without_adding_frames():
    previous = np.full((2, 3, 3), 40, dtype=np.uint8)
    frames = np.stack(
        [np.full_like(previous, value) for value in (140, 150, 160, 170, 180)]
    )

    blended = blend_gop_boundary(previous, frames, transition_frames=4)

    assert blended.shape == frames.shape
    assert np.all(blended[0] == 60)  # 80% of the boundary offset is concealed.
    assert np.array_equal(blended[3:], frames[3:])
    assert np.array_equal(frames[0], np.full_like(previous, 140))  # input untouched


def test_gop_boundary_blend_preserves_new_gop_motion():
    previous = np.full((1, 1, 3), 20, dtype=np.uint8)
    frames = np.array([[[[120, 120, 120]]], [[[140, 140, 140]]]], dtype=np.uint8)

    blended = blend_gop_boundary(previous, frames, transition_frames=2)

    assert np.all(blended[0] == 40)
    assert np.array_equal(blended[1], frames[1])
