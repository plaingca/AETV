import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aetv import clip_cache
from aetv.config import AETV_MODES
from aetv.station import TxEngine


def _setup(tmp_path, monkeypatch):
    video = tmp_path / "my clip.mp4"
    checkpoint = tmp_path / "model.pt"
    video.write_bytes(b"video")
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(clip_cache, "clip_cache_dir", lambda: tmp_path / "cache")
    return video, checkpoint


def _codec(checkpoint):
    mode = AETV_MODES["V0"]
    encoded = []

    def encode(frames):
        encoded.append(frames.copy())
        return np.full(mode.latents_per_gop, len(encoded), dtype=np.float32)

    return SimpleNamespace(
        mode=mode, checkpoint_path=checkpoint, backend="torch", encode_gop=encode,
        encoded=encoded,
    )


def _engine(codec):
    return TxEngine(SimpleNamespace(require_codec=lambda: codec, codec_lock=threading.Lock(), log=lambda _message: None))


def _frames(mode, count=2):
    return np.full((count * mode.gop_frames, mode.height, mode.width, 3), 123, dtype=np.uint8)


def test_prepared_clip_restores_in_a_new_session_without_decoding_or_encoding(tmp_path, monkeypatch):
    video, checkpoint = _setup(tmp_path, monkeypatch)
    first = _codec(checkpoint)
    monkeypatch.setattr("aetv.station.iter_video_file", lambda *_a, **_k: _frames(first.mode))
    prepared = _engine(first).prepare_clip(str(video), "V0", 2, start_s=3.0, framing="fit")
    assert len(first.encoded) == 2

    def unexpected_decode(*_a, **_k):
        raise AssertionError("restoring a cached clip must not decode the source again")

    monkeypatch.setattr("aetv.station.iter_video_file", unexpected_decode)
    second = _codec(checkpoint)
    progress = []
    restored = _engine(second).prepare_clip(str(video), "V0", 2, progress.append, start_s=3.0, framing="fit")
    assert second.encoded == []
    assert restored.path == str(video) and restored.start_s == 3.0
    assert np.array_equal(restored.latents, prepared.latents)
    assert np.array_equal(restored.preview_frames, prepared.preview_frames)
    assert progress == [1.0]


@pytest.mark.parametrize("change", ["source", "model", "start", "length", "framing", "mode", "encoder"])
def test_cache_identity_changes_with_preparation_inputs(tmp_path, monkeypatch, change):
    video, checkpoint = _setup(tmp_path, monkeypatch)
    codec = _codec(checkpoint)
    encoder = tmp_path / "model.encoder.onnx"
    encoder.write_bytes(b"encoder")
    codec.backend = "onnxruntime"
    codec.args = {"encoder": encoder.name}
    count, start, framing = 2, 3.0, "fit"
    before = clip_cache.prepared_clip_path(codec, str(video), count, start, framing)
    if change == "source":
        video.write_bytes(b"changed video")
    elif change == "model":
        checkpoint.write_bytes(b"changed model")
    elif change == "encoder":
        encoder.write_bytes(b"changed encoder")
    elif change == "start":
        start = 4.0
    elif change == "length":
        count = 3
    elif change == "framing":
        framing = "crop"
    else:
        codec.mode = AETV_MODES["V8"]
    assert clip_cache.prepared_clip_path(codec, str(video), count, start, framing) != before


@pytest.mark.parametrize("corruption", ["truncated", "wrong_shape", "nonfinite"])
def test_bad_cache_is_rebuilt(tmp_path, monkeypatch, corruption):
    video, checkpoint = _setup(tmp_path, monkeypatch)
    codec = _codec(checkpoint)
    monkeypatch.setattr("aetv.station.iter_video_file", lambda *_a, **_k: _frames(codec.mode))
    engine = _engine(codec)
    prepared = engine.prepare_clip(str(video), "V0", 2)
    path = clip_cache.prepared_clip_path(codec, str(video), 2, 0.0, "crop")
    if corruption == "truncated":
        path.write_bytes(b"broken")
    else:
        latents = np.asarray(prepared.latents).copy()
        if corruption == "wrong_shape":
            latents = latents[:, :-1]
        else:
            latents[0, 0] = np.nan
        np.savez(path, latents=latents, preview=prepared.preview_frames)
    engine.prepare_clip(str(video), "V0", 2)
    assert len(codec.encoded) == 4
    assert clip_cache.load_prepared_clip(path, str(video), codec.mode, 2, 0.0) is not None


def test_unwritable_cache_does_not_prevent_preparation(tmp_path, monkeypatch):
    video, checkpoint = _setup(tmp_path, monkeypatch)
    codec = _codec(checkpoint)
    monkeypatch.setattr("aetv.station.iter_video_file", lambda *_a, **_k: _frames(codec.mode))
    messages = []
    engine = _engine(codec)
    engine.station.log = messages.append

    def denied(*_args):
        raise OSError("disk full")

    monkeypatch.setattr("aetv.station.save_prepared_clip", denied)
    prepared = engine.prepare_clip(str(video), "V0", 2)
    assert prepared.gops == 2 and len(codec.encoded) == 2
    assert "could not be cached" in messages[0]


def test_interrupted_cache_write_preserves_previous_entry(tmp_path, monkeypatch):
    video, checkpoint = _setup(tmp_path, monkeypatch)
    codec = _codec(checkpoint)
    monkeypatch.setattr("aetv.station.iter_video_file", lambda *_a, **_k: _frames(codec.mode))
    prepared = _engine(codec).prepare_clip(str(video), "V0", 2)
    path = clip_cache.prepared_clip_path(codec, str(video), 2, 0.0, "crop")
    original = path.read_bytes()

    def interrupted(*_args):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(clip_cache.os, "replace", interrupted)
    with pytest.raises(OSError):
        clip_cache.save_prepared_clip(path, prepared)
    assert path.read_bytes() == original
    assert list(path.parent.glob("*.tmp")) == []
