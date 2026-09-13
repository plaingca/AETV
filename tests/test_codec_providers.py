"""GPU selection must work with both packaged accelerator providers."""

import json
import sys
from types import SimpleNamespace

import pytest

from aetv.codec import AETVCodec


@pytest.mark.parametrize(
    "provider,requested,actual",
    [
        ("DmlExecutionProvider", "cuda", "dml"),
        ("CUDAExecutionProvider", "cuda", "cuda"),
        ("CPUExecutionProvider", "cpu", "cpu"),
        ("CPUExecutionProvider", "cuda", None),
    ],
)
def test_packaged_compute_selection(tmp_path, monkeypatch, provider, requested, actual):
    available = list(dict.fromkeys([provider, "CPUExecutionProvider"]))
    sessions = []

    def session(_path, *, sess_options, providers):
        sessions.append(providers)
        return SimpleNamespace(get_providers=lambda: providers)

    runtime = SimpleNamespace(
        __version__="test",
        get_available_providers=lambda: available,
        preload_dlls=lambda **_: None,
        SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=0),
        InferenceSession=session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    manifest = tmp_path / "test.runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "aetv-onnx-v1",
                "mode": "AC16",
                "encoder": "encoder.onnx",
                "decoder": "decoder.onnx",
            }
        )
    )
    (tmp_path / "encoder.onnx").touch()
    (tmp_path / "decoder.onnx").touch()
    codec = AETVCodec.__new__(AETVCodec)
    if actual is None:
        with pytest.raises(RuntimeError, match="GPU inference is unavailable"):
            codec._init_onnx(manifest, device=requested, requested_mode="AC16")
        assert not sessions
    else:
        codec._init_onnx(manifest, device=requested, requested_mode="AC16")
        assert codec.device.type == actual
        expected = [provider] if actual == "cpu" else [provider, "CPUExecutionProvider"]
        assert sessions == [expected, expected]
