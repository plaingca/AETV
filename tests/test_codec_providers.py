"""GPU selection must work with both packaged accelerator providers."""

import json
import sys
from types import SimpleNamespace

import pytest

from aetv.codec import AETVCodec


@pytest.mark.parametrize(
    "provider,requested,actual,qualified_profile",
    [
        ("DmlExecutionProvider", "cuda", "dml", "default"),
        ("DmlExecutionProvider", "cuda", "cpu", "cpu"),
        ("CUDAExecutionProvider", "cuda", "cuda", None),
        ("CPUExecutionProvider", "cpu", "cpu", None),
        ("CPUExecutionProvider", "cuda", None, None),
    ],
)
def test_packaged_compute_selection(tmp_path, monkeypatch, provider, requested, actual, qualified_profile):
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
    def qualified(ort, encoder, decoder, _mode, _threads):
        providers = (["CPUExecutionProvider"] if qualified_profile == "cpu"
                     else [provider, "CPUExecutionProvider"])
        return tuple(session(path, sess_options=None, providers=providers)
                     for path in (encoder, decoder)), {"selected": qualified_profile}
    monkeypatch.setattr("aetv.runtime_validation.qualified_directml_sessions", qualified)
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
        expected = ["CPUExecutionProvider"] if actual == "cpu" else [provider, "CPUExecutionProvider"]
        assert sessions == [expected, expected]
        if qualified_profile == "cpu":
            assert "color accuracy check" in codec.runtime_notice
            assert codec.cpu_threads == 8
