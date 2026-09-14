"""A successful GPU call must not qualify an encoder/decoder with color errors."""

from types import SimpleNamespace

import numpy as np
import pytest

from aetv.runtime_validation import qualified_directml_sessions


@pytest.mark.parametrize(
    "failure,expected",
    [(None, "default"), ("default_decoder", "compatible"),
     ("all_decoder", "cpu"), ("all_encoder", "cpu"),
     ("nonfinite", "cpu"), ("missing_provider", "cpu")],
)
def test_accelerator_qualification_retries_or_returns_accurate_cpu(failure, expected):
    mode = SimpleNamespace(gop_frames=2, height=8, width=12)
    configured = []

    class Options:
        def __init__(self):
            self.config = {}

        def add_session_config_entry(self, key, value):
            self.config[key] = value

    class Session:
        def __init__(self, path, *, sess_options, providers):
            self.encoder = "encoder" in str(path)
            self.names = [p[0] if isinstance(p, tuple) else p for p in providers]
            self.gpu = "DmlExecutionProvider" in self.names
            self.compatible = sess_options.config.get("ep.dml.disable_graph_fusion") == "1"
            configured.append((providers, sess_options))

        def get_providers(self):
            return ["CPUExecutionProvider"] if failure == "missing_provider" else self.names

        def run(self, _, inputs):
            if self.encoder:
                value = inputs["frames"].mean(axis=(2, 3, 4)).copy()
                if self.gpu and failure == "all_encoder":
                    value[:, 1] += 0.1
            else:
                colors = inputs["latents"] * inputs["weights"]
                value = np.broadcast_to(colors[:, :, None, None, None], (1, 3, 2, 8, 12)).copy()
                if self.gpu and (failure == "all_decoder" or (
                    failure == "default_decoder" and not self.compatible
                )):
                    value[:, 0] += 0.2
                if self.gpu and failure == "nonfinite":
                    value[:, 0] = np.nan
            return [value]

    ort = SimpleNamespace(
        SessionOptions=Options, InferenceSession=Session,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL="disabled"),
    )
    sessions, report = qualified_directml_sessions(ort, "encoder", "decoder", mode, 2)
    assert report["selected"] == expected
    assert all(session.gpu == (expected != "cpu") for session in sessions)
    assert all(case["rgb_mean_error_levels"] == 0
               for attempt in report["attempts"] for case in attempt.get("cases", []))
    if expected != "default":
        assert report["attempts"][0]["passed"] is False
        assert any(isinstance(p[0], tuple) and p[0][1] == {"disable_metacommands": "true"}
                   for p, _ in configured)
    # The original request keeps the numerical checks visible to the user.
    if expected == "cpu":
        assert len(report["attempts"]) == 2
        assert all(not attempt["passed"] and attempt["reason"] for attempt in report["attempts"])
