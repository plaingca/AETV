"""Qualify accelerator numerics against the same model on the CPU."""

import numpy as np


def color_probes(mode):
    """Nonzero RGB clips with color, texture and motion; no external images."""
    t, y, x = np.meshgrid(
        np.arange(mode.gop_frames, dtype=np.float32),
        np.linspace(0, 1, mode.height, dtype=np.float32),
        np.linspace(0, 1, mode.width, dtype=np.float32), indexing="ij",
    )
    smooth = np.stack([
        0.15 + 0.7 * np.mod(x + 0.03 * t, 1.0),
        0.15 + 0.7 * y,
        0.5 + 0.3 * np.sin(2 * np.pi * (x + y + t / 20)),
    ])
    texture = np.stack([
        0.4 + 0.2 * np.sin(18 * x + t / 4),
        0.4 + 0.2 * np.cos(21 * y - t / 5),
        0.4 + 0.2 * np.sin(15 * (x + y) - t / 3),
    ])
    return [np.ascontiguousarray(clip[None], dtype=np.float32)
            for clip in (smooth, texture)]


def reference_cases(encoder, decoder, mode):
    cases = []
    for frames in color_probes(mode):
        latents = encoder.run(None, {"frames": frames})[0]
        for confidence in (1.0, 0.5):
            weights = np.full_like(latents, confidence)
            decoded = decoder.run(None, {"latents": latents, "weights": weights})[0]
            if not np.all(np.isfinite(latents)) or not np.all(np.isfinite(decoded)):
                raise RuntimeError("CPU model reference produced nonfinite values")
            cases.append((frames, latents, weights, decoded))
    return cases


def check_sessions(encoder, decoder, cases):
    """Check encoder independently and rendered GOPs end to end, including RGB bias."""
    measurements = []
    for frames, reference_latents, weights, reference_frames in cases:
        latents = encoder.run(None, {"frames": frames})[0]
        if latents.shape != reference_latents.shape or not np.all(np.isfinite(latents)):
            raise RuntimeError("accelerator encoder returned invalid latents")
        latent_error = float(np.sqrt(np.mean((latents - reference_latents) ** 2)))
        latent_error /= max(float(np.sqrt(np.mean(reference_latents ** 2))), 1e-6)
        decoded = decoder.run(None, {"latents": latents, "weights": weights})[0]
        if decoded.shape != reference_frames.shape or not np.all(np.isfinite(decoded)):
            raise RuntimeError("accelerator decoder returned invalid frames")
        difference = np.clip(decoded, 0, 1) - np.clip(reference_frames, 0, 1)
        mae = float(np.mean(np.abs(difference)))
        maximum = float(np.max(np.abs(difference)))
        rgb_bias = np.mean(difference, axis=(0, 2, 3, 4))
        measurement = {
            "latent_relative_rmse": latent_error,
            "rgb_mean_error_levels": mae * 255,
            "rgb_max_error_levels": maximum * 255,
            "rgb_bias_levels": (rgb_bias * 255).tolist(),
        }
        measurements.append(measurement)
        if latent_error > 0.002 or mae > 1 / 255 or maximum > 8 / 255:
            raise RuntimeError(
                "accelerator differs from CPU reference "
                f"(latent relative error {latent_error:.5f}, "
                f"RGB mean/max error {mae * 255:.2f}/{maximum * 255:.2f} levels)"
            )
    return measurements


def qualified_directml_sessions(ort, encoder_path, decoder_path, mode, threads):
    """Try normal and conservative DirectML, retaining CPU on incorrect output."""
    def sessions(providers, conservative=False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.enable_mem_pattern = providers == ["CPUExecutionProvider"]
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if conservative:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.add_session_config_entry("ep.dml.disable_graph_fusion", "1")
        return tuple(ort.InferenceSession(
            str(path), sess_options=options, providers=providers,
        ) for path in (encoder_path, decoder_path))

    cpu = sessions(["CPUExecutionProvider"])
    cases = reference_cases(*cpu, mode)
    attempts = []
    profiles = [
        ("default", ["DmlExecutionProvider", "CPUExecutionProvider"], False),
        ("compatible", [("DmlExecutionProvider", {"disable_metacommands": "true"}),
                        "CPUExecutionProvider"], True),
    ]
    for profile, providers, conservative in profiles:
        candidate = None
        try:
            candidate = sessions(providers, conservative)
            if any("DmlExecutionProvider" not in session.get_providers() for session in candidate):
                raise RuntimeError("DirectML provider did not initialize")
            measurements = check_sessions(*candidate, cases)
        except Exception as error:
            attempts.append({"profile": profile, "passed": False, "reason": str(error)})
            candidate = None
            continue
        attempts.append({"profile": profile, "passed": True, "cases": measurements})
        return candidate, {"selected": profile, "attempts": attempts}
    return cpu, {"selected": "cpu", "attempts": attempts}
