"""The fresh DeepStream HF codec fills the 2,816-real wire and does not load other checkpoints."""

import numpy as np
import torch

from aetv.deepstream_hf import (
    FEATURES,
    GOP_FRAMES,
    GOP_VALUES,
    HEIGHT,
    REPS,
    WIDTH,
    DeepStreamHF,
    DownsampleReference,
    apply_hf_fade,
    build_placement,
    load_deepstream_hf,
    reference_fits,
    REFERENCE_CANDIDATES,
)


def test_placement_fills_the_wire_once():
    index, signs = build_placement(FEATURES, REPS)
    assert index.shape == (FEATURES, REPS)
    assert len(np.unique(index)) == GOP_VALUES
    assert set(np.unique(signs).tolist()) <= {-1.0, 1.0}


def test_every_reference_candidate_fits_and_roundtrips_a_shape():
    video = np.random.default_rng(0).random((GOP_FRAMES, 3, HEIGHT, WIDTH)).astype(np.float32)
    for config in REFERENCE_CANDIDATES:
        assert reference_fits(config)
        reference = DownsampleReference(config)
        wire = reference.transmit(video)
        assert wire.shape == (GOP_VALUES,)
        assert abs(float(np.mean(wire**2)) - 1.0) < 1e-4
        recon = reference.receive(wire, np.ones_like(wire))
        assert recon.shape == video.shape
        assert np.isfinite(recon).all()


def test_encode_is_unit_rms_and_decode_matches_the_contract():
    model = DeepStreamHF().eval()
    video = torch.rand(2, 3, GOP_FRAMES, HEIGHT, WIDTH)
    wire = model.encode_gop(video, retain_state=False)
    assert wire.shape == (2, GOP_VALUES)
    power = wire.square().mean(-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    received, confidence = apply_hf_fade(wire, sigma=0.4)
    recon, feature_confidence = model.decode_gop(received, confidence=confidence, retain_state=False)
    assert recon.shape == video.shape
    assert feature_confidence.shape == (2, FEATURES)
    assert float(recon.min()) >= 0.0
    assert float(recon.max()) <= 1.0


def test_loader_rejects_a_foreign_architecture(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"architecture": "ac2k-psnr", "model": {}}, path)
    try:
        load_deepstream_hf(str(path))
    except RuntimeError as exc:
        assert "deepstream-hf-v1" in str(exc)
    else:
        raise AssertionError("foreign checkpoint was loaded")
