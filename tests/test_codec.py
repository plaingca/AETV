"""Checkpoint load and V7 encode/decode smoke test."""

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aetv.codec as codec_module
from aetv.codec import (
    AETVCodec,
    DEFAULT_MODE,
    RELEASE_CHECKPOINTS,
    download_default_checkpoint,
    download_runtime_bundle,
    resolve_checkpoint,
    resolve_runtime_bundle,
)
from aetv.config import AETV_MODES
from aetv.modem import demodulate_gop_stream, modulate_gop_stream

WIDE_CHECKPOINT = Path("models") / "v8-flex8k-ota-rxfix.pt"


def test_default_mode_is_the_standard_channel_release():
    assert DEFAULT_MODE == "V8"
    assert set(RELEASE_CHECKPOINTS) == {"V7", "V8"}


def test_resolve_checkpoint_from_environment(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    monkeypatch.setenv("AETV_CHECKPOINT", str(checkpoint))
    assert resolve_checkpoint() == checkpoint.resolve()


def test_default_checkpoint_download_is_atomic_and_verified(tmp_path, monkeypatch):
    payload = b"published model bytes"
    monkeypatch.setitem(
        codec_module.RELEASE_CHECKPOINTS,
        "TEST",
        {
            "filename": "test.pt",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(codec_module.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "models" / "test.pt"

    assert download_default_checkpoint("TEST", destination=target) == target.resolve()
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.download"))
    assert len(calls) == 1
    # A valid cached file is reused without touching the network.
    assert download_default_checkpoint("TEST", destination=target) == target.resolve()
    assert len(calls) == 1


def test_runtime_bundle_downloads_every_component_once(tmp_path, monkeypatch):
    payloads = {
        "test.runtime.json": b'{"format":"aetv-onnx-v1"}',
        "test.encoder.onnx": b"encoder",
        "test.decoder.onnx": b"decoder",
    }
    monkeypatch.setitem(
        codec_module.RELEASE_RUNTIME_FILES,
        "TEST",
        {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        },
    )
    calls = []

    def fake_urlopen(request, timeout):
        name = request.full_url.rsplit("/", 1)[-1].split("?", 1)[0]
        calls.append((name, timeout))
        return io.BytesIO(payloads[name])

    monkeypatch.setattr(codec_module.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "runtime"
    manifest = download_runtime_bundle("TEST", destination=target)
    assert manifest == (target / "test.runtime.json").resolve()
    assert {path.name: path.read_bytes() for path in target.iterdir()} == payloads
    assert len(calls) == 3
    assert download_runtime_bundle("TEST", destination=target) == manifest
    assert len(calls) == 3


def test_explicit_missing_checkpoint_is_never_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        codec_module,
        "download_default_checkpoint",
        lambda _mode: pytest.fail("explicit paths must not trigger a download"),
    )
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "missing.pt", mode="V8")


def test_runtime_bundle_can_be_selected_without_pytorch(tmp_path, monkeypatch):
    manifest = tmp_path / "test.runtime.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AETV_RUNTIME_MODEL", str(manifest))
    assert resolve_runtime_bundle(mode="V8") == manifest.resolve()

    result = subprocess.run(
        [sys.executable, "-c", "import sys, aetv; assert 'torch' not in sys.modules"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_station_runtime_imports_without_pytorch():
    script = """
import builtins

real_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
import aetv.station
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not WIDE_CHECKPOINT.is_file(), reason="models/v8-flex8k-ota-rxfix.pt not installed"
)
def test_v7_codec_and_modem_loopback():
    codec = AETVCodec(WIDE_CHECKPOINT, device="cpu", mode="V7")
    assert codec.mode.name == "V7"
    assert codec.mode.latents_per_gop == AETV_MODES["V7"].latents_per_gop
    rng = np.random.default_rng(0)
    frames = rng.integers(
        0,
        256,
        (codec.mode.gop_frames, codec.mode.height, codec.mode.width, 3),
        dtype=np.uint8,
    )
    latents = codec.encode_gop(frames)
    assert latents.shape == (codec.mode.latents_per_gop,)
    audio = modulate_gop_stream([latents], mode_name="V7", callsign="N0CALL")
    demod = demodulate_gop_stream(audio, band="U", drift_track="off")
    assert len(demod.gops_latents) >= 1
    recon = codec.decode_gop(demod.gops_latents[0], demod.gops_weights[0])
    assert recon.shape == frames.shape
    assert recon.dtype == np.uint8
