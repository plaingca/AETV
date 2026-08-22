"""Checkpoint load and V7 encode/decode smoke test."""

from pathlib import Path

import numpy as np
import pytest

from aetv.codec import DEFAULT_CHECKPOINT, AETVCodec
from aetv.config import AETV_MODES
from aetv.modem import demodulate_gop_stream, modulate_gop_stream

CHECKPOINT = DEFAULT_CHECKPOINT


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="models/v7-flex8k.pt not installed")
def test_v7_codec_and_modem_loopback():
    codec = AETVCodec(CHECKPOINT, device="cpu", mode="V7")
    assert codec.mode.name == "V7"
    assert codec.mode.latents_per_gop == AETV_MODES["V7"].latents_per_gop
    frames = np.random.randint(0, 256, (codec.mode.gop_frames, codec.mode.height, codec.mode.width, 3), dtype=np.uint8)
    latents = codec.encode_gop(frames)
    assert latents.shape == (codec.mode.latents_per_gop,)
    audio = modulate_gop_stream([latents], mode_name="V7", callsign="N0CALL")
    demod = demodulate_gop_stream(audio, band="U", drift_track="off")
    assert len(demod.gops_latents) >= 1
    recon = codec.decode_gop(demod.gops_latents[0], demod.gops_weights[0])
    assert recon.shape == frames.shape
    assert recon.dtype == np.uint8
