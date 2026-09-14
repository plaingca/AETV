"""Offline acquisition regression exercised inside every portable executable."""

import time

import numpy as np

from .config import AETV_MODES
from .hfchannel import freq_shift
from .modem import (
    blind_acquire_continuous_payload,
    demodulate_tracked_gop,
    modulate_continuous_chunks,
)


def acquisition_smoke() -> dict:
    mode = AETV_MODES["AC16"]
    fs = mode.geometry.fs
    sent = np.random.default_rng(439).standard_normal((17, mode.latents_per_gop)).astype(np.float32)
    waveform = np.concatenate(list(modulate_continuous_chunks(sent, mode.name, "N0CALL")))
    cases = []
    for offset in (-12.5, 12.5):
        started = time.perf_counter()
        audio = freq_shift(waveform, offset, fs=fs)[int(3.125 * fs):int(15.125 * fs)]
        acquired = blind_acquire_continuous_payload(audio, mode)
        result = demodulate_tracked_gop(
            audio[acquired.payload_start:acquired.payload_start + fs],
            mode, acquired.freq_offset,
        )
        received = result.gops_latents[0] * result.gops_weights[0]
        similarities = sent @ received / (np.linalg.norm(sent, axis=1) * np.linalg.norm(received))
        cosine = float(similarities.max())
        if (
            acquired.beacon.callsign != "N0CALL"
            or abs(acquired.freq_offset - offset) > 0.1
            or cosine < 0.90
        ):
            raise RuntimeError(f"AC16 late-entry acquisition failed at {offset:+g} Hz")
        cases.append({
            "offset_hz": offset, "estimated_offset_hz": acquired.freq_offset,
            "timing_metric": acquired.metric, "latent_cosine": cosine,
            "elapsed_s": time.perf_counter() - started,
        })
    return {"passed": True, "mode": mode.name, "cases": cases, "radio_opened": False}
