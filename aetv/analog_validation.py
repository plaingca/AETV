"""Analog-domain validation helpers for the production AETV modem path.

These helpers deliberately distinguish the linear OFDM transport from the
production transmit conditioner.  The former answers whether real passband
audio can preserve every latent value on an ideal channel; the latter measures
the clipping/filtering distortion that the GUI actually transmits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .beacon import generate_beacon_chips
from .config import AETV_MODES, FRAMES_PER_GOP
from .hfchannel import StreamingChannelEmulator
from .modem import (
    StreamingDemodulator,
    _payload_wave,
    demodulate_tracked_gop,
    modulate_continuous_chunks,
)


@dataclass(frozen=True)
class AnalogRoundTrip:
    mode: str
    profile: str
    seed: int
    decoded: bool
    nmse: float
    correlation: float
    gain: float
    bias: float
    mean_confidence: float
    estimated_snr_db: float
    damage_confidence_correlation: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def unit_rms_latents(mode_name: str, seed: int) -> np.ndarray:
    """Return a deterministic, representative analog latent vector."""
    mode = AETV_MODES[mode_name]
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(mode.latents_per_gop).astype(np.float32)
    return values / np.sqrt(np.mean(values.astype(np.float64) ** 2))


def _metrics(
    original: np.ndarray,
    recovered: np.ndarray,
    weights: np.ndarray,
    *,
    mode_name: str,
    profile: str,
    seed: int,
    estimated_snr_db: float,
) -> AnalogRoundTrip:
    original64 = np.asarray(original, dtype=np.float64)
    recovered64 = np.asarray(recovered, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    error = recovered64 - original64
    signal_power = max(float(np.mean(original64**2)), 1e-18)
    damage = error**2
    uncertainty = 1.0 - weights64
    if np.std(damage) > 0.0 and np.std(uncertainty) > 0.0:
        damage_confidence_correlation = float(
            np.corrcoef(damage, uncertainty)[0, 1]
        )
    else:
        damage_confidence_correlation = 0.0
    return AnalogRoundTrip(
        mode=mode_name,
        profile=profile,
        seed=seed,
        decoded=True,
        nmse=float(np.mean(error**2) / signal_power),
        correlation=float(np.corrcoef(original64, recovered64)[0, 1]),
        gain=float(np.dot(original64, recovered64) / np.dot(original64, original64)),
        bias=float(np.mean(error)),
        mean_confidence=float(np.mean(weights64)),
        estimated_snr_db=float(estimated_snr_db),
        damage_confidence_correlation=damage_confidence_correlation,
    )


def ideal_ofdm_roundtrip(mode_name: str, seed: int = 0) -> AnalogRoundTrip:
    """Exercise real passband OFDM without TX conditioning or impairments."""
    mode = AETV_MODES[mode_name]
    original = unit_rms_latents(mode_name, seed)
    chips = generate_beacon_chips(
        n_frames=FRAMES_PER_GOP,
        callsign="N0CALL",
        mode_index=mode.index,
    )
    audio = _payload_wave(original, chips, mode, interleave=True)
    result = demodulate_tracked_gop(audio, mode, interleave=True)
    return _metrics(
        original,
        result.gops_latents[0],
        result.gops_weights[0],
        mode_name=mode_name,
        profile="ideal-linear",
        seed=seed,
        estimated_snr_db=result.snr_db,
    )


def gui_channel_roundtrip(
    mode_name: str,
    profile: str,
    seed: int = 0,
    *,
    tx_level: float = 0.85,
) -> AnalogRoundTrip:
    """Exercise the same TX, channel, level, and streaming RX path as the GUI."""
    mode = AETV_MODES[mode_name]
    original = unit_rms_latents(mode_name, seed + 10_000)
    channel = StreamingChannelEmulator(profile, seed=seed, fs=mode.geometry.fs)
    receiver = StreamingDemodulator(
        mode.band,
        continuous=True,
        mode_name=mode_name,
    )
    results = []
    block_samples = max(1, mode.geometry.fs // 10)
    for clean in modulate_continuous_chunks([original], mode_name, "N0CALL"):
        impaired = channel.process(clean)
        peak = float(np.max(np.abs(impaired))) if impaired.size else 0.0
        if peak > 0.0:
            impaired = impaired * (tx_level / peak)
        for start in range(0, len(impaired), block_samples):
            results.extend(receiver.feed(impaired[start : start + block_samples]))
    if not results:
        return AnalogRoundTrip(
            mode=mode_name,
            profile=profile,
            seed=seed,
            decoded=False,
            nmse=float("inf"),
            correlation=0.0,
            gain=0.0,
            bias=float("nan"),
            mean_confidence=0.0,
            estimated_snr_db=float("nan"),
            damage_confidence_correlation=0.0,
        )
    result = results[0]
    return _metrics(
        original,
        result.gops_latents[0],
        result.gops_weights[0],
        mode_name=mode_name,
        profile=profile,
        seed=seed,
        estimated_snr_db=result.snr_db,
    )


def aggregate_roundtrips(rows: list[AnalogRoundTrip]) -> dict[str, float]:
    """Aggregate successful deterministic trials for a report or assertion."""
    decoded = [row for row in rows if row.decoded]
    count = len(rows)
    if not decoded:
        return {
            "trials": float(count),
            "decoded": 0.0,
            "decode_rate": 0.0,
            "mean_nmse": float("inf"),
            "mean_correlation": 0.0,
            "mean_gain": 0.0,
            "mean_confidence": 0.0,
            "mean_estimated_snr_db": float("nan"),
            "mean_damage_confidence_correlation": 0.0,
        }
    return {
        "trials": float(count),
        "decoded": float(len(decoded)),
        "decode_rate": len(decoded) / max(1, count),
        "mean_nmse": float(np.mean([row.nmse for row in decoded])),
        "mean_correlation": float(np.mean([row.correlation for row in decoded])),
        "mean_gain": float(np.mean([row.gain for row in decoded])),
        "mean_confidence": float(np.mean([row.mean_confidence for row in decoded])),
        "mean_estimated_snr_db": float(
            np.mean([row.estimated_snr_db for row in decoded])
        ),
        "mean_damage_confidence_correlation": float(
            np.mean([row.damage_confidence_correlation for row in decoded])
        ),
    }
