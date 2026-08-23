#!/usr/bin/env python3
"""Analyze an AETV TX WAV against a raw KiwiSDR IQ debug capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

from aetv.config import AETV_MODES
from aetv.kiwi import IqToPassband
from aetv.modem import StreamingDemodulator


def _float_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, values = wavfile.read(path)
    if np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.float32) / abs(float(np.iinfo(values.dtype).min))
    return int(rate), np.asarray(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx", required=True, type=Path, help="*.tx.wav from AETV debug")
    parser.add_argument("--iq", required=True, type=Path, help="*.iq.wav from AETV debug")
    parser.add_argument("--mode", default="V7")
    parser.add_argument(
        "--framing",
        choices=("continuous", "independent"),
        default="continuous",
        help="live continuous framing, or legacy per-GOP acquisition",
    )
    parser.add_argument("--dial-mhz", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    mode = AETV_MODES[args.mode]
    metadata_path = args.iq.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dial_mhz = float(args.dial_mhz or metadata["dial_mhz"])
    exact_rate = float(metadata.get("kiwi_sample_rate_exact", 0.0))

    tx_rate, tx = _float_wav(args.tx)
    iq_rate, raw = _float_wav(args.iq)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise SystemExit("expected a two-channel Kiwi IQ WAV")
    if tx_rate != mode.geometry.fs:
        raise SystemExit(f"TX rate {tx_rate} does not match {args.mode} rate {mode.geometry.fs}")
    exact_rate = exact_rate or float(iq_rate)
    iq = raw[:, 0] + 1j * raw[:, 1]
    converter = IqToPassband(
        exact_rate,
        mode.geometry.fs,
        offset_hz=-float(mode.geometry.fcenter_hz),
    )
    audio_chunks = [converter(iq[start : start + 512]) for start in range(0, len(iq), 512)]
    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, np.float32)

    probe_len = min(len(tx), tx_rate)
    peak_score = 0.0
    peak_sample = 0
    if probe_len and len(audio) >= probe_len:
        probe = tx[:probe_len]
        corr = correlate(audio, probe, mode="valid", method="fft")
        cumulative = np.pad(np.cumsum(audio**2, dtype=np.float64), (1, 0))
        energy = cumulative[probe_len:] - cumulative[:-probe_len]
        denom = np.sqrt(np.maximum(energy * np.sum(probe**2), 1e-30))
        scores = np.abs(corr) / denom
        peak_sample = int(np.argmax(scores))
        peak_score = float(scores[peak_sample])

    events: list[dict] = []
    receiver = StreamingDemodulator(
        mode.band,
        on_debug=events.append,
        continuous=args.framing == "continuous",
        mode_name=mode.name,
    )
    decoded = []
    for start in range(0, len(audio), 4096):
        decoded.extend(receiver.feed(audio[start : start + 4096]))

    report = {
        "tx": str(args.tx.resolve()),
        "iq": str(args.iq.resolve()),
        "mode": mode.name,
        "framing": args.framing,
        "dial_mhz": dial_mhz,
        "kiwi_sample_rate_exact": exact_rate,
        "iq_duration_s": len(iq) / exact_rate,
        "recovered_audio_duration_s": len(audio) / mode.geometry.fs,
        "tx_correlation_peak": peak_score,
        "tx_correlation_offset_s": peak_sample / mode.geometry.fs,
        "accepted_gops": len(decoded),
        "callsign": next((item.callsign for item in reversed(decoded) if item.callsign), ""),
        "discontinuities": metadata.get("discontinuities", []),
        "events": events,
    }
    output = args.out or args.iq.with_name(args.iq.stem.removesuffix(".iq") + ".analysis.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "events"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
