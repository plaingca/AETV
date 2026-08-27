#!/usr/bin/env python3
"""Run repeatable analog OFDM and GUI channel-emulator validations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.analog_validation import (
    aggregate_roundtrips,
    gui_channel_roundtrip,
    ideal_ofdm_roundtrip,
)


PROFILES = ("clean", "awgn12", "awgn6", "awgn0", "mpp12", "mpp6", "mpp0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["V0", "V1", "V7"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {"ideal": {}, "gui": {}}
    for mode_name in args.modes:
        ideal = [ideal_ofdm_roundtrip(mode_name, seed) for seed in range(args.seeds)]
        report["ideal"][mode_name] = aggregate_roundtrips(ideal)
        report["gui"][mode_name] = {}
        for profile in PROFILES:
            rows = [
                gui_channel_roundtrip(mode_name, profile, seed)
                for seed in range(args.seeds)
            ]
            report["gui"][mode_name][profile] = aggregate_roundtrips(rows)

    if args.json:
        print(json.dumps(report, indent=2, allow_nan=True))
        return

    print("Ideal linear real-audio OFDM")
    print("mode  decode  NMSE         correlation  gain")
    for mode_name in args.modes:
        row = report["ideal"][mode_name]
        print(
            f"{mode_name:<5} {row['decode_rate']:>6.0%}  {row['mean_nmse']:<11.3g} "
            f"{row['mean_correlation']:<11.6f}  {row['mean_gain']:.6f}"
        )

    print("\nProduction GUI path")
    print("mode profile  decode  NMSE     corr    confidence  est SNR")
    for mode_name in args.modes:
        for profile in PROFILES:
            row = report["gui"][mode_name][profile]
            print(
                f"{mode_name:<5} {profile:<8} {row['decode_rate']:>6.0%}  "
                f"{row['mean_nmse']:<8.3f} {row['mean_correlation']:<7.3f} "
                f"{row['mean_confidence']:<10.3f} "
                f"{row['mean_estimated_snr_db']:>6.1f} dB"
            )


if __name__ == "__main__":
    main()
