#!/usr/bin/env python3
"""Paired latent-SNR A/B of receiver channel-estimation arms on simulated channels.

Every arm demodulates the same impaired waveform, so per-seed differences are
exact pairs. Latents are random tanh values, not encoder output: this measures
the modem's effective SNR (latents x weights against the sent latents), not
PSNR.

Arms: ``pilot`` (v0.1.23 estimator, no placement), ``lmmse`` (LMMSE only),
``place`` (placement only), ``both`` (the shipped receiver).

Profiles are ``<kind><snr>``: ``awgn``, ``mpp``, ``mpd``, ``ota40m`` (Butterworth
taps, as in ``emulate``) and ``mppg`` (Gaussian taps). ``--stream`` sends one
continuous transmission through the live ``StreamingDemodulator`` with SDR
settings, so channel statistics accumulate across GOPs.

    python scripts/eval_receiver_ab.py --modes V8 --profiles mpp12,awgn0 --seeds 48 \\
        --out runs/receiver-ab/v8.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv import modem  # noqa: E402
from aetv.beacon import generate_beacon_chips  # noqa: E402
from aetv.config import AETV_MODES, FRAMES_PER_GOP  # noqa: E402
from aetv.hfchannel import ChannelProfile, emulate  # noqa: E402

ARMS = {"pilot": ("pilot", False), "lmmse": ("lmmse", False), "place": ("pilot", True), "both": ("lmmse", True)}
KINDS = {"awgn": "none", "mpp": "mpp", "mpd": "mpd", "ota40m": "ota40m", "mppg": "mpp-gauss"}


def profile(name: str) -> ChannelProfile:
    match = re.fullmatch(r"(awgn|mppg|mpp|mpd|ota40m)(m?)(\d+(?:\.\d+)?)", name)
    if not match:
        raise ValueError(f"bad profile {name!r}")
    kind, negative, snr = match.groups()
    return ChannelProfile(name, name, -float(snr) if negative else float(snr), KINDS[kind])


SNR_FLOOR_DB = -20.0


def effective_snr(x: np.ndarray, y: np.ndarray) -> float:
    """rho^2/(1-rho^2) in dB, floored so two lost GOPs compare as a tie, not as noise."""
    if not np.all(np.isfinite(y)) or np.std(y) == 0:
        return SNR_FLOOR_DB
    rho2 = min(float(np.corrcoef(x, y)[0, 1]) ** 2, 0.999999)
    return float(max(10 * np.log10(max(rho2, 1e-12) / (1 - rho2)), SNR_FLOOR_DB))


def transmission(mode_name: str, seed: int, gops: int, stream: bool):
    mode = AETV_MODES[mode_name]
    rng = np.random.default_rng(10_000 + seed)
    sent = []
    for _ in range(gops):
        x = np.tanh(rng.normal(size=mode.geometry.latents_per_gop))
        sent.append((x / np.sqrt(np.mean(x**2))).astype(np.float32))
    if stream:
        audio = np.concatenate(list(modem.modulate_continuous_chunks(sent, mode_name, "N0CALL", total_gops=gops)))
    else:
        audio = modem.modulate_gop_stream(sent, mode_name=mode_name, callsign="N0CALL")
    return sent, audio


def demodulate(mode_name: str, rx: np.ndarray, gops: int, stream: bool):
    mode = AETV_MODES[mode_name]
    if not stream:
        d = modem.demodulate_gop_stream(rx, band=mode.band, drift_track="off")
        chips = d.beacon_repeated_chips if mode.band == "U" else d.beacon_chips
        return d.gops_latents, d.gops_weights, chips, [d.beacon is not None and d.beacon.callsign == "N0CALL"], [d.window_shift], d.lmmse_gops
    receiver = modem.StreamingDemodulator(mode.band, continuous=True, mode_name=mode_name, boundary_tracking=True)
    results = []
    block = mode.geometry.fs // 10
    tail = np.zeros(2 * mode.geometry.fs, dtype=np.float32)
    padded = np.concatenate([rx, tail])
    for start in range(0, len(padded), block):
        results.extend(receiver.feed(padded[start : start + block]))
    latents = [r.gops_latents[0] for r in results]
    weights = [r.gops_weights[0] for r in results]
    chips = np.concatenate([r.beacon_repeated_chips if mode.band == "U" else r.beacon_chips
                            for r in results]) if results else np.zeros(0)
    verified = [r.stream_frame_counter is not None for r in results]
    return latents, weights, chips, verified, [r.window_shift for r in results], sum(r.lmmse_gops for r in results)


def run(job):
    mode_name, prof, seed, gops, stream, arms = job
    mode = AETV_MODES[mode_name]
    sent, audio = transmission(mode_name, seed, gops, stream)
    rx = emulate(audio, profile(prof), seed=seed, fs=mode.geometry.fs)
    chips_sent = np.asarray(generate_beacon_chips(n_frames=gops * FRAMES_PER_GOP, start_frame=0,
                                                  callsign="N0CALL", mode_index=mode.index), dtype=float)
    out = {}
    for arm in arms:
        modem.CHANNEL_ESTIMATOR, modem.WINDOW_PLACEMENT = ARMS[arm]
        try:
            latents, weights, chips, verified, shifts, lmmse_gops = demodulate(mode_name, rx, gops, stream)
        except Exception:  # noqa: BLE001 - an acquisition failure drops the seed for every arm
            return None
        complete = len(latents) == gops
        if not stream and not complete:
            return None
        n = min(len(chips), len(chips_sent))
        row = dict(
            delivered=len(latents) / gops,
            verified=float(np.sum(verified)) / gops,
            beacon_snr=effective_snr(chips_sent[:n], np.asarray(chips[:n], float)) if complete and n > 8 else float("nan"),
            shift=float(np.mean(np.abs(shifts))) if shifts else 0.0,
            moved=float(np.mean(np.asarray(shifts) != 0)) if shifts else 0.0,
            lmmse=lmmse_gops / gops,
            snr=float("nan"),
            snr_gop_mean=float("nan"),
        )
        if complete:
            x = np.concatenate(sent)
            lat, w = np.concatenate(latents), np.concatenate(weights)
            row["snr"] = effective_snr(x, lat * w)
            row["snr_gop_mean"] = float(np.mean([effective_snr(s, l * ww) for s, l, ww in zip(sent, latents, weights)]))
        out[arm] = row
    return out


def summarize(rows: list[dict], arms: list[str]) -> dict:
    cell = {"n": len(rows)}
    if not rows:
        return cell
    for arm in arms:
        cell[arm] = {k: float(np.nanmean([r[arm][k] for r in rows])) for k in rows[0][arm]}
    base = arms[0]
    for arm in arms[1:]:
        for key in ("snr", "snr_gop_mean", "beacon_snr", "verified", "delivered"):
            d = np.array([r[arm][key] - r[base][key] for r in rows])
            d = d[np.isfinite(d)]
            if len(d) > 1:
                cell[f"d_{key}/{arm}"] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), float(np.mean(d > 0))]
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", default="V8,V9,V7")
    parser.add_argument("--profiles", default="awgn12,awgn6,awgn0,mpp12,mpp6,mpp3,mpp0,ota40m3")
    parser.add_argument("--arms", default="pilot,lmmse,place,both")
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--gops", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arms = args.arms.split(",")
    jobs = [(m, p, s, args.gops, args.stream, arms)
            for m in args.modes.split(",") for p in args.profiles.split(",")
            for s in range(args.seed0, args.seed0 + args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, jobs, chunksize=1))
    cells: dict[str, list] = {}
    for job, result in zip(jobs, results):
        if result is not None:
            cells.setdefault(f"{job[0]}/{job[1]}", []).append(result)
    summary = {}
    for m in args.modes.split(","):
        for p in args.profiles.split(","):
            key = f"{m}/{p}"
            summary[key] = summarize(cells.get(key, []), arms)
            c = summary[key]
            parts = [f"{key:14s} n={c['n']:3d}"]
            if c["n"]:
                parts.append(f"base {c[arms[0]]['snr']:6.2f}")
                for arm in arms[1:]:
                    d = c.get(f"d_snr/{arm}")
                    if d:
                        parts.append(f"{arm} {d[0]:+.2f}±{d[1]:.2f}({100 * d[2]:.0f}%)")
                parts.append("delivered " + " ".join(f"{c[a]['delivered']:.3f}" for a in arms))
                parts.append("verified " + " ".join(f"{c[a]['verified']:.3f}" for a in arms))
                parts.append("moved " + " ".join(f"{c[a]['moved']:.2f}" for a in arms))
                parts.append("lmmse " + " ".join(f"{c[a]['lmmse']:.2f}" for a in arms))
            print(" ".join(parts), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "per_seed": cells}, indent=1))


if __name__ == "__main__":
    main()
