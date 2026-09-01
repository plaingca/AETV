#!/usr/bin/env python3
"""Run a 60-second Simpsons excerpt through analog voice + V8 channel profiles."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pystoi import stoi
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.analog_av import COMPOSITE_FS, compose_delayed_stream, extract_aetv, extract_voice, prepare_voice
from aetv.audio_metrics import AudioPerceptualLoss
from aetv.beacon import generate_beacon_chips
from aetv.codec import AETVCodec
from aetv.config import FRAMES_PER_GOP
from aetv.hfchannel import CHANNEL_PROFILES, StreamingChannelEmulator
from aetv.modem import _payload_wave, demodulate_tracked_gop
from experiment_analog_voice_aetv import write_wav


PROFILES = ("clean", "awgn12", "awgn6", "awgn0", "mpp12", "mpp6", "mpp0")


def extract_source(path: Path, duration: int) -> tuple[np.ndarray, np.ndarray]:
    frames = duration * 6
    video_command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-t", str(duration),
        "-vf", "fps=6,scale=192:108:force_original_aspect_ratio=increase,crop=192:108",
        "-frames:v", str(frames), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    video = subprocess.run(video_command, capture_output=True, timeout=max(120, duration * 4))
    expected = frames * 108 * 192 * 3
    if video.returncode or len(video.stdout) != expected:
        raise RuntimeError(video.stderr.decode(errors="replace")[-1000:])
    images = np.frombuffer(video.stdout, dtype=np.uint8).copy().reshape(frames, 108, 192, 3)

    audio_command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-t", str(duration), "-vn",
        "-ac", "1", "-ar", "8000", "-af", "aresample=8000:filter_size=64",
        "-f", "f32le", "pipe:1",
    ]
    audio = subprocess.run(audio_command, capture_output=True, timeout=max(120, duration * 4))
    samples = np.frombuffer(audio.stdout, dtype="<f4").copy()
    if audio.returncode or len(samples) < duration * 8000:
        raise RuntimeError(audio.stderr.decode(errors="replace")[-1000:])
    return images, samples[: duration * 8000]


def estimate_native_delay(reference: np.ndarray, received: np.ndarray) -> int:
    count = min(len(reference), len(received), 4 * 8000)
    correlation = signal.correlate(received[:count], reference[:count], mode="full", method="fft")
    lags = signal.correlation_lags(count, count, mode="full")
    selected = np.abs(lags) <= 256
    return int(lags[selected][np.argmax(np.abs(correlation[selected]))])


def shift_to_reference(values: np.ndarray, lag: int, length: int) -> np.ndarray:
    if lag > 0:
        aligned = values[lag:]
    elif lag < 0:
        aligned = np.pad(values, (-lag, 0))
    else:
        aligned = values
    return np.pad(aligned[:length], (0, max(0, length - len(aligned))))


def psnr(source: np.ndarray, reconstruction: np.ndarray) -> float:
    mse = np.mean((source.astype(np.float64) / 255 - reconstruction.astype(np.float64) / 255) ** 2)
    return 10 * math.log10(1 / max(mse, 1e-12))


def label_frames(panels: list[tuple[str, np.ndarray]], columns: int) -> np.ndarray:
    count, height, width, _ = panels[0][1].shape
    rows = math.ceil(len(panels) / columns)
    black = np.zeros_like(panels[0][1])
    padded = panels + [("", black)] * (rows * columns - len(panels))
    output = []
    for frame_index in range(count):
        row_images = []
        for row in range(rows):
            row_images.append(np.concatenate([
                padded[row * columns + column][1][frame_index]
                for column in range(columns)
            ], axis=1))
        image = Image.fromarray(np.concatenate(row_images, axis=0))
        draw = ImageDraw.Draw(image)
        for index, (label, _) in enumerate(padded):
            if not label:
                continue
            x = (index % columns) * width
            y = (index // columns) * height
            draw.rectangle((x, y, x + width, y + 16), fill=(0, 0, 0))
            draw.text((x + 3, y + 2), label, fill=(255, 255, 255))
        output.append(np.asarray(image))
    return np.stack(output)


def mux(path: Path, frames: np.ndarray, source_audio: np.ndarray, received_audio: np.ndarray) -> None:
    # One positive RMS gain is receiver AGC; time-varying fades, phase changes,
    # and noise remain audible instead of cancelling in a whole-clip LS fit.
    source_rms = float(np.sqrt(np.mean(source_audio.astype(np.float64) ** 2)))
    received_rms = float(np.sqrt(np.mean(received_audio.astype(np.float64) ** 2)))
    gain = float(np.clip(source_rms / max(received_rms, 1e-12), 0.1, 10.0))
    stereo = np.stack((source_audio, gain * received_audio), axis=1)
    scale = 0.95 / max(0.95, float(np.max(np.abs(stereo))))
    stereo *= scale
    with tempfile.TemporaryDirectory(prefix="aetv-simpsons-") as directory:
        raw = Path(directory) / "frames.rgb"
        wav = Path(directory) / "audio.wav"
        raw.write_bytes(frames.astype(np.uint8).tobytes())
        write_wav(wav, stereo, 8000)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "6", "-i", str(raw),
            "-i", str(wav), "-c:v", "libopenh264", "-profile:v", "constrained_baseline",
            "-pix_fmt", "yuv420p", "-b:v", "1800k", "-maxrate", "2400k",
            "-bufsize", "3600k", "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k",
            "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-shortest", str(path),
        ], check=True, timeout=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/home/plaing/SSTVAE/The Simpsons Season 31 Episode 20 - The Simpsons Full NoCuts-iex52uxH460.mp4",
    )
    parser.add_argument("--checkpoint", default="models/v8-hf3k-face-gan.pt")
    parser.add_argument("--out", default="runs/simpsons-analog-channel-60s")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.duration < 2:
        raise ValueError("duration must be at least two seconds")

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_frames, raw_audio = extract_source(Path(args.input), args.duration)
    codec = AETVCodec(args.checkpoint, device=args.device, mode="V8")
    native_waves = []
    latents = []
    for gop in range(args.duration):
        frames = source_frames[gop * 6 : (gop + 1) * 6]
        latent = codec.encode_gop(frames)
        chips = generate_beacon_chips(
            FRAMES_PER_GOP, start_frame=gop * FRAMES_PER_GOP,
            callsign="SIMPS", mode_index=codec.mode.index,
        )
        latents.append(latent)
        native_waves.append(_payload_wave(latent, chips, codec.mode, interleave=True))
    voice_gops = [raw_audio[gop * 8000 : (gop + 1) * 8000] for gop in range(args.duration)]
    reference_audio = np.concatenate([
        signal.resample_poly(prepare_voice(gop), 2, 3)[:8000] for gop in voice_gops
    ])
    composite, _, _ = compose_delayed_stream(native_waves, voice_gops)
    write_wav(out / "composite_transmit.wav", composite, COMPOSITE_FS)
    reference_native = extract_aetv(composite)

    audio_metric = AudioPerceptualLoss(si_sdr_weight=0.0)
    profile_frames: dict[str, np.ndarray] = {}
    profile_audio: dict[str, np.ndarray] = {}
    metrics = []
    for profile_index, profile in enumerate(PROFILES):
        channel = StreamingChannelEmulator(profile, seed=20260825, fs=COMPOSITE_FS)
        impaired = channel.process(composite)
        write_wav(out / f"channel_{profile}.wav", impaired, COMPOSITE_FS)
        received_native_unaligned = extract_aetv(impaired)
        lag = estimate_native_delay(reference_native, received_native_unaligned)
        received_native = shift_to_reference(received_native_unaligned, lag, len(reference_native))
        received_voice = shift_to_reference(extract_voice(impaired), lag, len(reference_native))
        recovered_audio = received_voice[8000 : (args.duration + 1) * 8000]

        reconstructions = []
        gop_psnr = []
        latent_nmse = []
        decoded = 0
        for gop in range(args.duration):
            segment = received_native[gop * 8000 : (gop + 1) * 8000]
            try:
                result = demodulate_tracked_gop(segment, codec.mode, interleave=True)
                recovered_latent = result.gops_latents[0]
                weights = result.gops_weights[0]
                decoded += 1
            except Exception:
                recovered_latent = np.zeros_like(latents[gop])
                weights = np.zeros_like(latents[gop])
            reconstruction = codec.decode_gop(recovered_latent, weights)
            reconstructions.append(reconstruction)
            source = source_frames[gop * 6 : (gop + 1) * 6]
            gop_psnr.append(psnr(source, reconstruction))
            error = recovered_latent.astype(np.float64) - latents[gop].astype(np.float64)
            latent_nmse.append(float(np.mean(error**2) / np.mean(latents[gop].astype(np.float64) ** 2)))
        reconstructed_frames = np.concatenate(reconstructions)
        profile_frames[profile] = reconstructed_frames
        profile_audio[profile] = recovered_audio

        pred = torch.from_numpy(recovered_audio[None]).float()
        target = torch.from_numpy(reference_audio[None]).float()
        components = audio_metric.components(pred, target)
        row = {
            "profile": profile,
            "label": CHANNEL_PROFILES[profile].label,
            "alignment_lag_native_samples": lag,
            "decoded_gops": decoded,
            "gops": args.duration,
            "mean_video_psnr_db": float(np.mean(gop_psnr)),
            "p10_video_psnr_db": float(np.percentile(gop_psnr, 10)),
            "mean_latent_nmse": float(np.mean(latent_nmse)),
            "audio_si_sdr_db": float(-components["si_sdr"]),
            "audio_mel": float(components["mel"]),
            "audio_stoi": float(stoi(reference_audio, recovered_audio, 8000, extended=False)),
        }
        metrics.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        comparison = label_frames([
            ("source | audio L", source_frames),
            (f"{profile} | recovered audio R", reconstructed_frames),
        ], columns=2)
        mux(out / f"simpsons_{args.duration}s_{profile}.mp4", comparison, reference_audio, recovered_audio)

    grid_panels = [("source", source_frames)] + [
        (profile, profile_frames[profile]) for profile in PROFILES
    ]
    grid = label_frames(grid_panels, columns=4)
    mux(
        out / f"simpsons_{args.duration}s_channel_grid.mp4", grid,
        reference_audio, profile_audio["mpp6"],
    )
    Image.fromarray(grid[len(grid) // 2]).save(out / "channel_grid_midpoint.png")
    report = {
        "input": str(Path(args.input).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "duration_seconds": args.duration,
        "audio_delay_seconds": 1.0,
        "grid_audio": {"left": "source", "right": "mpp6 recovered"},
        "profiles": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
