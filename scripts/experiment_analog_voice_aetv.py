#!/usr/bin/env python3
"""Render a delayed analog-voice plus upper-slice V8 AETV experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pystoi import stoi
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.analog_av import (
    AETV_HIGH_HZ,
    AETV_FILTER_HIGH_HZ,
    AETV_FILTER_LOW_HZ,
    AETV_LOW_HZ,
    COMPOSITE_FS,
    VOICE_HIGH_HZ,
    compose_delayed_stream,
    extract_aetv,
    extract_voice,
    prepare_voice,
)
from aetv.audio_metrics import AudioPerceptualLoss
from aetv.beacon import generate_beacon_chips
from aetv.codec import AETVCodec
from aetv.config import FRAMES_PER_GOP
from aetv.modem import _payload_wave, demodulate_tracked_gop


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(audio)
    pcm = np.clip(values, -1, 1)
    pcm = np.rint(pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def labeled_comparison(source: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    frames = np.concatenate((source, reconstruction), axis=2)
    labeled = []
    width = source.shape[2]
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 2 * width, 17), fill=(0, 0, 0))
        draw.text((4, 2), "source", fill=(255, 255, 255))
        draw.text((width + 4, 2), "V8 over upper AETV slice", fill=(255, 255, 255))
        labeled.append(np.asarray(image))
    return np.stack(labeled)


def mux_comparison(
    path: Path,
    source_video: np.ndarray,
    reconstruction_video: np.ndarray,
    source_audio: np.ndarray,
    reconstruction_audio: np.ndarray,
) -> None:
    frames = labeled_comparison(source_video, reconstruction_video)
    stereo = np.stack((source_audio, reconstruction_audio), axis=1)
    with tempfile.TemporaryDirectory(prefix="aetv-analog-render-") as directory:
        raw_path = Path(directory) / "frames.rgb"
        wav_path = Path(directory) / "audio.wav"
        raw_path.write_bytes(frames.astype(np.uint8).tobytes())
        write_wav(wav_path, stereo, 8000)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "6", "-i", str(raw_path),
            "-i", str(wav_path), "-c:v", "libopenh264", "-profile:v", "constrained_baseline",
            "-pix_fmt", "yuv420p", "-b:v", "1000k", "-maxrate", "1400k",
            "-bufsize", "2000k", "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k",
            "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-shortest", str(path),
        ], check=True)


def psnr(source: np.ndarray, reconstruction: np.ndarray) -> float:
    error = np.mean((source.astype(np.float64) / 255 - reconstruction.astype(np.float64) / 255) ** 2)
    return 10 * math.log10(1 / max(error, 1e-12))


def band_power(values: np.ndarray, low: float, high: float) -> float:
    frequency, density = signal.welch(values, fs=COMPOSITE_FS, nperseg=4096)
    selected = (frequency >= low) & (frequency < high)
    return float(np.trapezoid(density[selected], frequency[selected]))


def plot_spectrum(path: Path, composite: np.ndarray) -> None:
    frequency, density = signal.welch(composite, fs=COMPOSITE_FS, nperseg=4096)
    db = 10 * np.log10(np.maximum(density, 1e-14))
    width, height = 1200, 520
    left, right, top, bottom = 72, 24, 48, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image, "RGBA")

    def x_at(hz: float) -> int:
        return int(left + plot_width * hz / (COMPOSITE_FS / 2))

    draw.rectangle((x_at(0), top, x_at(VOICE_HIGH_HZ), top + plot_height), fill=(76, 120, 168, 35))
    draw.rectangle((x_at(VOICE_HIGH_HZ), top, x_at(AETV_FILTER_LOW_HZ), top + plot_height), fill=(120, 120, 120, 35))
    draw.rectangle((x_at(AETV_LOW_HZ), top, x_at(AETV_HIGH_HZ), top + plot_height), fill=(245, 133, 24, 35))
    floor, ceiling = float(db.max() - 90), float(db.max() + 3)
    for hz in range(0, 6001, 1000):
        x = x_at(hz)
        draw.line((x, top, x, top + plot_height), fill=(170, 170, 170, 100))
        draw.text((x - 12, top + plot_height + 8), str(hz), fill=(20, 20, 20, 255))
    points = []
    for hz, value in zip(frequency, db):
        x = x_at(float(hz))
        y = int(top + plot_height * (ceiling - float(value)) / (ceiling - floor))
        points.append((x, max(top, min(top + plot_height, y))))
    draw.line(points, fill=(25, 25, 25, 255), width=2)
    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=(40, 40, 40, 255), width=1)
    draw.text((left, 14), "0-2.2 kHz analog voice | 300 Hz guard | 2.6-4.8 kHz V8/W carriers", fill=(10, 10, 10, 255))
    draw.text((left + plot_width // 2 - 70, height - 24), "frequency (Hz)", fill=(20, 20, 20, 255))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/v8-hf3k-face-gan.pt")
    parser.add_argument("--cache-dir", default="data/v9-full-epoch-cache")
    parser.add_argument("--out", default="runs/analog-voice-v8-upper")
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.cache_dir).glob("clip_*.pt"))[: args.clips]
    if len(files) != args.clips:
        raise RuntimeError(f"needed {args.clips} cached AV clips, found {len(files)}")

    codec = AETVCodec(args.checkpoint, device=args.device, mode="V8")
    source_video: list[np.ndarray] = []
    source_audio: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    native_waves: list[np.ndarray] = []
    for index, path in enumerate(files):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        video = saved["video"].permute(1, 2, 3, 0).numpy().astype(np.uint8)
        audio = saved["audio"].float().div(32767).numpy()
        latent = codec.encode_gop(video)
        chips = generate_beacon_chips(
            FRAMES_PER_GOP, start_frame=index * FRAMES_PER_GOP,
            callsign="ANALOG", mode_index=codec.mode.index,
        )
        source_video.append(video)
        source_audio.append(audio)
        latents.append(latent)
        native_waves.append(_payload_wave(latent, chips, codec.mode, interleave=True))

    composite, speech_component, aetv_component = compose_delayed_stream(
        native_waves, source_audio
    )
    write_wav(out / "composite_transmit.wav", composite, COMPOSITE_FS)
    write_wav(out / "voice_component_delayed.wav", speech_component, COMPOSITE_FS)
    write_wav(out / "aetv_component_upper.wav", aetv_component, COMPOSITE_FS)
    plot_spectrum(out / "composite_spectrum.png", composite)

    recovered_native = extract_aetv(composite)
    recovered_voice = extract_voice(composite)
    audio_metric = AudioPerceptualLoss(si_sdr_weight=0.0)
    rows = []
    rendered = []
    for index in range(args.clips):
        native_segment = recovered_native[index * 8000 : (index + 1) * 8000]
        result = demodulate_tracked_gop(native_segment, codec.mode, interleave=True)
        recovered_latent = result.gops_latents[0]
        weights = result.gops_weights[0]
        reconstruction = codec.decode_gop(recovered_latent, weights)
        clean_reconstruction = codec.decode_gop(
            latents[index], np.ones_like(latents[index], dtype=np.float32)
        )

        # Voice GOP i is intentionally carried during composite interval i+1.
        recovered_audio = recovered_voice[(index + 1) * 8000 : (index + 2) * 8000]
        reference_audio = signal.resample_poly(prepare_voice(source_audio[index]), 2, 3)[:8000]
        pred_tensor = torch.from_numpy(recovered_audio[None]).float()
        ref_tensor = torch.from_numpy(reference_audio[None]).float()
        components = audio_metric.components(pred_tensor, ref_tensor)
        latent_error = recovered_latent.astype(np.float64) - latents[index].astype(np.float64)
        clean_psnr = psnr(source_video[index], clean_reconstruction)
        composite_psnr = psnr(source_video[index], reconstruction)
        row = {
            "clip": index,
            "source": str(files[index]),
            "video_clean_psnr_db": clean_psnr,
            "video_composite_psnr_db": composite_psnr,
            "video_transport_psnr_delta_db": composite_psnr - clean_psnr,
            "latent_nmse": float(np.mean(latent_error**2) / np.mean(latents[index].astype(np.float64) ** 2)),
            "latent_correlation": float(np.corrcoef(latents[index], recovered_latent)[0, 1]),
            "pilot_snr_db": float(result.snr_db),
            "audio_si_sdr_db": float(-components["si_sdr"]),
            "audio_mel": float(components["mel"]),
            "audio_stoi": float(stoi(reference_audio, recovered_audio, 8000, extended=False)),
        }
        rows.append(row)
        render = out / f"sample_{index:02d}_comparison.mp4"
        mux_comparison(
            render, source_video[index], reconstruction, reference_audio, recovered_audio
        )
        rendered.append(render)
        print(json.dumps(row, sort_keys=True), flush=True)

    concat_file = out / "comparison_inputs.txt"
    concat_file.write_text("".join(f"file '{path.resolve()}'\n" for path in rendered))
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", "-movflags", "+faststart", str(out / "all_samples_comparison.mp4"),
    ], check=True)
    concat_file.unlink()

    active = composite[COMPOSITE_FS : -COMPOSITE_FS]
    voice_power = band_power(active, 0, VOICE_HIGH_HZ)
    guard_power = band_power(active, VOICE_HIGH_HZ, AETV_FILTER_LOW_HZ)
    upper_power = band_power(active, AETV_FILTER_LOW_HZ, AETV_FILTER_HIGH_HZ + 1)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "composite_sample_rate": COMPOSITE_FS,
        "voice_band_hz": [0, int(VOICE_HIGH_HZ)],
        "guard_band_hz": [int(VOICE_HIGH_HZ), int(AETV_FILTER_LOW_HZ)],
        "aetv_band_hz": [int(AETV_LOW_HZ), int(AETV_HIGH_HZ)],
        "aetv_filter_band_hz": [int(AETV_FILTER_LOW_HZ), int(AETV_FILTER_HIGH_HZ)],
        "audio_delay_samples": COMPOSITE_FS,
        "audio_delay_seconds": 1.0,
        "guard_to_voice_db": 10 * math.log10(max(guard_power, 1e-20) / max(voice_power, 1e-20)),
        "guard_to_aetv_db": 10 * math.log10(max(guard_power, 1e-20) / max(upper_power, 1e-20)),
        "clips": rows,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
