#!/usr/bin/env python3
"""Render source, decoded video, and synchronized RX FFT waterfall per channel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv.analog_av import COMPOSITE_FS, extract_aetv, extract_voice, prepare_voice
from aetv.codec import AETVCodec
from aetv.hfchannel import CHANNEL_PROFILES, StreamingChannelEmulator
from aetv.modem import demodulate_tracked_gop
from experiment_simpsons_analog_channel import (
    estimate_native_delay,
    extract_source,
    mux,
    shift_to_reference,
)


DEFAULT_PROFILES = ("clean", "awgn12", "awgn6", "awgn0", "mpp12", "mpp6", "mpp0")
WATERFALL_WIDTH = 320
HISTORY_SECONDS = 3.0
FFT_MIN_DB = -60.0
FFT_MAX_DB = 10.0


def cool_waterfall_palette() -> np.ndarray:
    """Dependency-free black-blue-cyan-green-yellow waterfall palette."""
    anchors = np.asarray([
        (0, 0, 0),
        (7, 17, 54),
        (8, 50, 120),
        (0, 112, 170),
        (0, 178, 181),
        (42, 205, 118),
        (154, 221, 55),
        (244, 226, 52),
        (255, 255, 235),
    ], dtype=np.float64)
    position = np.linspace(0.0, 255.0, len(anchors))
    samples = np.arange(256, dtype=np.float64)
    return np.stack([
        np.interp(samples, position, anchors[:, channel]) for channel in range(3)
    ], axis=1).astype(np.uint8)


WATERFALL_PALETTE = cool_waterfall_palette()


def read_wav_float(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, values = wavfile.read(path)
    if values.ndim != 1:
        raise ValueError("transmit waveform must be mono")
    if np.issubdtype(values.dtype, np.integer):
        scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        values = values.astype(np.float64) / scale
    else:
        values = values.astype(np.float64)
    return int(sample_rate), values


def decode_received(
    impaired: np.ndarray,
    *,
    reference_native: np.ndarray,
    codec: AETVCodec,
    duration: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    native_unaligned = extract_aetv(impaired)
    lag = estimate_native_delay(reference_native, native_unaligned)
    native = shift_to_reference(native_unaligned, lag, len(reference_native))
    voice = shift_to_reference(extract_voice(impaired), lag, len(reference_native))
    recovered_audio = voice[8_000 : (duration + 1) * 8_000]
    frames = []
    latent_shape = (codec.mode.geometry.latents_per_gop,)
    for index in range(duration):
        segment = native[index * 8_000 : (index + 1) * 8_000]
        try:
            result = demodulate_tracked_gop(segment, codec.mode, interleave=True)
            latent = result.gops_latents[0]
            weights = result.gops_weights[0]
        except Exception:
            latent = np.zeros(latent_shape, dtype=np.float32)
            weights = np.zeros(latent_shape, dtype=np.float32)
        frames.append(codec.decode_gop(latent, weights))
    return np.concatenate(frames), recovered_audio, lag


def waterfall_spectra(
    received: np.ndarray,
    *,
    reference_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    frequency, times, spectrum = signal.stft(
        np.asarray(received, dtype=np.float64),
        fs=COMPOSITE_FS,
        window="hann",
        nperseg=512,
        noverlap=112,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    relative_db = 20.0 * np.log10(
        np.maximum(np.abs(spectrum), 1e-12) / max(reference_rms, 1e-12)
    )
    return times, relative_db.T


def waterfall_panel(
    history: np.ndarray,
    frequency_bins: int,
    *,
    profile: str,
) -> np.ndarray:
    panel_height = 108
    title_height = 17
    axis_height = 17
    plot_height = panel_height - title_height - axis_height
    normalized = np.clip(
        (history - FFT_MIN_DB) / (FFT_MAX_DB - FFT_MIN_DB), 0.0, 1.0
    )
    indices = np.rint(normalized * 255.0).astype(np.uint8)
    rgb = WATERFALL_PALETTE[indices]
    plot = Image.fromarray(rgb).resize(
        (WATERFALL_WIDTH, plot_height), Image.Resampling.BILINEAR
    )
    panel = Image.new("RGB", (WATERFALL_WIDTH, panel_height), (0, 0, 0))
    panel.paste(plot, (0, title_height))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.text((4, 2), f"{profile} RX FFT | -60..+10 dBr | 3 s history", fill=(255, 255, 255, 255))

    for hz, color in (
        (2_200, (100, 190, 255, 230)),
        (2_500, (255, 255, 255, 190)),
        (2_600, (255, 190, 90, 220)),
        (4_900, (255, 190, 90, 220)),
    ):
        x = int(round(hz / 6_000.0 * (WATERFALL_WIDTH - 1)))
        draw.line((x, title_height, x, title_height + plot_height - 1), fill=color, width=1)
    draw.text((2, title_height + 1), "-3s", fill=(255, 255, 255, 220))
    draw.text((2, title_height + plot_height - 12), "now", fill=(255, 255, 255, 220))

    axis_y = title_height + plot_height
    draw.rectangle((0, axis_y, WATERFALL_WIDTH, panel_height), fill=(0, 0, 0, 255))
    for hz, label, offset in (
        (0, "0", 2),
        (2_200, "2.2", -20),
        (2_500, "2.5", 2),
        (4_900, "4.9", -10),
        (6_000, "6k", -16),
    ):
        x = int(round(hz / 6_000.0 * (WATERFALL_WIDTH - 1)))
        anchor = max(0, min(WATERFALL_WIDTH - 18, x + offset))
        draw.text((anchor, axis_y + 2), label, fill=(230, 230, 230, 255))
    draw.text((34, axis_y + 2), "VOICE", fill=(100, 190, 255, 255))
    draw.text((124, axis_y + 2), "G", fill=(230, 230, 230, 255))
    draw.text((176, axis_y + 2), "AETV", fill=(255, 190, 90, 255))
    return np.asarray(panel)


def render_frames(
    source: np.ndarray,
    decoded: np.ndarray,
    *,
    received: np.ndarray,
    reference_rms: float,
    profile: str,
) -> np.ndarray:
    times, spectra = waterfall_spectra(received, reference_rms=reference_rms)
    rows_per_second = COMPOSITE_FS / (512 - 112)
    history_rows = int(round(HISTORY_SECONDS * rows_per_second))
    panels = []
    for frame_index, (source_frame, decoded_frame) in enumerate(zip(source, decoded)):
        current_time = (frame_index + 0.5) / 6.0
        end = int(np.searchsorted(times, current_time, side="right"))
        start = max(0, end - history_rows)
        history = spectra[start:end]
        if len(history) < history_rows:
            history = np.pad(
                history,
                ((history_rows - len(history), 0), (0, 0)),
                constant_values=FFT_MIN_DB,
            )
        waterfall = waterfall_panel(history, spectra.shape[1], profile=profile)

        source_image = Image.fromarray(source_frame)
        decoded_image = Image.fromarray(decoded_frame)
        for image, label in ((source_image, "source"), (decoded_image, "decoded")):
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 16), fill=(0, 0, 0))
            draw.text((4, 2), label, fill=(255, 255, 255))
        panels.append(np.concatenate((np.asarray(source_image), np.asarray(decoded_image), waterfall), axis=1))
    return np.stack(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/home/plaing/SSTVAE/The Simpsons Season 31 Episode 20 - The Simpsons Full NoCuts-iex52uxH460.mp4",
    )
    parser.add_argument("--checkpoint", default="models/v8-hf3k-face-gan.pt")
    parser.add_argument(
        "--transmit", default="runs/simpsons-composite-cfr-60s/transmit_candidate.wav"
    )
    parser.add_argument("--out", default="runs/simpsons-composite-cfr-60s/waterfalls")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--name", default="simpsons", help="Filename prefix for rendered clips")
    parser.add_argument("--profiles", nargs="+", choices=tuple(CHANNEL_PROFILES), default=DEFAULT_PROFILES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.name or "/" in args.name:
        raise ValueError("name must be a non-empty filename prefix without slashes")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_frames, raw_audio = extract_source(Path(args.input), args.duration)
    sample_rate, transmit = read_wav_float(Path(args.transmit))
    if sample_rate != COMPOSITE_FS:
        raise ValueError(f"expected {COMPOSITE_FS} Hz transmit WAV, got {sample_rate}")
    required = (args.duration + 1) * COMPOSITE_FS
    if len(transmit) < required:
        raise ValueError(f"transmit WAV has {len(transmit)} samples, requires {required}")
    transmit = transmit[:required]
    codec = AETVCodec(args.checkpoint, device=args.device, mode="V8")
    reference_native = extract_aetv(transmit)
    reference_audio = np.concatenate([
        signal.resample_poly(
            prepare_voice(raw_audio[index * 8_000 : (index + 1) * 8_000]), 2, 3
        )[:8_000]
        for index in range(args.duration)
    ])
    reference_rms = float(np.sqrt(np.mean(transmit[COMPOSITE_FS:-COMPOSITE_FS] ** 2)))

    for profile in args.profiles:
        channel = StreamingChannelEmulator(profile, seed=20260825, fs=COMPOSITE_FS)
        impaired = channel.process(transmit)
        decoded, recovered_audio, lag = decode_received(
            impaired,
            reference_native=reference_native,
            codec=codec,
            duration=args.duration,
        )
        frames = render_frames(
            source_frames,
            decoded,
            received=impaired,
            reference_rms=reference_rms,
            profile=profile,
        )
        destination = out / f"{args.name}_{args.duration}s_{profile}_rx_waterfall.mp4"
        mux(destination, frames, reference_audio, recovered_audio)
        print(f"{profile}: lag={lag} samples -> {destination}", flush=True)


if __name__ == "__main__":
    main()
