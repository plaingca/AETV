#!/usr/bin/env python3
"""Render a labeled source/reconstruction sanity clip from an overlap checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aetv.overlap_models import OverlappingGOPAutoencoder  # noqa: E402


def extract_video(path: Path, frames: int, fps: int, width: int, height: int) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-vf",
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}",
        "-frames:v", str(frames), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True, timeout=180)
    expected = frames * height * width * 3
    if len(result.stdout) != expected:
        raise RuntimeError(f"ffmpeg returned {len(result.stdout)} bytes, expected {expected}")
    return np.frombuffer(result.stdout, dtype=np.uint8).copy().reshape(
        frames, height, width, 3
    )


def load_model(path: Path, device: torch.device) -> tuple[OverlappingGOPAutoencoder, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != OverlappingGOPAutoencoder.checkpoint_kind:
        raise ValueError(f"{path} is not an overlapping-GOP checkpoint")
    config = payload["model_config"]
    model = OverlappingGOPAutoencoder(
        mode=config["mode"],
        width=config["width"],
        latent_channels=config["latent_channels"],
        window_gops=config["window_gops"],
        emit_gops=config.get("emit_gops", 1),
        synthesis_halo_frames=config.get("synthesis_halo_frames", 2),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), int(payload["step"])


def write_comparison(path: Path, source: np.ndarray, recon: np.ndarray, fps: int, step: int) -> None:
    frames = []
    for source_frame, recon_frame in zip(source, recon, strict=True):
        image = Image.fromarray(np.concatenate((source_frame, recon_frame), axis=1))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 17), fill=(0, 0, 0))
        draw.text((4, 3), "SOURCE", fill=(255, 255, 255))
        draw.text((source_frame.shape[1] + 4, 3), f"OVERLAP STEP {step}", fill=(255, 255, 255))
        frames.append(np.asarray(image).tobytes())
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{source.shape[2] * 2}x{source.shape[1]}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libopenh264", "-b:v", "1800k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    subprocess.run(command, input=b"".join(frames), check=True, timeout=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gops", type=int, default=14)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    mode = model.mode
    if args.gops < model.window_gops or (args.gops - model.window_gops) % model.emit_gops:
        raise ValueError("gops must tile complete overlapping decoder windows")

    raw = extract_video(
        args.input, args.gops * mode.gop_frames, mode.fps, mode.width, mode.height
    )
    source = torch.from_numpy(raw).permute(0, 3, 1, 2).unsqueeze(0).permute(0, 2, 1, 3, 4)
    source = source.to(device=device, dtype=torch.float32).div_(255)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        latents = model.encode_sequence(source)
        recon = model.decode_sequence(latents)
        target = model.target_for_sequence(source)
    mse = float(torch.mean((recon.float() - target) ** 2))
    psnr = 10 * math.log10(1 / max(mse, 1e-12))
    target_frames = target[0].permute(1, 2, 3, 0).mul(255).byte().cpu().numpy()
    recon_frames = recon[0].permute(1, 2, 3, 0).mul(255).byte().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_comparison(args.out, target_frames, recon_frames, mode.fps, step)
    report = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "source": str(args.input),
        "input_gops": args.gops,
        "output_frames": len(recon_frames),
        "clean_psnr_db": psnr,
        "clean_mse": mse,
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
