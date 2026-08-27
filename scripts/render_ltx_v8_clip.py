#!/usr/bin/env python3
"""Render a held-out clip through released V8 and the trained LTX-V8 path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetv.ltx_channel import LTX_REPO, LTXV8ChannelAdapter, finish_ltx_video, prepare_ltx_video  # noqa: E402
from compare_checkpoints import load_clips, metric_values  # noqa: E402
from train import AETV_MODES, AETVAutoencoder, simulate_transmission  # noqa: E402


def tensor_frames(video: torch.Tensor) -> np.ndarray:
    return (
        video[0].detach().float().clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy() * 255
    ).round().astype(np.uint8)


def write_video(path: Path, frames: np.ndarray, fps: float = 6.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libopenh264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.run(command, input=frames.tobytes(), check=False)
    if process.returncode:
        command[command.index("libopenh264")] = "mpeg4"
        subprocess.run(command, input=frames.tobytes(), check=True)


def comparison_frames(panels: list[tuple[str, torch.Tensor]]) -> np.ndarray:
    arrays = [(label, tensor_frames(video)) for label, video in panels]
    output = []
    for frame_index in range(arrays[0][1].shape[0]):
        images = []
        for label, frames in arrays:
            image = Image.fromarray(frames[frame_index])
            canvas = Image.new("RGB", (image.width, image.height + 22), "black")
            canvas.paste(image, (0, 22))
            ImageDraw.Draw(canvas).text((5, 5), label, fill="white")
            images.append(np.asarray(canvas))
        output.append(np.concatenate(images, axis=1))
    return np.stack(output)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip", type=int, default=18)
    parser.add_argument("--snr", type=float, default=12.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/openvid-cache-5fps-eval-v68"))
    args = parser.parse_args()
    device = torch.device("cuda")
    mode = AETV_MODES["V8"]
    clips = load_clips(args.cache_dir, mode, args.clip + 1)
    source = clips[args.clip].to(device)

    v8 = AETVAutoencoder(mode=mode, width=128, latent_channels=3).to(device).eval()
    v8.load_pretrained_weights("models/v8-hf3k-face-gan.pt", device=device)
    v8_latent = v8.encoder(source)
    v8_rx, v8_confidence, _ = simulate_transmission(
        v8_latent[0].float().cpu().numpy(), "V8", snr_db=args.snr
    )
    v8_render = v8.decoder(
        torch.from_numpy(v8_rx).unsqueeze(0).to(device),
        torch.from_numpy(v8_confidence).unsqueeze(0).to(device),
        (6, 108, 192),
    )
    del v8
    torch.cuda.empty_cache()

    from diffusers import AutoencoderKLLTXVideo

    payload = torch.load(args.candidate, map_location="cpu", weights_only=False)
    adapter = LTXV8ChannelAdapter().to(device).eval()
    adapter.load_state_dict(payload["adapter"])
    vae = AutoencoderKLLTXVideo.from_pretrained(
        payload.get("ltx_repo", LTX_REPO), subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    vae.decoder.conv_in.float()
    vae.decoder.conv_in.load_state_dict(payload["decoder_conv_in"])
    latent = vae.encode(prepare_ltx_video(source).to(torch.bfloat16)).latent_dist.mode()
    symbols = adapter.encode(latent.float())
    ltx_rx, ltx_confidence, _ = simulate_transmission(
        symbols[0].float().cpu().numpy(), "V8", snr_db=args.snr
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ltx_clean = finish_ltx_video(
            vae.decode(adapter.decode(symbols, torch.ones_like(symbols))).sample
        )
        ltx_render = finish_ltx_video(
            vae.decode(
                adapter.decode(
                    torch.from_numpy(ltx_rx).unsqueeze(0).to(device),
                    torch.from_numpy(ltx_confidence).unsqueeze(0).to(device),
                )
            ).sample
        )

    args.out.mkdir(parents=True, exist_ok=True)
    write_video(args.out / "source.mp4", tensor_frames(source))
    write_video(args.out / "ltx-clean.mp4", tensor_frames(ltx_clean))
    write_video(args.out / f"ltx-{args.snr:g}db.mp4", tensor_frames(ltx_render))
    write_video(
        args.out / f"comparison-{args.snr:g}db.mp4",
        comparison_frames([
            ("Source", source),
            (f"V8 {args.snr:g} dB", v8_render),
            ("LTX clean", ltx_clean),
            (f"LTX {args.snr:g} dB", ltx_render),
        ]),
    )
    manifest = {
        "clip_index": args.clip,
        "snr_db": args.snr,
        "frames": 6,
        "fps": 6,
        "metrics": {
            "v8": metric_values(v8_render, source, device),
            "ltx_clean": metric_values(ltx_clean, source, device),
            "ltx_channel": metric_values(ltx_render, source, device),
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
