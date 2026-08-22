#!/usr/bin/env python3
"""Evaluate AETV model on held-out OpenVid clips and dump side-by-side comparison videos.

Generates labeled multi-panel MP4 videos showing:
Source | Clean Loopback | HF 18 dB | HF 12 dB | HF 6 dB | Multipath Fading

Usage:
  python scripts/eval_aetv_clips.py --checkpoint runs/aetv-v1-stage2/checkpoint.pt --out runs/aetv-eval-clips --clips 4
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv import (
    AETV_MODES,
    AETVAutoencoder,
    AETVSyntheticVideoDataset,
    demodulate_gop_stream,
    modulate_gop_stream,
)
from aetv.hfchannel import awgn, fading, freq_shift
from aetv.video_data import HFViewerVideoDataset, VideoClipSpec


def write_labeled_grid_mp4(panels: list[tuple[str, torch.Tensor]], path: Path, fps: float, columns: int = 3):
    """Write labeled video panels as one compact grid MP4."""
    if not panels:
        return
    _, _, frames_count, height, width = panels[0][1].shape
    rows = math.ceil(len(panels) / columns)
    black = torch.zeros_like(panels[0][1])
    padded = list(panels) + [("", black)] * (rows * columns - len(panels))
    grid_rows = []
    for r in range(rows):
        row_panels = [video[0] for _, video in padded[r * columns : (r + 1) * columns]]
        grid_rows.append(torch.cat(row_panels, dim=-1))
    grid = torch.cat(grid_rows, dim=-2).clamp(0, 1)
    raw_frames = grid.mul(255).byte().permute(1, 2, 3, 0).contiguous().cpu().numpy()
    labeled_frames = []
    for raw_frame in raw_frames:
        image = Image.fromarray(raw_frame)
        draw = ImageDraw.Draw(image)
        for index, (label, _) in enumerate(padded):
            if not label:
                continue
            x = (index % columns) * width
            y = (index // columns) * height
            draw.rectangle((x, y, x + width, y + 16), fill=(0, 0, 0))
            draw.text((x + 4, y + 2), label, fill=(255, 255, 255))
        labeled_frames.append(image)

    # Save first frame snapshot PNG
    if labeled_frames:
        png_path = path.with_suffix(".png")
        labeled_frames[0].save(png_path)

    raw_bytes = b"".join(img.tobytes() for img in labeled_frames)
    output_height, output_width = rows * height, columns * width
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{output_width}x{output_height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.run(cmd, input=raw_bytes, stderr=subprocess.PIPE, timeout=60)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-500:])


def compute_psnr(recon: torch.Tensor, orig: torch.Tensor) -> float:
    mse = torch.mean((recon - orig) ** 2).item()
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def simulate_transmission(
    latents: np.ndarray,
    mode_name: str,
    snr_db: float | None = None,
    fading_preset: str | None = None,
    cfo_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Modulate, impair, demodulate, and return (rec_latents, rec_weights, pilot_snr)."""
    mode = AETV_MODES[mode_name]
    audio = modulate_gop_stream([latents], mode_name=mode_name, callsign="EVAL")
    impaired = audio.copy()
    fs = mode.geometry.fs
    if fading_preset:
        impaired = fading(impaired, preset=fading_preset, seed=42, fs=fs)
    if cfo_hz != 0.0:
        impaired = freq_shift(impaired, cfo_hz, fs=fs)
    if snr_db is not None:
        impaired = awgn(impaired, snr_db=snr_db, seed=42, fs=fs)

    demod_res = demodulate_gop_stream(impaired, band=mode.band, drift_track="off")
    rec_lat = demod_res.gops_latents[0] if demod_res.gops_latents else np.zeros_like(latents)
    rec_w = demod_res.gops_weights[0] if demod_res.gops_weights else np.zeros_like(latents)
    return rec_lat, rec_w, demod_res.snr_db


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    ap.add_argument("--out", type=str, default="runs/aetv-eval-clips", help="Output directory")
    ap.add_argument("--clips", type=int, default=4, help="Number of held-out clips to evaluate")
    ap.add_argument("--cache-dir", type=str, default="data/openvid_aetv_cache", help="Cache directory for video clips")
    ap.add_argument("--hf-dataset", type=str, default="lance-format/Openvid-1M")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    mode_name = ckpt.get("mode", ckpt_args.get("mode", "V1"))
    mode_spec = AETV_MODES[mode_name]
    model_width = ckpt_args.get("model_width", 192)
    latent_channels = ckpt_args.get("latent_channels", 12)
    compact = ckpt_args.get("compact", False)
    step = ckpt.get("step", 0)

    print(f"=== Evaluating AETV Mode {mode_spec.name} ({mode_spec.description}) ===", flush=True)
    print(f"Checkpoint: {args.checkpoint} (Step={step}, model_width={model_width}, latent_channels={latent_channels})", flush=True)

    model = AETVAutoencoder(
        mode=mode_spec,
        width=model_width,
        latent_channels=latent_channels,
        compact=compact,
        causal=mode_spec.causal,
    ).to(device).eval()
    model.load_pretrained_weights(args.checkpoint, device=device)

    # Load held-out evaluation clips
    clip_spec = VideoClipSpec(
        frames=mode_spec.gop_frames,
        fps=mode_spec.fps,
        height=mode_spec.height,
        width=mode_spec.width,
    )
    clips = []

    # Check local cache first if available
    mode_suffix = f"mode_{mode_spec.name.lower()}_{mode_spec.width}x{mode_spec.height}_{int(mode_spec.fps)}fps"
    cache_path = Path(args.cache_dir) / mode_suffix if args.cache_dir else None
    if cache_path and cache_path.exists():
        cached_files = sorted(list(cache_path.glob("*.pt")))
        if len(cached_files) >= args.clips:
            print(f"Loading {args.clips} held-out evaluation clips from cache ({cache_path})...", flush=True)
            # Pick evenly spaced clips across the cache for diversity
            indices = np.linspace(0, len(cached_files) - 1, args.clips, dtype=int)
            for idx in indices:
                c = torch.load(cached_files[idx]).float()
                if c.max() > 1.0:
                    c = c / 255.0
                if c.ndim == 4:
                    c = c.unsqueeze(0)
                clips.append(c)

    if len(clips) < args.clips:
        print(f"Streaming {args.clips} held-out evaluation clips from {args.hf_dataset}...", flush=True)
        try:
            stream = HFViewerVideoDataset(
                dataset=args.hf_dataset,
                spec=clip_spec,
                epoch_size=args.clips,
                seed=10000,
                page_size=min(16, args.clips),
            )
            for c in stream:
                clips.append(c.unsqueeze(0))
                if len(clips) >= args.clips:
                    break
        except Exception as error:
            print(f"Streaming held-out clips fell back to synthetic ({error})", flush=True)
            synthetic = AETVSyntheticVideoDataset(mode_spec=mode_spec, count=args.clips, seed=10000)
            clips = [synthetic[i].unsqueeze(0) for i in range(args.clips)]

    print(f"Evaluating {len(clips)} clips through AETV OFDM modem & HF channel simulations...", flush=True)
    summary_results = []

    for clip_idx, video in enumerate(clips):
        video = video.to(device)
        with torch.no_grad():
            z = model.encoder(video)  # (1, N_latents)
        lat_np = z.squeeze(0).cpu().numpy()

        panels = [("Source (Clean)", video.cpu())]
        clip_metrics = {"clip_index": clip_idx}

        # 1. Clean loopback (no channel noise)
        rec_lat_clean, rec_w_clean, _ = simulate_transmission(lat_np, mode_name=mode_name, snr_db=None)
        with torch.no_grad():
            recon_clean = model.decoder(
                torch.from_numpy(rec_lat_clean).float().unsqueeze(0).to(device),
                torch.from_numpy(rec_w_clean).float().unsqueeze(0).to(device),
                output_shape=(mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
        psnr_clean = compute_psnr(recon_clean, video)
        panels.append((f"AETV Clean ({psnr_clean:.1f} dB)", recon_clean.cpu()))
        clip_metrics["clean_psnr"] = psnr_clean

        # 2. HF Channel @ 24 dB SNR (Excellent)
        rec_lat_24, rec_w_24, _ = simulate_transmission(lat_np, mode_name=mode_name, snr_db=24.0)
        with torch.no_grad():
            recon_24 = model.decoder(
                torch.from_numpy(rec_lat_24).float().unsqueeze(0).to(device),
                torch.from_numpy(rec_w_24).float().unsqueeze(0).to(device),
                output_shape=(mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
        psnr_24 = compute_psnr(recon_24, video)
        panels.append((f"HF 24 dB ({psnr_24:.1f} dB)", recon_24.cpu()))
        clip_metrics["snr24_psnr"] = psnr_24

        # 3. HF Channel @ 12 dB SNR (Good conditions)
        rec_lat_18, rec_w_18, _ = simulate_transmission(lat_np, mode_name=mode_name, snr_db=12.0)
        with torch.no_grad():
            recon_18 = model.decoder(
                torch.from_numpy(rec_lat_18).float().unsqueeze(0).to(device),
                torch.from_numpy(rec_w_18).float().unsqueeze(0).to(device),
                output_shape=(mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
        psnr_18 = compute_psnr(recon_18, video)
        panels.append((f"HF 12 dB ({psnr_18:.1f} dB)", recon_18.cpu()))
        clip_metrics["snr18_psnr"] = psnr_18

        # 4. HF Channel @ 6 dB SNR (Standard HF conditions)
        rec_lat_12, rec_w_12, _ = simulate_transmission(lat_np, mode_name=mode_name, snr_db=6.0)
        with torch.no_grad():
            recon_12 = model.decoder(
                torch.from_numpy(rec_lat_12).float().unsqueeze(0).to(device),
                torch.from_numpy(rec_w_12).float().unsqueeze(0).to(device),
                output_shape=(mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
        psnr_12 = compute_psnr(recon_12, video)
        panels.append((f"HF 6 dB ({psnr_12:.1f} dB)", recon_12.cpu()))
        clip_metrics["snr12_psnr"] = psnr_12

        # 5. HF Channel with Multipath Fading (mpg preset)
        rec_lat_fade, rec_w_fade, _ = simulate_transmission(
            lat_np, mode_name=mode_name, fading_preset="mpg", snr_db=3.0
        )
        with torch.no_grad():
            recon_fade = model.decoder(
                torch.from_numpy(rec_lat_fade).float().unsqueeze(0).to(device),
                torch.from_numpy(rec_w_fade).float().unsqueeze(0).to(device),
                output_shape=(mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
        psnr_fade = compute_psnr(recon_fade, video)
        panels.append((f"HF Fading ({psnr_fade:.1f} dB)", recon_fade.cpu()))
        clip_metrics["fading_psnr"] = psnr_fade

        # Write output grid MP4
        video_out_path = out_dir / f"clip_{clip_idx:02d}_comparison.mp4"
        write_labeled_grid_mp4(panels, video_out_path, fps=mode_spec.fps, columns=3)
        print(
            f"Clip {clip_idx:02d}: Clean={psnr_clean:.2f} dB, 24dB={psnr_24:.2f} dB, "
            f"18dB={psnr_18:.2f} dB, 12dB={psnr_12:.2f} dB, Fading={psnr_fade:.2f} dB -> {video_out_path.name}",
            flush=True,
        )
        summary_results.append(clip_metrics)

    # Save summary JSON
    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nEvaluation complete! Side-by-side comparison videos dumped to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
