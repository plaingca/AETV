"""Matched-air-symbol VVC reference for the V7 held-out clip set.

V7 carries 5,056 complex payload symbols per second. QPSK on the same slots
would carry 10,112 raw bits/s. This script encodes one independently decodable
one-second VVC GOP at several fractions of that raw budget, leaving the rest as
an explicit FEC/header margin, then reports the same spatial and temporal
metrics used by ``compare_checkpoints.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compare_checkpoints import load_clips, metric_values
from train import AETV_MODES


def rgb_to_yuv420(video: torch.Tensor) -> bytes:
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy()
    r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    uv = torch.from_numpy(np.stack([u, v], axis=1)).float()
    uv = F.avg_pool2d(uv, 2, 2).numpy()
    chunks: list[bytes] = []
    for index in range(len(frames)):
        chunks.append(np.rint(y[index] * 255).clip(0, 255).astype(np.uint8).tobytes())
        chunks.append(np.rint(uv[index, 0] * 255).clip(0, 255).astype(np.uint8).tobytes())
        chunks.append(np.rint(uv[index, 1] * 255).clip(0, 255).astype(np.uint8).tobytes())
    return b"".join(chunks)


def yuv420_to_rgb(raw: bytes, frames: int, height: int, width: int) -> torch.Tensor:
    plane = width * height
    frame8 = plane * 3 // 2
    if len(raw) == frames * frame8:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    elif len(raw) == frames * frame8 * 2:
        values = np.frombuffer(raw, dtype="<u2").astype(np.float32) / 1023.0
    else:
        raise RuntimeError(
            f"unexpected decoded YUV size {len(raw)} for {frames}x{width}x{height}"
        )
    samples_per_frame = values.size // frames
    output = []
    for index in range(frames):
        chunk = values[index * samples_per_frame : (index + 1) * samples_per_frame]
        y = chunk[:plane].reshape(height, width)
        uv_plane = plane // 4
        u = chunk[plane : plane + uv_plane].reshape(height // 2, width // 2)
        v = chunk[plane + uv_plane :].reshape(height // 2, width // 2)
        u = np.repeat(np.repeat(u, 2, 0), 2, 1) - 0.5
        v = np.repeat(np.repeat(v, 2, 0), 2, 1) - 0.5
        rgb = np.stack(
            [y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u],
            axis=0,
        )
        output.append(rgb)
    tensor = torch.from_numpy(np.stack(output, axis=1)).clamp(0, 1)
    return tensor.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitrates", nargs="+", type=int, default=[5000, 7500, 10000])
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--cache-dir", default=r"D:\SSTVAE\runs\openvid-cache-5fps-eval-v68")
    parser.add_argument("--vvenc", type=Path, required=True)
    parser.add_argument("--vvdec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mode = AETV_MODES["V7"]
    clips = load_clips(Path(args.cache_dir), mode, args.clips)
    if len(clips) != args.clips:
        raise SystemExit(f"requested {args.clips} clips, found {len(clips)}")
    source = torch.cat(clips, dim=2)
    frame_count = source.shape[2]
    raw_qpsk_bps = mode.latents_per_gop
    report = {
        "codec": "Fraunhofer VVenC 1.14.0 / VVdeC 3.2.0",
        "air_payload_complex_symbols_per_second": raw_qpsk_bps // 2,
        "matched_qpsk_raw_bps": raw_qpsk_bps,
        "clips": len(clips),
        "gop_frames": mode.gop_frames,
        "rates": {},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aetv-vvc-") as temp_name:
        temp = Path(temp_name)
        source_path = temp / "source.yuv"
        source_path.write_bytes(rgb_to_yuv420(source))
        for target_bps in args.bitrates:
            bitstream = args.out / f"vvc_{target_bps}bps.266"
            decoded = temp / f"decoded_{target_bps}.yuv"
            encode = [
                str(args.vvenc), "-i", str(source_path), "-s", f"{mode.width}x{mode.height}",
                "-c", "yuv420", "--fps", str(int(mode.fps)), "-f", str(frame_count),
                "--preset", "fast", "--bitrate", str(target_bps), "--passes", "2",
                "--intraperiod", str(mode.gop_frames), "--refreshtype", "idr",
                "--qpa", "on", "--sdr", "sdr_709", "-o", str(bitstream), "-v", "warning",
            ]
            subprocess.run(encode, check=True, timeout=1800)
            subprocess.run(
                [str(args.vvdec), "-b", str(bitstream), "-o", str(decoded), "-v", "1"],
                check=True,
                timeout=300,
            )
            reconstruction = yuv420_to_rgb(
                decoded.read_bytes(), frame_count, mode.height, mode.width
            )
            per_clip = []
            for index, clip in enumerate(clips):
                start = index * mode.gop_frames
                recon_clip = reconstruction[:, :, start : start + mode.gop_frames]
                per_clip.append(metric_values(recon_clip, clip, torch.device("cuda")))
            actual_bps = bitstream.stat().st_size * 8.0 / len(clips)
            report["rates"][str(target_bps)] = {
                "actual_bps": actual_bps,
                "transport_margin_fraction": 1.0 - actual_bps / raw_qpsk_bps,
                "mean": {
                    metric: st.mean(item[metric] for item in per_clip)
                    for metric in per_clip[0]
                },
                "per_clip": per_clip,
            }
            print(target_bps, report["rates"][str(target_bps)], flush=True)

    output = args.out / "vvc_results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
