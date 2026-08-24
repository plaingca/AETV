#!/usr/bin/env python3
"""Train native AETV spatiotemporal autoencoder for 1 full epoch (250,000 steps / 1M clips).

Features:
- Continuous live-streaming from lance-format/Openvid-1M (16 background threads)
- Direct pixel distortion minimization (MSE + L1 + Spatial Gradient)
- TensorBoard tracking every 50 steps (scalars + throughput)
- Periodic full OFDM HF modem evaluation on 5 held-out clips every 2,000 steps
- Video/Image grid snapshots logged directly into TensorBoard and exported to disk

Usage:
  python scripts/train_aetv.py --mode V1 --init-checkpoint runs/aetv-v1-extended-psnr/checkpoint.pt --out runs/aetv-v1-epoch1 --steps 250000 --eval-interval 2000 --tb-interval 50 --batch 4 --threads 16 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv import (
    AETV_MODES,
    AETVAutoencoder,
    AETVChannelConfig,
    AETVLatentChannel,
    AETVModeSpec,
    AETVSyntheticVideoDataset,
    AETVWaveformChannel,
    SpatioTemporalDiscriminator3D,
    demodulate_gop_stream,
    modulate_gop_stream,
)
from aetv.hfchannel import awgn, fading, freq_shift
from aetv.video_data import HFViewerVideoDataset, VideoClipSpec


def spatial_gradient_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match horizontal and vertical edge maps position by position.

    The gradient maps are compared elementwise. Reducing each map to its mean
    before comparing would only match average edge energy, which says nothing
    about where the edges are and is cheapest to satisfy by adding uniform grain.
    """
    recon_dx = recon[:, :, :, :, 1:] - recon[:, :, :, :, :-1]
    target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    recon_dy = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    target_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def temporal_delta_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match inter-frame differences position by position (motion fidelity)."""
    if recon.shape[2] < 2:
        return recon.new_zeros(())
    return F.l1_loss(recon[:, :, 1:] - recon[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def temporal_acceleration_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match second-order frame differences, directly penalizing flicker/judder."""
    if recon.shape[2] < 3:
        return recon.new_zeros(())
    recon_accel = recon[:, :, 2:] - 2.0 * recon[:, :, 1:-1] + recon[:, :, :-2]
    target_accel = target[:, :, 2:] - 2.0 * target[:, :, 1:-1] + target[:, :, :-2]
    return F.l1_loss(recon_accel, target_accel)


def temporal_energy_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match visible motion magnitude without rewarding spatial noise."""
    if recon.shape[2] < 2:
        return recon.new_zeros(())
    recon_delta = recon[:, :, 1:] - recon[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(
        recon_delta.abs().mean(dim=(-2, -1)),
        target_delta.abs().mean(dim=(-2, -1)),
    )


def temporal_cosine_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reward inter-frame changes in the correct signed direction."""
    if recon.shape[2] < 2:
        return recon.new_zeros(())
    recon_delta = (recon[:, :, 1:] - recon[:, :, :-1]).flatten(1)
    target_delta = (target[:, :, 1:] - target[:, :, :-1]).flatten(1)
    return (1.0 - F.cosine_similarity(
        recon_delta, target_delta, dim=1, eps=1e-6
    )).mean()


_SQRT_HALF = 0.7071067811865476


def _haar_split(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One orthonormal Haar lift along `dim`, which must have even length."""
    x = x.movedim(dim, -1)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    lo = (even + odd) * _SQRT_HALF
    hi = (even - odd) * _SQRT_HALF
    return lo.movedim(-1, dim), hi.movedim(-1, dim)


def dwt3d_bands(x: torch.Tensor, levels: int = 3) -> list[torch.Tensor]:
    """Spatio-temporal Haar wavelet pyramid of a (B, C, T, H, W) tensor.

    Each level splits time, height and width into 8 subbands, keeps the 7 detail
    bands and recurses on the all-lowpass one. Only even-length axes are split,
    so a 12-frame GOP stops subdividing time (12 -> 6 -> 3) while space carries
    on. Truncating an odd axis instead would drop that sample from every band at
    this level and below, leaving part of the clip outside the loss entirely;
    skipping keeps the transform orthonormal, which the Parseval check in
    tests asserts.
    """
    bands: list[torch.Tensor] = []
    current = x
    for _ in range(levels):
        components = [current]
        for dim in (2, 3, 4):
            if components[0].shape[dim] % 2 != 0:
                continue
            split: list[torch.Tensor] = []
            for comp in components:
                lo, hi = _haar_split(comp, dim)
                split.append(lo)
                split.append(hi)
            components = split
        if len(components) == 1:
            break
        bands.extend(components[1:])
        current = components[0]
    bands.append(current)
    return bands


def dwt3d_loss(recon: torch.Tensor, target: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """L1 across a spatio-temporal wavelet pyramid (LTX-Video's Video-DWT loss).

    Pixel L1/L2 underweights high-frequency detail; comparing per-subband makes
    each frequency band count equally, which also gives the encoder a basis on
    which to decide what to protect against channel corruption. Computed in fp32
    because the detail bands are small differences.
    """
    pred_bands = dwt3d_bands(recon.float(), levels)
    targ_bands = dwt3d_bands(target.float(), levels)
    total = sum(F.l1_loss(p, t) for p, t in zip(pred_bands, targ_bands))
    return total / len(pred_bands)


class LeCAM:
    """LeCAM regularizer (Tseng et al. 2021) for discriminator stability.

    Tracks EMAs of the real and fake scores and penalizes each side for drifting
    past the other's average, which bounds how far apart the two distributions
    can be pulled. Standard weight is 0.001.
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema_real: torch.Tensor | None = None
        self.ema_fake: torch.Tensor | None = None

    def update(self, real: torch.Tensor, fake: torch.Tensor) -> bool:
        """Fold this batch into the averages. Returns False if it was rejected.

        An EMA is absorbing: once a NaN enters, every later value is NaN and the
        state can never recover. So one non-finite batch here permanently kills
        the discriminator's loss while leaving its weights and the generator
        untouched -- which presents as the critic silently ceasing to train,
        several thousand steps after the batch that caused it. Rejecting the
        batch costs one update out of thousands.
        """
        r = real.detach().mean()
        f = fake.detach().mean()
        if not (torch.isfinite(r) and torch.isfinite(f)):
            return False
        self.ema_real = r if self.ema_real is None else self.decay * self.ema_real + (1 - self.decay) * r
        self.ema_fake = f if self.ema_fake is None else self.decay * self.ema_fake + (1 - self.decay) * f
        return True

    def is_poisoned(self) -> bool:
        """True if the state has become non-finite despite the update guard."""
        return any(
            e is not None and not torch.isfinite(e) for e in (self.ema_real, self.ema_fake)
        )

    def reset(self) -> None:
        """Drop the averages so they re-seed from the next clean batch."""
        self.ema_real = None
        self.ema_fake = None

    def regularizer(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        if self.ema_real is None or self.ema_fake is None:
            return real.new_zeros(())
        return (real - self.ema_fake).pow(2).mean() + (fake - self.ema_real).pow(2).mean()

    def state_dict(self) -> dict:
        return {
            "ema_real": None if self.ema_real is None else float(self.ema_real),
            "ema_fake": None if self.ema_fake is None else float(self.ema_fake),
        }


_LPIPS_METRIC: "lpips.LPIPS | None" = None
_LPIPS_UNAVAILABLE = False


def lpips_metric(recon: torch.Tensor, target: torch.Tensor, device: torch.device) -> float:
    """Reference LPIPS (AlexNet) on a (1, 3, T, H, W) clip pair in [0, 1].

    Returned as an evaluation metric only, never as a training objective, so the
    trainer's perceptual loss and this number stay independently checkable.
    """
    global _LPIPS_METRIC, _LPIPS_UNAVAILABLE
    if _LPIPS_UNAVAILABLE:
        return float("nan")
    if _LPIPS_METRIC is None:
        try:
            _LPIPS_METRIC = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:  # no cached weights and no network
            print(f"Warning: LPIPS metric unavailable ({exc}); reporting NaN.", flush=True)
            _LPIPS_UNAVAILABLE = True
            return float("nan")
    with torch.no_grad():
        r = recon[0].permute(1, 0, 2, 3).clamp(0, 1).to(device) * 2 - 1
        t = target[0].permute(1, 0, 2, 3).clamp(0, 1).to(device) * 2 - 1
        return float(_LPIPS_METRIC(r, t).mean().item())


def compute_psnr(recon: torch.Tensor, orig: torch.Tensor) -> float:
    mse = torch.mean((recon - orig) ** 2).item()
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / max(1e-7, mse)))


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


def create_labeled_grid_image(panels: list[tuple[str, torch.Tensor]], columns: int = 3) -> Image.Image:
    """Create a single composite labeled image of the first frame across all channel conditions."""
    rows = math.ceil(len(panels) / columns)
    height, width = panels[0][1].shape[-2:]
    out_img = Image.new("RGB", (columns * width, rows * height), color=(0, 0, 0))
    draw = ImageDraw.Draw(out_img)

    for index, (label, tensor) in enumerate(panels):
        x = (index % columns) * width
        y = (index // columns) * height
        frame_np = (tensor[0, :, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        panel_img = Image.fromarray(frame_np)
        out_img.paste(panel_img, (x, y))
        draw.rectangle((x, y, x + width, y + 16), fill=(0, 0, 0))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 255))

    return out_img


def write_labeled_grid_mp4(panels: list[tuple[str, torch.Tensor]], path: Path, fps: float, columns: int = 3):
    """Write labeled video panels as one compact grid MP4 and snapshot PNG."""
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

    # Save snapshot PNG
    if labeled_frames:
        labeled_frames[0].save(path.with_suffix(".png"))

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


class BackgroundVideoStreamingQueue:
    """Continuously streams and caches OpenVid clips in background threads."""

    def __init__(
        self,
        mode_spec: AETVModeSpec,
        dataset_name: str = "lance-format/Openvid-1M",
        cache_dir: Path | str = "data/openvid_aetv_cache",
        queue_size: int = 256,
        num_fetch_threads: int = 16,
    ):
        self.mode_spec = mode_spec
        self.dataset_name = dataset_name
        # Use mode-specific cache directory to avoid mixing resolutions across modes
        mode_suffix = f"mode_{mode_spec.name.lower()}_{mode_spec.width}x{mode_spec.height}_{int(mode_spec.fps)}fps"
        self.cache_dir = Path(cache_dir) / mode_suffix
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.cached_files = list(self.cache_dir.glob("*.pt"))
        self.spec = VideoClipSpec(
            frames=mode_spec.gop_frames,
            fps=mode_spec.fps,
            height=mode_spec.height,
            width=mode_spec.width,
        )
        self.num_fetch_threads = num_fetch_threads


        # Start background stream producers
        self.worker_threads = []
        for worker_id in range(num_fetch_threads):
            t = threading.Thread(target=self._stream_producer, args=(worker_id,), daemon=True)
            t.start()
            self.worker_threads.append(t)

    def _stream_producer(self, worker_id: int):
        seed = 1000 + worker_id * 37
        while not self.stop_event.is_set():
            try:
                stream = HFViewerVideoDataset(
                    dataset=self.dataset_name,
                    spec=self.spec,
                    epoch_size=100_000,
                    seed=seed,
                    page_size=64,
                )
                for i, clip in enumerate(stream):
                    if self.stop_event.is_set():
                        break
                    if len(self.cached_files) < 10000:
                        cache_path = self.cache_dir / f"clip_{worker_id}_{time.time_ns()}.pt"
                        torch.save((clip * 255).byte(), cache_path)
                        self.cached_files.append(cache_path)
                    
                    self.q.put(clip)
            except Exception:
                time.sleep(0.5)
                seed += 101

    def get_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fetch next batch of clips from queue or disk cache."""
        clips = []
        for _ in range(batch_size):
            try:
                clip = self.q.get(timeout=0.05)
            except queue.Empty:
                if self.cached_files:
                    rand_f = np.random.choice(self.cached_files)
                    clip = torch.load(rand_f).float() / 255.0
                else:
                    clip = torch.rand(3, self.mode_spec.gop_frames, self.mode_spec.height, self.mode_spec.width)
            clips.append(clip)
        return torch.stack(clips, dim=0).to(device, non_blocking=True)

    def close(self):
        self.stop_event.set()


def run_evaluation(
    model: AETVAutoencoder,
    eval_clips: list[torch.Tensor],
    mode_spec: AETVModeSpec,
    step: int,
    out_dir: Path,
    writer: SummaryWriter,
    device: torch.device,
):
    """Run full OFDM modem and HF channel evaluation on 5 held-out clips."""
    model.eval()
    eval_step_dir = out_dir / f"eval_step_{step:06d}"
    eval_step_dir.mkdir(parents=True, exist_ok=True)

    clean_psnrs = []
    psnrs_18 = []
    psnrs_12 = []
    psnrs_6 = []
    psnrs_0 = []
    psnrs_fade_mpg = []
    psnrs_fade_ota40m = []
    psnrs_fade_mpp = []
    # PSNR alone cannot show a perceptual-objective change: shifting weight from
    # MSE toward perceptual terms is expected to cost PSNR while improving the
    # picture, so the two are tracked side by side at every operating point.
    lpips_clean = []
    lpips_6 = []
    lpips_0 = []
    lpips_fade_ota40m = []
    lpips_fade_mpp = []

    print(f"\n--- Running OFDM Modem Evaluation at Step {step} (5 Clips) ---", flush=True)
    with torch.no_grad():
        for idx, video in enumerate(eval_clips):
            video_dev = video.to(device)
            z = model.encoder(video_dev)
            lat_np = z[0].cpu().numpy()

            # 1. Clean Loopback
            recon_clean = model.decoder(
                z, torch.ones_like(z), (mode_spec.gop_frames, mode_spec.height, mode_spec.width)
            )
            p_clean = compute_psnr(recon_clean, video_dev)
            clean_psnrs.append(p_clean)
            lpips_clean.append(lpips_metric(recon_clean, video_dev, device))

            # 2. HF 18 dB SNR
            r_lat18, r_w18, _ = simulate_transmission(lat_np, mode_name=mode_spec.name, snr_db=18.0)
            recon_18 = model.decoder(
                torch.from_numpy(r_lat18).unsqueeze(0).to(device),
                torch.from_numpy(r_w18).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_18 = compute_psnr(recon_18, video_dev)
            psnrs_18.append(p_18)

            # 3. HF 12 dB SNR
            r_lat12, r_w12, _ = simulate_transmission(lat_np, mode_name=mode_spec.name, snr_db=12.0)
            recon_12 = model.decoder(
                torch.from_numpy(r_lat12).unsqueeze(0).to(device),
                torch.from_numpy(r_w12).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_12 = compute_psnr(recon_12, video_dev)
            psnrs_12.append(p_12)

            # 4. HF 6 dB SNR (Low SNR static)
            r_lat6, r_w6, _ = simulate_transmission(lat_np, mode_name=mode_spec.name, snr_db=6.0)
            recon_6 = model.decoder(
                torch.from_numpy(r_lat6).unsqueeze(0).to(device),
                torch.from_numpy(r_w6).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_6 = compute_psnr(recon_6, video_dev)
            psnrs_6.append(p_6)
            lpips_6.append(lpips_metric(recon_6, video_dev, device))

            # 5. HF 0 dB SNR (Extreme noise floor)
            r_lat0, r_w0, _ = simulate_transmission(lat_np, mode_name=mode_spec.name, snr_db=0.0)
            recon_0 = model.decoder(
                torch.from_numpy(r_lat0).unsqueeze(0).to(device),
                torch.from_numpy(r_w0).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_0 = compute_psnr(recon_0, video_dev)
            psnrs_0.append(p_0)
            lpips_0.append(lpips_metric(recon_0, video_dev, device))

            # 6. Multipath Fading (mpg - CCIR Good)
            r_lat_mpg, r_w_mpg, _ = simulate_transmission(lat_np, mode_name=mode_spec.name, fading_preset="mpg", snr_db=18.0)
            recon_mpg = model.decoder(
                torch.from_numpy(r_lat_mpg).unsqueeze(0).to(device),
                torch.from_numpy(r_w_mpg).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_mpg = compute_psnr(recon_mpg, video_dev)
            psnrs_fade_mpg.append(p_mpg)

            # 7. Measured 40 m OTA path (K9CZI-1-like)
            r_lat_ota40m, r_w_ota40m, _ = simulate_transmission(
                lat_np,
                mode_name=mode_spec.name,
                fading_preset="ota40m",
                snr_db=5.0,
            )
            recon_ota40m = model.decoder(
                torch.from_numpy(r_lat_ota40m).unsqueeze(0).to(device),
                torch.from_numpy(r_w_ota40m).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_ota40m = compute_psnr(recon_ota40m, video_dev)
            psnrs_fade_ota40m.append(p_ota40m)
            lpips_fade_ota40m.append(
                lpips_metric(recon_ota40m, video_dev, device)
            )

            # 8. Multipath Fading (mpp - CCIR Poor, deep notches)
            r_lat_mpp, r_w_mpp, _ = simulate_transmission(lat_np, mode_name=mode_spec.name, fading_preset="mpp", snr_db=14.0)
            recon_mpp = model.decoder(
                torch.from_numpy(r_lat_mpp).unsqueeze(0).to(device),
                torch.from_numpy(r_w_mpp).unsqueeze(0).to(device),
                (mode_spec.gop_frames, mode_spec.height, mode_spec.width),
            )
            p_mpp = compute_psnr(recon_mpp, video_dev)
            psnrs_fade_mpp.append(p_mpp)
            lpips_fade_mpp.append(lpips_metric(recon_mpp, video_dev, device))

            panels = [
                ("Source", video.cpu()),
                (f"Clean ({p_clean:.1f} dB)", recon_clean.cpu()),
                (f"18 dB ({p_18:.1f} dB)", recon_18.cpu()),
                (f"12 dB ({p_12:.1f} dB)", recon_12.cpu()),
                (f"6 dB ({p_6:.1f} dB)", recon_6.cpu()),
                (f"0 dB ({p_0:.1f} dB)", recon_0.cpu()),
                (f"Fade mpg ({p_mpg:.1f} dB)", recon_mpg.cpu()),
                (f"OTA40m 5 ({p_ota40m:.1f} dB)", recon_ota40m.cpu()),
                (f"Fade mpp ({p_mpp:.1f} dB)", recon_mpp.cpu()),
            ]

            # Save comparison MP4 and PNG snapshot
            vid_out = eval_step_dir / f"eval_clip_{idx:02d}.mp4"
            write_labeled_grid_mp4(panels, vid_out, fps=mode_spec.fps, columns=4)

            # Add snapshot image to TensorBoard
            grid_img = create_labeled_grid_image(panels, columns=4)
            grid_tensor = torch.from_numpy(np.array(grid_img)).permute(2, 0, 1)  # (C, H, W)
            writer.add_image(f"eval_snapshots/clip_{idx:02d}", grid_tensor, global_step=step)

            writer.add_scalar(f"eval_per_clip/clip_{idx:02d}_clean_psnr", p_clean, step)
            writer.add_scalar(f"eval_per_clip/clip_{idx:02d}_6db_psnr", p_6, step)
            writer.add_scalar(f"eval_per_clip/clip_{idx:02d}_0db_psnr", p_0, step)
            writer.add_scalar(
                f"eval_per_clip/clip_{idx:02d}_ota40m_5db_psnr",
                p_ota40m,
                step,
            )
            writer.add_scalar(f"eval_per_clip/clip_{idx:02d}_fading_mpp_psnr", p_mpp, step)

            print(
                f"Eval Clip {idx}: Clean={p_clean:.2f} dB | 18dB={p_18:.2f} dB | 12dB={p_12:.2f} dB | "
                f"6dB={p_6:.2f} dB | 0dB={p_0:.2f} dB | "
                f"Fade mpg={p_mpg:.2f} dB | OTA40m 5={p_ota40m:.2f} dB | "
                f"Fade mpp={p_mpp:.2f} dB",
                flush=True,
            )

    # Log aggregate scalar evaluation metrics to TensorBoard
    mean_clean = float(np.mean(clean_psnrs))
    mean_18 = float(np.mean(psnrs_18))
    mean_12 = float(np.mean(psnrs_12))
    mean_6 = float(np.mean(psnrs_6))
    mean_0 = float(np.mean(psnrs_0))
    mean_fade_mpg = float(np.mean(psnrs_fade_mpg))
    mean_fade_ota40m = float(np.mean(psnrs_fade_ota40m))
    mean_fade_mpp = float(np.mean(psnrs_fade_mpp))

    writer.add_scalar("eval/psnr_clean_mean", mean_clean, step)
    writer.add_scalar("eval/psnr_18db_mean", mean_18, step)
    writer.add_scalar("eval/psnr_12db_mean", mean_12, step)
    writer.add_scalar("eval/psnr_6db_mean", mean_6, step)
    writer.add_scalar("eval/psnr_0db_mean", mean_0, step)
    writer.add_scalar("eval/psnr_fading_mpg_mean", mean_fade_mpg, step)
    writer.add_scalar("eval/psnr_ota40m_5db_mean", mean_fade_ota40m, step)
    writer.add_scalar("eval/psnr_fading_mpp_mean", mean_fade_mpp, step)

    lp_clean = float(np.nanmean(lpips_clean)) if lpips_clean else float("nan")
    lp_6 = float(np.nanmean(lpips_6)) if lpips_6 else float("nan")
    lp_0 = float(np.nanmean(lpips_0)) if lpips_0 else float("nan")
    lp_ota40m = (
        float(np.nanmean(lpips_fade_ota40m))
        if lpips_fade_ota40m
        else float("nan")
    )
    lp_mpp = float(np.nanmean(lpips_fade_mpp)) if lpips_fade_mpp else float("nan")
    if not math.isnan(lp_clean):
        writer.add_scalar("eval/lpips_clean_mean", lp_clean, step)
        writer.add_scalar("eval/lpips_6db_mean", lp_6, step)
        writer.add_scalar("eval/lpips_0db_mean", lp_0, step)
        writer.add_scalar("eval/lpips_ota40m_5db_mean", lp_ota40m, step)
        writer.add_scalar("eval/lpips_fading_mpp_mean", lp_mpp, step)

    print(
        f"Step {step} Eval Summary: Mean Clean={mean_clean:.2f} dB | Mean 12dB={mean_12:.2f} dB | "
        f"Mean 6dB={mean_6:.2f} dB | Mean 0dB={mean_0:.2f} dB | "
        f"Mean OTA40m 5={mean_fade_ota40m:.2f} dB | "
        f"Mean Fade MPP={mean_fade_mpp:.2f} dB",
        flush=True,
    )
    print(
        f"Step {step} Eval LPIPS (lower better): Clean={lp_clean:.4f} | 6dB={lp_6:.4f} | "
        f"0dB={lp_0:.4f} | OTA40m 5={lp_ota40m:.4f} | "
        f"Fade MPP={lp_mpp:.4f}\n",
        flush=True,
    )
    model.train()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=list(AETV_MODES.keys()), default="V7", help="AETV mode (V0-V8)")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1, help="1=Latent AWGN/truncation, 2=OFDM Waveform channel")
    ap.add_argument("--out", type=str, default="runs/aetv-v1-epoch1", help="Output directory for checkpoints & TB logs")
    ap.add_argument("--hf-dataset", type=str, default="lance-format/Openvid-1M", help="Hugging Face dataset")
    ap.add_argument("--cache-dir", type=str, default="data/openvid_aetv_cache", help="Disk cache directory")
    ap.add_argument("--steps", type=int, default=250000, help="Total training steps (250,000 for 1 full epoch)")
    ap.add_argument("--eval-interval", type=int, default=2000, help="Steps between modem evaluation runs")
    ap.add_argument("--tb-interval", type=int, default=50, help="Steps between TensorBoard scalar log updates")
    ap.add_argument("--checkpoint-interval", type=int, default=5000, help="Steps between saving model checkpoints")
    ap.add_argument("--clean-warmup", type=int, default=0, help="Steps with 100%% clean reconstruction warmup")
    ap.add_argument("--channel-ramp", type=int, default=1500, help="Steps to ramp channel noise from 0 to full")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--threads", type=int, default=16, help="Streaming worker threads")
    ap.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate")
    ap.add_argument("--d-lr", type=float, default=3.0e-5, help="Discriminator learning rate (~30%% of --lr)")
    ap.add_argument("--d-every", type=int, default=4, help="Update discriminator on 1 of every N optimizer steps")
    ap.add_argument("--disc-warmup", type=int, default=0, help="Steps before enabling discriminator training")

    # --- Loss weights -------------------------------------------------------
    # Defaults set from measured term magnitudes at mode V7, not from nominal
    # intuition: see scripts/probe_aetv_loss_old_vs_new.py. Target contribution
    # shares are perceptual ~55%, DWT ~17%, L1 ~13%, consistency ~8%, with the
    # rest in temporal/edge/MSE. Re-run that probe after any codec change, since
    # the raw magnitudes move and fixed weights then mean something different.
    ap.add_argument("--mse-weight", type=float, default=2.0, help="Pixel MSE weight")
    ap.add_argument("--l1-weight", type=float, default=1.5, help="Pixel L1 weight")
    ap.add_argument("--dwt-weight", type=float, default=1.0, help="Spatio-temporal Haar DWT L1 weight")
    ap.add_argument("--dwt-levels", type=int, default=3, help="DWT pyramid levels")
    ap.add_argument("--grad-weight", type=float, default=0.5, help="Spatial edge-map L1 weight")
    ap.add_argument("--temporal-weight", type=float, default=0.7, help="Inter-frame delta L1 weight")
    ap.add_argument("--temporal-accel-weight", type=float, default=0.0, help="Second-order temporal L1 weight for flicker/judder suppression")
    ap.add_argument("--temporal-energy-weight", type=float, default=0.0, help="Per-frame motion-energy matching weight")
    ap.add_argument("--temporal-cosine-weight", type=float, default=0.0, help="Signed temporal-delta cosine fidelity weight")
    ap.add_argument("--lpips-weight", type=float, default=0.06, help="Multi-layer VGG perceptual weight on the transmitted render (0.0 to disable)")
    ap.add_argument("--temporal-lpips-weight", type=float, default=0.0, help="VGG perceptual weight on signed inter-frame differences")
    ap.add_argument(
        "--adv-confidence-power",
        type=float,
        default=1.0,
        help=(
            "Exponent on the measured latent-survival confidence that scales the "
            "adversarial term; 0.0 disables scaling, higher retreats faster as the "
            "latent degrades"
        ),
    )
    ap.add_argument("--adv-weight", type=float, default=0.05, help="PatchGAN adversarial loss weight (0.0 to disable)")
    ap.add_argument("--fm-weight", type=float, default=0.10, help="Discriminator feature-matching weight")
    ap.add_argument("--lecam-weight", type=float, default=0.001, help="LeCAM discriminator regularization weight")
    ap.add_argument(
        "--consistency-weight",
        type=float,
        default=1.5,
        help="Weight on |noisy recon - clean recon|; targets decoder insensitivity to channel corruption",
    )
    ap.add_argument(
        "--clean-anchor-weight",
        type=float,
        default=0.0,
        help="Auxiliary clean-render fidelity weight during channel fine-tuning",
    )
    ap.add_argument(
        "--max-nonfinite",
        type=int,
        default=50,
        help=(
            "Abort after this many non-finite generator steps. Isolated bad "
            "batches are skipped and counted; a rising count means divergence, "
            "and continuing only burns GPU on a model that cannot recover"
        ),
    )
    ap.add_argument("--p-truncate", type=float, default=0.30, help="Probability of progressive group truncation")
    ap.add_argument("--model-width", type=int, default=192, help="Base model channels")
    ap.add_argument("--latent-channels", type=int, default=12, help="Latent channels")
    ap.add_argument("--compact", action="store_true", help="Preserve one latent time slice per video frame")
    ap.add_argument("--snr-min", type=float, default=-2.0, help="Minimum channel SNR (dB)")
    ap.add_argument("--snr-max", type=float, default=10.0, help="Maximum channel SNR (dB)")
    ap.add_argument("--snr-focus-min", type=float, default=None, help="Optional OTA-focused mixture minimum SNR (dB)")
    ap.add_argument("--snr-focus-max", type=float, default=None, help="Optional OTA-focused mixture maximum SNR (dB)")
    ap.add_argument("--p-snr-focus", type=float, default=0.0, help="Probability of sampling from the OTA-focused SNR range")
    ap.add_argument("--p-fading", type=float, default=0.70, help="Probability of Watterson frequency-selective fading")
    ap.add_argument(
        "--p-measured-path",
        type=float,
        default=0.40,
        help=(
            "Conditional fraction of fading examples drawn from the measured "
            "40 m OTA mixture (about 0.6 ms delay, 0.24 Hz Doppler and 5 dB SNR)"
        ),
    )
    ap.add_argument("--init-checkpoint", type=str, default=None, help="Initial checkpoint path (optional)")
    ap.add_argument(
        "--reset-steps",
        action="store_true",
        help="Warm-start weights from --init-checkpoint but restart the step count, LR schedule "
        "and optimizer state. Use when changing the objective: inherited Adam moments and an "
        "annealed LR are tuned to the old loss.",
    )
    ap.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision (bfloat16)")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir))

    print(f"TensorBoard logging to: {tb_dir}", flush=True)

    mode_spec = AETV_MODES[args.mode]

    # Built here rather than beside the channel so the startup banner can report
    # the *effective* values. Stage 2 overrides two of them, and a banner that
    # reported the raw arguments claimed p_truncate=0.3 on a stage-2 run that
    # was training with no truncation at all.
    if (args.snr_focus_min is None) != (args.snr_focus_max is None):
        ap.error("--snr-focus-min and --snr-focus-max must be supplied together")
    if not 0.0 <= args.p_snr_focus <= 1.0:
        ap.error("--p-snr-focus must be between 0 and 1")
    if not 0.0 <= args.p_fading <= 1.0:
        ap.error("--p-fading must be between 0 and 1")
    if not 0.0 <= args.p_measured_path <= 1.0:
        ap.error("--p-measured-path must be between 0 and 1")
    focus_range = (
        None
        if args.snr_focus_min is None
        else (args.snr_focus_min, args.snr_focus_max)
    )
    channel_cfg = AETVChannelConfig(
        snr_db_range=(args.snr_min, args.snr_max),
        snr_focus_range=focus_range,
        p_snr_focus=args.p_snr_focus,
        p_fading=args.p_fading,
        p_measured_path=args.p_measured_path,
        p_truncate=args.p_truncate if args.stage == 1 else 0.0,
        erasure_rate_max=0.08 if args.stage == 1 else 0.0,
    )

    print(f"=== Native AETV Full 1-Epoch Training: Mode {mode_spec.name} ({mode_spec.description}) ===", flush=True)
    print(f"Video Spec: {mode_spec.width}x{mode_spec.height} @ {mode_spec.fps} fps, Budget: {mode_spec.latents_per_gop} latents/GOP", flush=True)
    print(
        f"Plan: {args.steps:,} steps ({args.steps * args.batch:,} sampled clips), "
        f"Eval every {args.eval_interval} steps, TB every {args.tb_interval} steps",
        flush=True,
    )
    print(f"Model Width: {args.model_width}, Latent Channels: {args.latent_channels} | Device: {device} | AMP: {args.amp}", flush=True)
    print(
        f"Loss Config: mse={args.mse_weight} l1={args.l1_weight} dwt={args.dwt_weight}"
        f"(L{args.dwt_levels}) grad={args.grad_weight} temporal={args.temporal_weight} "
        f"temporal_accel={args.temporal_accel_weight} perceptual={args.lpips_weight} "
        f"temporal_perceptual={args.temporal_lpips_weight} "
        f"consistency={args.consistency_weight} clean_anchor={args.clean_anchor_weight} "
        f"temporal_energy={args.temporal_energy_weight} "
        f"temporal_cosine={args.temporal_cosine_weight} "
        f"adv={args.adv_weight} fm={args.fm_weight} lecam={args.lecam_weight}\n"
        "Attachment: transmitted-render losses score the noisy latent; the "
        "clean render is a detached consistency target and, when enabled, a "
        "jointly optimized fidelity anchor\n"
        f"Adversarial confidence scaling: conf^{args.adv_confidence_power} from "
        "measured latent SNR (1.0 clean, 0.5 at 0 dB)",
        flush=True,
    )
    print(
        f"Disc Config: d_lr={args.d_lr} update 1-in-{args.d_every} optimizer steps, "
        f"warmup={args.disc_warmup}\n"
        f"Channel: stage {args.stage} "
        f"({'AETVLatentChannel, 0/1 weights' if args.stage == 1 else 'AETVWaveformChannel, continuous pilot-EQ weights'}) "
        f"snr={channel_cfg.snr_db_range[0]}..{channel_cfg.snr_db_range[1]} dB "
        f"focus={channel_cfg.snr_focus_range} p_focus={channel_cfg.p_snr_focus} "
        f"p_fading={channel_cfg.p_fading} "
        f"p_measured_path={channel_cfg.p_measured_path} "
        f"measured={channel_cfg.measured_delay_range_ms} ms/"
        f"{channel_cfg.measured_doppler_range_hz} Hz/"
        f"{channel_cfg.measured_snr_db_range} dB "
        f"p_truncate={channel_cfg.p_truncate} "
        f"erasure_max={channel_cfg.erasure_rate_max}",
        flush=True,
    )

    # Initialize model
    model = AETVAutoencoder(
        mode=mode_spec,
        width=args.model_width,
        latent_channels=args.latent_channels,
        compact=args.compact,
        causal=mode_spec.causal,
    ).to(device)

    start_step = 0
    ckpt_data = None
    if args.init_checkpoint:
        model.load_pretrained_weights(args.init_checkpoint, device=device)
        try:
            ckpt_data = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "step" in ckpt_data:
                if args.reset_steps:
                    print(
                        f"Warm start: loaded weights from step {ckpt_data['step']}, "
                        "restarting step count and LR schedule.",
                        flush=True,
                    )
                else:
                    start_step = ckpt_data["step"]
                    print(f"Resuming training from Step {start_step}...", flush=True)
        except Exception:
            pass

    # Initialize Shallow VGG Perceptual Loss (relu1_2, relu2_2, relu3_3)
    lpips_fn = None
    if args.lpips_weight > 0.0 or args.temporal_lpips_weight > 0.0:
        print(
            "Initializing multi-layer VGG16 perceptual loss "
            "(relu1_2, relu2_2, relu3_3, relu4_3, relu5_3; channel-normalized)...",
            flush=True,
        )
        from aetv.models import MultiLayerVGGPerceptualLoss
        lpips_fn = MultiLayerVGGPerceptualLoss().to(device).eval()

    # Initialize Spatio-Temporal 3D Discriminator
    discriminator = None
    optimizer_d = None
    lr_scheduler_d = None
    if args.adv_weight > 0.0:
        print("Initializing Local 3D Spatio-Temporal PatchGAN Discriminator (32x32 RF)...", flush=True)
        from aetv.models import SpatioTemporalPatchGAN3D
        discriminator = SpatioTemporalPatchGAN3D().to(device)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.9), weight_decay=1e-4)
        lr_scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.steps, eta_min=1e-6)
        if isinstance(ckpt_data, dict) and "discriminator_state_dict" in ckpt_data:
            try:
                discriminator.load_state_dict(ckpt_data["discriminator_state_dict"])
                if "optimizer_d_state_dict" in ckpt_data and not args.reset_steps:
                    optimizer_d.load_state_dict(ckpt_data["optimizer_d_state_dict"])
                print("Restored discriminator weights from checkpoint.", flush=True)
            except Exception as e:
                print(f"Warning: could not restore discriminator state ({e})", flush=True)
        for _ in range(start_step):
            lr_scheduler_d.step()
    lecam = LeCAM()

    # Initialize Channel
    if args.stage == 1:
        channel = AETVLatentChannel(cfg=channel_cfg).to(device)
    else:
        channel = AETVWaveformChannel(band=mode_spec.band, cfg=channel_cfg).to(device)

    # Optimizer with Cosine Annealing over 250k steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    if isinstance(ckpt_data, dict) and "optimizer_state_dict" in ckpt_data and not args.reset_steps:
        try:
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            print("Restored generator optimizer state from checkpoint.", flush=True)
        except Exception as e:
            print(f"Warning: could not restore optimizer state ({e})", flush=True)

    # Fast forward scheduler to start_step if resuming
    for _ in range(start_step):
        lr_scheduler.step()


    # Load 5 fixed held-out evaluation clips
    print("Loading 5 fixed held-out evaluation clips...", flush=True)
    eval_clips = []
    # Use existing cached clips from eval directory or streaming buffer
    cached_eval_files = sorted(list(Path("runs/openvid-cache-5fps-eval-v68").glob("*.pt")))
    if not cached_eval_files:
        mode_suffix = (
            f"mode_{mode_spec.name.lower()}_{mode_spec.width}x"
            f"{mode_spec.height}_{int(mode_spec.fps)}fps"
        )
        cached_eval_files = sorted(
            list((Path(args.cache_dir) / mode_suffix).glob("*.pt"))
        )[:5]
    
    for f in cached_eval_files[:5]:
        clip = torch.load(f).float()
        if clip.ndim == 4:
            if clip.shape[1] < mode_spec.gop_frames:
                repeats = math.ceil(mode_spec.gop_frames / clip.shape[1])
                clip = clip.repeat(1, repeats, 1, 1)[:, :mode_spec.gop_frames]
            else:
                clip = clip[:, :mode_spec.gop_frames]
            if clip.shape[-2:] != (mode_spec.height, mode_spec.width):
                # Proportional scale to cover target dimensions, then center crop
                h_orig, w_orig = clip.shape[-2:]
                scale = max(mode_spec.height / h_orig, mode_spec.width / w_orig)
                h_new = int(round(h_orig * scale))
                w_new = int(round(w_orig * scale))
                clip_scaled = F.interpolate(
                    clip.unsqueeze(0),
                    size=(mode_spec.gop_frames, h_new, w_new),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
                top = (h_new - mode_spec.height) // 2
                left = (w_new - mode_spec.width) // 2
                clip = clip_scaled[:, :, top : top + mode_spec.height, left : left + mode_spec.width]
            if clip.max() > 1.0:
                clip = clip / 255.0
            eval_clips.append(clip.unsqueeze(0))

    if len(eval_clips) < 5:
        # Uniform noise is not reconstructable at any rate, so a padded eval set
        # reports a number that says nothing about the model. Say so loudly.
        print(
            f"WARNING: only {len(eval_clips)} real eval clips found (searched "
            f"'runs/openvid-cache-5fps-eval-v68' then '{args.cache_dir}'). Padding with "
            "uniform noise, which will drag every eval metric down for reasons "
            "unrelated to the model.",
            flush=True,
        )
    while len(eval_clips) < 5:
        synth = torch.rand(1, 3, mode_spec.gop_frames, mode_spec.height, mode_spec.width)
        eval_clips.append(synth)

    # Background streaming pipeline (16 threads)
    print(f"Connecting to background OpenVid streaming queue ({args.threads} threads)...", flush=True)
    stream_pipeline = BackgroundVideoStreamingQueue(
        mode_spec=mode_spec,
        dataset_name=args.hf_dataset,
        cache_dir=args.cache_dir,
        num_fetch_threads=args.threads,
    )

    # Step 0 Baseline Evaluation
    if start_step == 0:
        run_evaluation(model, eval_clips, mode_spec, step=0, out_dir=out_dir, writer=writer, device=device)

    # Training Loop
    print(f"\nStarting 1-Epoch AETV Training Loop (from Step {start_step + 1} to {args.steps})...", flush=True)
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    if optimizer_d is not None:
        optimizer_d.zero_grad(set_to_none=True)

    # Carried across steps so the console line can report a real discriminator
    # value. --tb-interval and the --d-every cycle alias against each other, so
    # sampling loss_d only on logged steps can show "idle" for thousands of
    # steps while the discriminator is training normally.
    last_loss_d = 0.0
    last_disc_step = 0

    # Counted rather than tolerated silently: isolated bad batches are worth
    # skipping, a rising count means the run is diverging and should stop.
    nonfinite_g = 0
    nonfinite_d = 0
    lecam_rejected = 0

    for step in range(start_step + 1, args.steps + 1):
        model.train()

        # Dynamic channel noise/fading ramp
        if step <= args.clean_warmup:
            channel_mix = 0.0
        elif step <= (args.clean_warmup + args.channel_ramp):
            channel_mix = float(step - args.clean_warmup) / float(args.channel_ramp)
        else:
            channel_mix = 1.0

        # Fetch batch
        video = stream_pipeline.get_batch(args.batch, device=device)

        # The discriminator updates on whole accumulation groups, 1 in --d-every,
        # so it learns more slowly than the generator rather than outrunning it.
        # Resolved before the forward pass because it decides whether the clean
        # reconstruction below is needed at all.
        disc_active = discriminator is not None and step > args.disc_warmup
        disc_training = disc_active and ((step - 1) // args.accum) % args.d_every == 0

        # Forward Pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=args.amp):
            clean_z = model.encoder(video)

            if channel_mix > 0.0:
                noisy_ch_z, weights_ch = channel(clean_z.float())
                noisy_ch_z = noisy_ch_z.to(clean_z.dtype)
                weights_ch = weights_ch.to(clean_z.dtype)
                noisy_z = (1.0 - channel_mix) * clean_z + channel_mix * noisy_ch_z
                weights = (1.0 - channel_mix) * torch.ones_like(weights_ch) + channel_mix * weights_ch
            else:
                noisy_z = clean_z
                weights = torch.ones_like(clean_z)

            out_shape = (video.shape[2], video.shape[3], video.shape[4])
            recon = model.decoder(noisy_z, weights, output_shape=out_shape)

            # Clean-latent render is either a detached consistency target or,
            # when requested, a jointly optimized fidelity anchor that prevents
            # channel adaptation from buying robustness by softening the codec.
            recon_clean = None
            if channel_mix > 0.0 and args.clean_anchor_weight > 0.0:
                recon_clean = model.decoder(
                    clean_z, torch.ones_like(clean_z), output_shape=out_shape
                )
            elif channel_mix > 0.0 and args.consistency_weight > 0.0:
                with torch.no_grad():
                    recon_clean = model.decoder(
                        clean_z.detach(), torch.ones_like(clean_z), output_shape=out_shape
                    )

            # How much of this step's latent survived, measured rather than
            # assumed: both tensors are in hand, so this covers AWGN,
            # truncation, erasure and fading in one number. The channel's own
            # `weights` cannot serve here — in stage 1 it is a 0/1
            # truncation/erasure mask and says nothing about the AWGN level,
            # which is the axis the noisy eval cells actually vary.
            # conf -> 1 on a clean latent, 0.5 at 0 dB, 0 as the latent is
            # destroyed, so the realism prior fades out exactly where it would
            # otherwise be inventing detail the latent no longer carries.
            with torch.no_grad():
                sig_p = clean_z.float().pow(2).mean()
                err_p = (noisy_z.float() - clean_z.float()).pow(2).mean().clamp_min(1e-8)
                latent_snr = sig_p / err_p
                adv_conf = (latent_snr / (1.0 + latent_snr)).pow(args.adv_confidence_power)

        # 1. Discriminator Optimization (if enabled)
        loss_d = torch.tensor(0.0, device=device)
        loss_lecam = torch.tensor(0.0, device=device)
        score_real = torch.tensor(1.0, device=device)
        score_fake = torch.tensor(0.0, device=device)
        if disc_training:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=args.amp):
                pred_real, feats_real = discriminator(video.detach())
                pred_fake, feats_fake = discriminator(recon.detach())
                loss_d_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
                loss_d_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                if args.lecam_weight > 0.0:
                    # Belt and braces: `update` refuses non-finite batches, and
                    # this recovers state that went bad anyway -- through a
                    # resumed checkpoint, or arithmetic on an extreme score.
                    if lecam.is_poisoned():
                        lecam.reset()
                        print(
                            f"WARNING: step {step} LeCAM state non-finite; reset "
                            "to re-seed from the next clean batch",
                            flush=True,
                        )
                    loss_lecam = lecam.regularizer(pred_real, pred_fake)
                    loss_d = loss_d + args.lecam_weight * loss_lecam
                if not lecam.update(pred_real, pred_fake):
                    lecam_rejected += 1
                score_real = pred_real.mean().detach()
                score_fake = pred_fake.mean().detach()
            last_loss_d = loss_d.item()
            last_disc_step = step

            if torch.isfinite(loss_d):
                (loss_d / args.accum).backward()
            else:
                nonfinite_d += 1
                print(
                    f"WARNING: step {step} non-finite discriminator loss, "
                    f"skipping its backward ({nonfinite_d} so far, "
                    f"{lecam_rejected} LeCAM batches rejected)",
                    flush=True,
                )
                optimizer_d.zero_grad(set_to_none=True)
                if nonfinite_d > args.max_nonfinite:
                    raise SystemExit(
                        f"aborting after {nonfinite_d} non-finite discriminator "
                        f"steps (--max-nonfinite {args.max_nonfinite}). The "
                        "critic cannot compute a finite loss, so it has stopped "
                        "training while the generator carries on -- which is "
                        "silent unless this aborts."
                    )
            if step % args.accum == 0:
                d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                if torch.isfinite(d_norm):
                    optimizer_d.step()
                else:
                    nonfinite_d += 1
                    print(
                        f"WARNING: step {step} non-finite discriminator gradient "
                        f"norm, skipping its update ({nonfinite_d} so far)",
                        flush=True,
                    )
                optimizer_d.zero_grad(set_to_none=True)
                lr_scheduler_d.step()

        # 2. Generator Optimization
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=args.amp):
            loss_mse = F.mse_loss(recon, video)
            loss_l1 = (recon - video).abs().mean()
            loss_grad = spatial_gradient_loss(recon, video)
            loss_temporal = temporal_delta_loss(recon, video)
            loss_temporal_accel = temporal_acceleration_loss(recon, video)
            loss_temporal_energy = temporal_energy_loss(recon, video)
            loss_temporal_cosine = temporal_cosine_loss(recon, video)

            loss_dwt = torch.tensor(0.0, device=device)
            if args.dwt_weight > 0.0:
                loss_dwt = dwt3d_loss(recon, video, levels=args.dwt_levels)

            # Perceptual distance is scored on the render that actually goes
            # over the air. It is a *reference* metric against the target, so
            # unlike the adversarial term it cannot be satisfied by inventing
            # plausible detail, and it is the only term that optimizes
            # perceptual quality of the corrupted render.
            loss_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None:
                loss_lpips = lpips_fn(recon, video)

            loss_temporal_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None and args.temporal_lpips_weight > 0.0:
                recon_delta = (0.5 * (recon[:, :, 1:] - recon[:, :, :-1]) + 0.5).clamp(0, 1)
                video_delta = (0.5 * (video[:, :, 1:] - video[:, :, :-1]) + 0.5).clamp(0, 1)
                loss_temporal_lpips = lpips_fn(recon_delta, video_delta)

            # Decoder insensitivity to channel corruption: the noisy-latent
            # reconstruction should land where the clean-latent one did. The
            # target is detached so the term is satisfied by pulling the noisy
            # render up, never by degrading the clean one to meet it halfway.
            loss_consistency = torch.tensor(0.0, device=device)
            if recon_clean is not None:
                loss_consistency = F.l1_loss(recon, recon_clean.detach())

            loss_clean_anchor = torch.tensor(0.0, device=device)
            if recon_clean is not None and args.clean_anchor_weight > 0.0:
                loss_clean_anchor = (
                    args.mse_weight * F.mse_loss(recon_clean, video)
                    + args.l1_weight * F.l1_loss(recon_clean, video)
                    + args.grad_weight * spatial_gradient_loss(recon_clean, video)
                    + args.temporal_weight * temporal_delta_loss(recon_clean, video)
                    + args.temporal_accel_weight
                    * temporal_acceleration_loss(recon_clean, video)
                )

            loss_adv = torch.tensor(0.0, device=device)
            loss_fm = torch.tensor(0.0, device=device)
            if disc_active:
                # The critic is frozen for the generator's pass. Without this
                # its parameters accumulate the generator's anti-critic
                # gradient, and because optimizer_d.zero_grad() runs right
                # after its step rather than before its backward, that sum is
                # still there when the discriminator next updates.
                for p in discriminator.parameters():
                    p.requires_grad_(False)
                pred_fake_g, feats_fake_g = discriminator(recon)
                _, feats_real_g = discriminator(video.detach())
                loss_adv = F.mse_loss(pred_fake_g, torch.ones_like(pred_fake_g))
                loss_fm = sum(
                    F.l1_loss(f_fake, f_real.detach()) for f_fake, f_real in zip(feats_fake_g, feats_real_g)
                ) / len(feats_fake_g)
                for p in discriminator.parameters():
                    p.requires_grad_(True)

            total_loss = (
                args.mse_weight * loss_mse
                + args.l1_weight * loss_l1
                + args.dwt_weight * loss_dwt
                + args.grad_weight * loss_grad
                + args.temporal_weight * loss_temporal
                + args.temporal_accel_weight * loss_temporal_accel
                + args.temporal_energy_weight * loss_temporal_energy
                + args.temporal_cosine_weight * loss_temporal_cosine
                + args.lpips_weight * loss_lpips
                + args.temporal_lpips_weight * loss_temporal_lpips
                + args.consistency_weight * loss_consistency
                + args.clean_anchor_weight * loss_clean_anchor
                # Only the realism prior is confidence-scaled. Feature matching
                # is |phi_D(recon) - phi_D(target)|, a reference term against
                # the real frames, so it cannot be satisfied by invention and
                # is wanted at full strength however bad the latent is.
                + args.adv_weight * adv_conf * loss_adv
                + args.fm_weight * loss_fm
            )

        # One non-finite batch is otherwise fatal *and* silent. clip_grad_norm_
        # rescales by 1/total_norm, so a single NaN gradient makes the norm NaN
        # and multiplies every parameter's gradient by it -- the damage is
        # global after one step and permanent, which is how the first stage-2
        # attempt reported nan from step 1100 to the end. Skipping the batch
        # costs `accum` samples; not skipping costs the run. Naming the
        # offending term matters as much, since the console line reports every
        # term as nan once the weights are gone and says nothing about which
        # one went first.
        if not torch.isfinite(total_loss):
            offenders = [
                name
                for name, t in (
                    ("mse", loss_mse),
                    ("l1", loss_l1),
                    ("dwt", loss_dwt),
                    ("grad", loss_grad),
                    ("temporal", loss_temporal),
                    ("temporal_accel", loss_temporal_accel),
                    ("temporal_energy", loss_temporal_energy),
                    ("temporal_cosine", loss_temporal_cosine),
                    ("perceptual", loss_lpips),
                    ("temporal_perceptual", loss_temporal_lpips),
                    ("consistency", loss_consistency),
                    ("clean_anchor", loss_clean_anchor),
                    ("adv", loss_adv),
                    ("fm", loss_fm),
                    ("adv_conf", adv_conf),
                )
                if not torch.isfinite(t).all()
            ]
            nonfinite_g += 1
            print(
                f"WARNING: step {step} non-finite generator loss, skipping batch. "
                f"Offending terms: {', '.join(offenders) or 'none (recon or input)'}. "
                f"video finite={bool(torch.isfinite(video).all())} "
                f"clean latent finite={bool(torch.isfinite(clean_z).all())} "
                f"noisy latent finite={bool(torch.isfinite(noisy_z).all())} "
                f"weights finite={bool(torch.isfinite(weights).all())} "
                f"recon finite={bool(torch.isfinite(recon).all())} "
                f"({nonfinite_g} so far)",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            if nonfinite_g > args.max_nonfinite:
                raise SystemExit(
                    f"aborting after {nonfinite_g} non-finite generator steps "
                    f"(--max-nonfinite {args.max_nonfinite}); the run is diverging "
                    "rather than hitting isolated bad batches"
                )
            # Skipping the backward leaves the graph alive, since these names
            # stay bound until the next iteration rebinds them -- so the next
            # forward pass would hold two graphs and roughly double peak VRAM.
            # Measured as a 6.9 -> 12.4 GiB spike on the step after a skip.
            del (
                total_loss,
                recon,
                recon_clean,
                noisy_z,
                clean_z,
                weights,
                loss_mse,
                loss_l1,
                loss_dwt,
                loss_grad,
                loss_temporal,
                loss_temporal_accel,
                loss_lpips,
                loss_temporal_lpips,
                loss_consistency,
                loss_clean_anchor,
                loss_adv,
                loss_fm,
                adv_conf,
            )
            torch.cuda.empty_cache()
            continue

        (total_loss / args.accum).backward()
        if step % args.accum == 0:
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(g_norm):
                optimizer.step()
            else:
                nonfinite_g += 1
                print(
                    f"WARNING: step {step} non-finite generator gradient norm "
                    f"(loss was finite), skipping update ({nonfinite_g} so far)",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()


        # TensorBoard & Console Logging
        if step % args.tb_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            clean_mse = loss_mse.item()
            psnr = -10.0 * math.log10(max(1e-7, clean_mse))
            rate = (step - start_step) / max(1e-3, elapsed)
            current_lr = lr_scheduler.get_last_lr()[0]

            # Log to TensorBoard
            writer.add_scalar("train/loss_total", total_loss.item(), step)
            writer.add_scalar("train/loss_mse", loss_mse.item(), step)
            writer.add_scalar("train/loss_l1", loss_l1.item(), step)
            writer.add_scalar("train/loss_gradient", loss_grad.item(), step)
            writer.add_scalar("train/loss_dwt", loss_dwt.item(), step)
            writer.add_scalar("train/loss_temporal", loss_temporal.item(), step)
            writer.add_scalar("train/loss_temporal_accel", loss_temporal_accel.item(), step)
            writer.add_scalar("train/loss_perceptual", loss_lpips.item(), step)
            writer.add_scalar("train/loss_temporal_perceptual", loss_temporal_lpips.item(), step)
            writer.add_scalar("train/adv_confidence", adv_conf.item(), step)
            writer.add_scalar("train/latent_snr_db", 10.0 * math.log10(max(1e-8, latent_snr.item())), step)
            writer.add_scalar("train/loss_consistency", loss_consistency.item(), step)
            writer.add_scalar("train/loss_clean_anchor", loss_clean_anchor.item(), step)
            writer.add_scalar("train/loss_adv", loss_adv.item(), step)
            writer.add_scalar("train/loss_feature_matching", loss_fm.item(), step)
            # Only on steps the discriminator actually trained: on the other
            # three of every four these hold defaults, and plotting those would
            # draw a sawtooth to zero that looks like a collapsing discriminator.
            if disc_training:
                writer.add_scalar("train/loss_disc", loss_d.item(), step)
                writer.add_scalar("train/loss_lecam", loss_lecam.item(), step)
                writer.add_scalar("train/disc_real_score", score_real.item(), step)
                writer.add_scalar("train/disc_fake_score", score_fake.item(), step)
            writer.add_scalar("train/psnr_db", psnr, step)
            writer.add_scalar("train/channel_mix", channel_mix, step)
            writer.add_scalar("train/rate_steps_per_sec", rate, step)
            writer.add_scalar("train/learning_rate", current_lr, step)
            writer.add_scalar("train/cached_clips", len(stream_pipeline.cached_files), step)

            print(
                f"Step {step:6d}/{args.steps} [Mix: {channel_mix:4.2f}] | "
                f"MSE: {clean_mse:.6f} | PSNR: {psnr:.2f} dB | Perc: {loss_lpips.item():.4f} | "
                f"DWT: {loss_dwt.item():.4f} | Cons: {loss_consistency.item():.4f} | "
                f"Adv: {loss_adv.item():.4f}x{adv_conf.item():.2f} | "
                f"D: {last_loss_d:.4f}"
                f"{'' if last_disc_step == step else f'@{last_disc_step}'} | "
                f"Rate: {rate:.1f} steps/s | "
                f"LR: {current_lr:.2e} | Cache: {len(stream_pipeline.cached_files)} | "
                # Peak since the previous log line, not cumulative: a run that
                # OOMs 1800 steps after an eval gives no warning at all unless
                # the headroom is on screen, and the interval peak is what shows
                # the footprint creeping up.
                f"VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f}/"
                f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
                flush=True,
            )
            writer.add_scalar(
                "train/vram_peak_gib", torch.cuda.max_memory_allocated() / 2**30, step
            )
            torch.cuda.reset_peak_memory_stats()

        # Periodic Evaluation on 5 Held-Out Clips
        if step % args.eval_interval == 0:
            run_evaluation(
                model=model,
                eval_clips=eval_clips,
                mode_spec=mode_spec,
                step=step,
                out_dir=out_dir,
                writer=writer,
                device=device,
            )
            # Evaluation runs whole clips through the modem at shapes training
            # never allocates, and stage 2's FFTs take cuFFT workspace from
            # outside the caching allocator's pool. Returning those blocks keeps
            # the eval from ratcheting the steady-state footprint upward -- a
            # stage-2 run died of OOM 1800 steps after an eval with ~2 GiB
            # nominally free.
            torch.cuda.empty_cache()

        # Periodic Checkpoint
        if step % args.checkpoint_interval == 0 or step == args.steps:
            ckpt_path = out_dir / f"checkpoint_step_{step:06d}.pt"
            state = {
                "step": step,
                "mode": mode_spec.name,
                "stage": args.stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }
            if discriminator is not None:
                state["discriminator_state_dict"] = discriminator.state_dict()
                state["optimizer_d_state_dict"] = optimizer_d.state_dict()
            torch.save(state, ckpt_path)
            torch.save(state, out_dir / "checkpoint.pt")
            print(f"--> Saved checkpoint to {ckpt_path}", flush=True)

    stream_pipeline.close()
    writer.close()
    print(f"\n1-Epoch training complete! Final checkpoint saved to {out_dir / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
