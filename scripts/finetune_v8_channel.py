#!/usr/bin/env python3
"""Fine-tune the V8 release checkpoint with real V8 modem draws in the batch.

Warm start: ``models/v8-hf3k-face-gan.pt``. The objective is the release's
stage-2 generator loss with its weights (pixel, DWT, gradient, temporal, VGG
perceptual, YuNet region/detail/contrast, face-crop perceptual, consistency and
clean anchor). The face critic and PatchGAN are not trained here.

Each step, most rows go through the differentiable waveform channel
(``AETVWaveformChannel``, release sampler: 0-16 dB, p_fading 0.5, measured-path
0.4). ``--real-rows`` rows instead go through the real V8 OFDM modem and
``emulate`` with a training-only fade seed. The real channel error is added
straight-through, so the decoder sees true V8 demodulation with the modem's own
confidence while the encoder still receives a gradient.

Data and selection never touch the 64 eval clips: training uses the training
pool after ``--select-clips``; checkpoints are chosen on the first
``--select-clips`` pool clips through V8 + ``mpp12`` with fade seed base 7300.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetv import AETV_MODES, AETVChannelConfig, AETVWaveformChannel  # noqa: E402
from aetv.attention import (  # noqa: E402
    RegionAttentionTeacher,
    face_crop_grid,
    region_contrast_loss,
    region_detail_loss,
    region_gradient_loss,
    region_reconstruction_loss,
    sample_face_crops,
)
from aetv.models import MultiLayerVGGPerceptualLoss  # noqa: E402
from aetv.shared_eval import (  # noqa: E402
    AutoencoderAdapter,
    FaceMasks,
    load_clips,
    modem_exchange,
    paired,
    score_clip,
    summarize,
    training_pool,
)
from train import (  # noqa: E402
    dwt3d_loss,
    spatial_gradient_loss,
    temporal_acceleration_loss,
    temporal_cosine_loss,
    temporal_delta_loss,
    temporal_energy_loss,
)

RELEASE = "models/v8-hf3k-face-gan.pt"
REAL_PROFILES = ("mpp12", "mpp12", "mpp12", "mpp6", "awgn12", "clean")
REAL_PROFILES_CHANNEL_ONLY = ("mpp12",)


def region_terms(recon, video, mask, boost):
    return (
        region_reconstruction_loss(recon, video, mask, boost) + 0.5 * region_gradient_loss(recon, video, mask, boost),
        region_detail_loss(recon, video, mask, boost),
        region_contrast_loss(recon, video, mask, boost),
    )


def pixel_mse(recon, video, vvc, weight, distill):
    if weight is None:
        weight = torch.ones_like(recon[:, :1])
    source = (weight * (recon - video).square()).mean()
    if vvc is None or distill <= 0.0:
        return source
    return (1.0 - distill) * source + distill * (weight * (recon - vvc).square()).mean()


def pixel_l1(recon, video, weight):
    if weight is None:
        return (recon - video).abs().mean()
    return (weight * (recon - video).abs()).mean()


def face_crop_loss(recon, video, boxes, geometry, mse_w, l1_w, grad_w, size: int = 64):
    """Per-pixel reconstruction loss on 64x64 face crops, so the face counts as a full frame's worth of loss."""
    src, index = geometry.frame_crops(video, boxes)
    if index.numel() == 0:
        return recon.new_zeros(())
    b, c, t, h, w = recon.shape
    from aetv.face_geometry import crop_faces, fan_boxes

    flat = fan_boxes(boxes.reshape(b * t, 4).to(recon.device, recon.dtype))[index]
    rec = crop_faces(recon.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)[index], flat, size)
    src = F.interpolate(src, size=(size, size), mode="area")
    grad = ((rec[..., 1:, :] - rec[..., :-1, :]) - (src[..., 1:, :] - src[..., :-1, :])).abs().mean() + \
        ((rec[..., :, 1:] - rec[..., :, :-1]) - (src[..., :, 1:] - src[..., :, :-1])).abs().mean()
    return mse_w * F.mse_loss(rec, src) + l1_w * (rec - src).abs().mean() + grad_w * grad


def box_mask(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, T, 4) xywh boxes (NaN = no face) -> (B, 1, T, H, W) float mask."""
    ys = torch.arange(height, device=boxes.device).view(1, 1, height, 1) + 0.5
    xs = torch.arange(width, device=boxes.device).view(1, 1, 1, width) + 0.5
    x, y, w, h = (boxes[..., k].unsqueeze(-1).unsqueeze(-1) for k in range(4))
    inside = (xs >= x) & (xs <= x + w) & (ys >= y) & (ys <= y + h)
    return inside.float().unsqueeze(1)


def real_channel(wire: np.ndarray, seed: int, rng: random.Random, profiles=REAL_PROFILES) -> tuple[np.ndarray, np.ndarray]:
    profile = rng.choice(profiles)
    received, confidence, _ = modem_exchange(wire, "V8", seed, profile)
    return received, confidence


@torch.no_grad()
def select_score(adapter, clips, names, faces, lpips_metric, device, seed0: int, geometry=None) -> dict:
    adapter.model.eval()
    records = []
    for index in range(clips.shape[0]):
        mask = faces.mask(names[index], clips[index])
        boxes = faces.boxes(names[index], clips[index]) if geometry is not None else None
        records.append(score_clip(adapter, clips[index], index, device, {"ft": 1.0}, seed0=seed0,
                                  face_mask=mask, face_boxes=boxes, face_geometry=geometry,
                                  lpips_metric=lpips_metric)["ft"])
    out = {"per_clip": {
        "mpp12": [r.rows["mpp12"] for r in records],
        "clean": [r.rows["clean"] for r in records],
        "modem_clean": [r.rows["modem_clean"] for r in records],
        "awgn6": [r.rows["awgn6"] for r in records],
        "lpips_clean": [r.lpips["clean"] for r in records],
        "lpips_mpp12": [r.lpips["mpp12"] for r in records],
        "face_mpp12": [r.face.get("mpp12") for r in records],
        "nme_mpp12": [r.nme.get("mpp12") for r in records],
        "face_lpips_mpp12": [r.face_lpips.get("mpp12") for r in records],
    }}
    out["summary"] = {k: summarize(v) for k, v in out["per_clip"].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/v8-channel-ft")
    ap.add_argument("--init", default=RELEASE)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--kill-step", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--real-rows", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--select-clips", type=int, default=24)
    ap.add_argument("--select-seed0", type=int, default=7300)
    ap.add_argument("--train-seed0", type=int, default=1_000_000)
    ap.add_argument("--kill-mpp12", type=float, default=0.15, help="paired mpp12 gain required at --kill-step")
    ap.add_argument("--kill-clean", type=float, default=-0.10, help="paired clean change allowed at --kill-step")
    ap.add_argument(
        "--channel-only",
        action="store_true",
        help=(
            "every row through the real V8 modem on mpp12; no clean render, clean anchor, consistency "
            "or clean face term; selection and the kill check use mpp12 PSNR only"
        ),
    )
    ap.add_argument("--seed", type=int, default=20260925)
    # Release (face-GAN) generator weights.
    ap.add_argument("--mse-weight", type=float, default=0.25)
    ap.add_argument("--l1-weight", type=float, default=0.8)
    ap.add_argument("--dwt-weight", type=float, default=3.0)
    ap.add_argument("--grad-weight", type=float, default=1.5)
    ap.add_argument("--temporal-weight", type=float, default=1.5)
    ap.add_argument("--temporal-accel-weight", type=float, default=0.3)
    ap.add_argument("--temporal-energy-weight", type=float, default=2.0)
    ap.add_argument("--temporal-cosine-weight", type=float, default=0.2)
    ap.add_argument("--lpips-weight", type=float, default=0.18)
    ap.add_argument("--temporal-lpips-weight", type=float, default=0.1)
    ap.add_argument("--face-perceptual-weight", type=float, default=0.25)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
    ap.add_argument("--clean-anchor-weight", type=float, default=1.0)
    ap.add_argument("--region-weight", type=float, default=1.5)
    ap.add_argument("--detail-weight", type=float, default=3.0)
    ap.add_argument("--contrast-weight", type=float, default=2.5)
    ap.add_argument("--region-boost", type=float, default=12.0)
    ap.add_argument("--face-model", default="data/teachers/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--teacher-dir", help="VVC teacher npz files from scripts/precompute_vvc_teacher.py")
    ap.add_argument("--importance-weight", type=float, default=0.0,
                    help="alpha: pixel MSE/L1 weight is 1 + alpha * (VVC bit-allocation map - 1)")
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="lambda: pixel MSE target mixes (1 - lambda) * source + lambda * VVC decode")
    ap.add_argument("--aligned-windows", action="store_true",
                    help="train only on the two V8-aligned 6-frame GOPs of each clip (implied by --teacher-dir)")
    ap.add_argument("--stop-after-kill", action="store_true", help="stop at --kill-step whatever the verdict")
    ap.add_argument("--face-goal", action="store_true",
                    help=("select and kill-check on facial landmark error (2D-FAN NME) on pool face clips, "
                          "with face PSNR not falling and whole-frame mpp12 within --kill-mpp12-floor"))
    ap.add_argument("--face-roi-weight", type=float, default=0.0, help="extra pixel MSE inside YuNet face boxes")
    ap.add_argument("--landmark-weight", type=float, default=0.0,
                    help="MSE between frozen 2D-FAN heatmaps of channel output and source face crops")
    ap.add_argument("--face-lowfreq-weight", type=float, default=0.0,
                    help="MSE of 5x5-blurred channel output vs source inside face boxes")
    ap.add_argument("--face-oversample", type=float, default=0.5, help="share of rows drawn from face clips")
    ap.add_argument("--select-face-extra", type=int, default=16,
                    help="extra pool face clips moved from training into selection (face goal)")
    ap.add_argument("--landmark-crops", type=int, default=8, help="face crops per step for the landmark loss")
    ap.add_argument("--kill-mpp12-floor", type=float, default=-0.10)
    ap.add_argument("--face-priority", action="store_true",
                    help=("feed the encoder a transmitter-side YuNet face mask (4th input plane) and kill-check / "
                          "select on landmark error, face PSNR and face-crop LPIPS; whole-frame mpp12 only stops a "
                          "run below --whole-frame-stop"))
    ap.add_argument("--face-loss-weight", type=float, default=0.0,
                    help=("weight k of the per-pixel reconstruction loss (MSE, L1, gradient) on 64x64 face crops; "
                          "the background keeps weight 1 on the whole-frame loss"))
    ap.add_argument("--whole-frame-stop", type=float, default=-1.5)
    ap.add_argument("--mask-lr-scale", type=float, default=30.0, help="learning-rate multiplier for the face-mask stem")
    args = ap.parse_args()
    if args.face_priority:
        args.face_goal = True
    if args.teacher_dir:
        args.aligned_windows = True
    if args.channel_only:
        args.real_rows = args.batch
        args.kill_clean = None
        args.clean_anchor_weight = 0.0
        args.consistency_weight = 0.0
    profiles = REAL_PROFILES_CHANNEL_ONLY if args.channel_only else REAL_PROFILES

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")

    pool = training_pool()
    select_paths, train_paths = pool[: args.select_clips], pool[args.select_clips :]
    faces = FaceMasks(args.face_model)
    box_of = {}
    if args.face_goal:
        cache = Path("data/face_boxes_pool.pt")
        box_of = torch.load(cache) if cache.exists() else {}
        missing = [p for p in pool if p.name not in box_of]
        for n, path in enumerate(missing):
            box_of[path.name] = faces.boxes(path.name, torch.load(path, map_location="cpu", weights_only=False))
            if (n + 1) % 200 == 0:
                print(f"face boxes {n + 1}/{len(missing)}", flush=True)
        if missing:
            torch.save(box_of, cache)
        extra = [p for p in train_paths if torch.isfinite(box_of[p.name]).any()][: args.select_face_extra]
        extra_names = {p.name for p in extra}
        select_paths = select_paths + extra
        train_paths = [p for p in train_paths if p.name not in extra_names]
        n_face = sum(bool(torch.isfinite(box_of[p.name]).any()) for p in select_paths)
        print(f"selection clips {len(select_paths)}, of which face clips {n_face}", flush=True)
    select_clips = load_clips(select_paths)
    select_names = [p.name for p in select_paths]
    print(f"train clips {len(train_paths)}, selection clips {len(select_paths)} (pool only)", flush=True)
    train_clips = load_clips(train_paths)
    teacher_decode = teacher_map = None
    if args.teacher_dir:
        decodes, maps = [], []
        for path in train_paths:
            data = np.load(Path(args.teacher_dir) / (path.stem + ".npz"))
            decodes.append(torch.from_numpy(data["decode"]))
            maps.append(torch.from_numpy(data["importance"]))
        teacher_decode, teacher_map = torch.stack(decodes), torch.stack(maps)
        print(f"VVC teacher loaded for {len(decodes)} clips", flush=True)

    adapter = AutoencoderAdapter("v8-ft", args.init, "V8", device)
    model = adapter.model
    if args.face_priority:
        from aetv.face_priority import add_mask_input, encode_with_mask

        add_mask_input(model)
        adapter.face_priority = True
        adapter.detector = FaceMasks(args.face_model)
    spec = AETV_MODES["V8"]
    gop = spec.gop_frames
    channel = AETVWaveformChannel(band=spec.band, cfg=AETVChannelConfig(
        snr_db_range=(0.0, 16.0), p_fading=0.5, p_measured_path=0.4, p_truncate=0.0, erasure_rate_max=0.0,
    )).to(device)
    vgg = MultiLayerVGGPerceptualLoss().to(device).eval()
    teacher = RegionAttentionTeacher(args.face_model, device, face_score_threshold=0.72)
    geometry = None
    train_boxes = face_rows = None
    if args.face_goal:
        from aetv.face_geometry import FaceGeometry

        geometry = FaceGeometry(device)
        train_boxes = torch.stack([box_of[p.name] for p in train_paths])
        face_rows = [i for i, p in enumerate(train_paths) if torch.isfinite(box_of[p.name]).any()]
        print(f"training face clips {len(face_rows)} of {len(train_paths)}", flush=True)
    import lpips

    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    if args.face_priority:
        from aetv.face_priority import mask_parameters

        mask_ids = {id(q) for q in mask_parameters(model)}
        groups = [{"params": [q for q in model.parameters() if id(q) not in mask_ids], "lr_scale": 1.0},
                  {"params": mask_parameters(model), "lr_scale": args.mask_lr_scale}]
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))

    baseline = select_score(adapter, select_clips, select_names, faces, lpips_metric, device, args.select_seed0,
                            geometry)
    s = baseline["summary"]
    print(f"step 0 (release): pool mpp12 {s['mpp12']['mean']:.3f} clean {s['clean']['mean']:.3f} "
          f"lpips_mpp12 {s['lpips_mpp12']['mean']:.4f}", flush=True)
    history = [{"step": 0, **{k: v["mean"] for k, v in s.items()}}]
    (out / "select_step0.json").write_text(json.dumps(baseline))
    best = {"step": 0, "mpp12": s["mpp12"]["mean"], "nme_mpp12": s["nme_mpp12"]["mean"]}

    rng = random.Random(args.seed)
    pool_exec = ThreadPoolExecutor(max_workers=max(1, args.real_rows))
    started = time.time()
    real_seed = args.train_seed0
    verdict = None
    for step in range(1, args.steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step) * group.get("lr_scale", 1.0)
        if face_rows:
            idx = [rng.choice(face_rows) if rng.random() < args.face_oversample else rng.randrange(train_clips.shape[0])
                   for _ in range(args.batch)]
        else:
            idx = [rng.randrange(train_clips.shape[0]) for _ in range(args.batch)]
        if args.aligned_windows:
            starts = [rng.choice((0, gop)) for _ in idx]
        else:
            starts = [rng.randrange(0, 12 - gop + 1) for _ in idx]
        video = torch.stack([train_clips[i, :, s0 : s0 + gop] for i, s0 in zip(idx, starts)]).to(device).float().div(255)
        vvc = weight = None
        if teacher_decode is not None:
            vvc = torch.stack([teacher_decode[i, :, s0 : s0 + gop] for i, s0 in zip(idx, starts)]).to(device).float().div(255)
            imp = torch.stack([teacher_map[i, s0 : s0 + gop] for i, s0 in zip(idx, starts)]).to(device).float().div(32)
            weight = (1.0 + args.importance_weight * (imp - 1.0)).clamp_min(0.0).unsqueeze(1)
            weight = weight / weight.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        boxes = None
        if train_boxes is not None:
            boxes = torch.stack([train_boxes[i, s0 : s0 + gop] for i, s0 in zip(idx, starts)]).to(device)
        if rng.random() < 0.5:
            video = video.flip(-1)
            if vvc is not None:
                vvc, weight = vvc.flip(-1), weight.flip(-1)
            if boxes is not None:
                boxes = boxes.clone()
                boxes[..., 0] = spec.width - boxes[..., 0] - boxes[..., 2]
        with torch.no_grad():
            attention_mask, face_clips = teacher(video)
        face_grid, face_indices = (None, torch.empty(0, dtype=torch.long, device=device))
        face_target = None
        if bool(face_clips.any()):
            face_grid, face_indices = face_crop_grid(attention_mask, face_clips, crop_size=64)
            face_target = sample_face_crops(video, face_grid, face_indices)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            clean_z = encode_with_mask(model, video, boxes) if args.face_priority else model.encoder(video)
            if args.real_rows < args.batch:
                noisy_z, weights = channel(clean_z.float())
            else:
                noisy_z, weights = clean_z.float(), torch.ones_like(clean_z, dtype=torch.float32)
            if args.real_rows > 0:
                wires = clean_z[: args.real_rows].detach().float().cpu().numpy()
                seeds = list(range(real_seed, real_seed + args.real_rows))
                real_seed += args.real_rows
                rngs = [random.Random(sd) for sd in seeds]
                results = list(pool_exec.map(real_channel, wires, seeds, rngs, [profiles] * len(seeds)))
                rx = torch.from_numpy(np.stack([r for r, _ in results])).to(device)
                cf = torch.from_numpy(np.stack([c for _, c in results])).to(device)
                real_z = clean_z[: args.real_rows].float() + (rx - clean_z[: args.real_rows].float()).detach()
                noisy_z = torch.cat([real_z, noisy_z[args.real_rows :]], 0)
                weights = torch.cat([cf, weights[args.real_rows :]], 0)
            shape = (gop, spec.height, spec.width)
            recon = model.decoder(noisy_z.to(clean_z.dtype), weights.to(clean_z.dtype), output_shape=shape)
            recon_clean = None
            if not args.channel_only:
                recon_clean = model.decoder(clean_z, torch.ones_like(clean_z), output_shape=shape)
            with torch.no_grad():
                err = (noisy_z.float() - clean_z.float()).pow(2).mean().clamp_min(1e-8)
                snr = clean_z.float().pow(2).mean() / err
                conf = snr / (1 + snr)

            loss_region, loss_detail, loss_contrast = region_terms(recon, video, attention_mask, args.region_boost)
            anchor = torch.zeros((), device=device)
            if recon_clean is not None:
                anchor = (
                    args.mse_weight * F.mse_loss(recon_clean, video)
                    + args.l1_weight * F.l1_loss(recon_clean, video)
                    + args.grad_weight * spatial_gradient_loss(recon_clean, video)
                    + args.temporal_weight * temporal_delta_loss(recon_clean, video)
                    + args.temporal_accel_weight * temporal_acceleration_loss(recon_clean, video)
                )
                a_region, a_detail, a_contrast = region_terms(recon_clean, video, attention_mask, args.region_boost)
                anchor = anchor + args.region_weight * a_region + args.detail_weight * a_detail + args.contrast_weight * a_contrast
            recon_delta = (0.5 * (recon[:, :, 1:] - recon[:, :, :-1]) + 0.5).clamp(0, 1)
            video_delta = (0.5 * (video[:, :, 1:] - video[:, :, :-1]) + 0.5).clamp(0, 1)
            loss_face = torch.zeros((), device=device)
            if face_target is not None and recon_clean is not None:
                loss_face = 0.5 * (
                    vgg(sample_face_crops(recon_clean, face_grid, face_indices), face_target)
                    + vgg(sample_face_crops(recon, face_grid, face_indices), face_target)
                )
            elif face_target is not None:
                loss_face = vgg(sample_face_crops(recon, face_grid, face_indices), face_target)
            ramp = 0.5 + 0.5 * conf
            loss = (
                args.mse_weight * pixel_mse(recon, video, vvc, weight, args.distill_weight)
                + args.l1_weight * pixel_l1(recon, video, weight)
                + args.dwt_weight * dwt3d_loss(recon, video, levels=3)
                + args.grad_weight * spatial_gradient_loss(recon, video)
                + args.temporal_weight * temporal_delta_loss(recon, video)
                + args.temporal_accel_weight * temporal_acceleration_loss(recon, video)
                + args.temporal_energy_weight * temporal_energy_loss(recon, video)
                + args.temporal_cosine_weight * temporal_cosine_loss(recon, video)
                + ramp * (args.region_weight * loss_region + args.detail_weight * loss_detail
                          + args.contrast_weight * loss_contrast)
                + args.lpips_weight * vgg(recon, video)
                + args.temporal_lpips_weight * vgg(recon_delta, video_delta)
                + (args.consistency_weight * F.l1_loss(recon, recon_clean.detach()) if recon_clean is not None else 0.0)
                + args.clean_anchor_weight * anchor
                + args.face_perceptual_weight * loss_face
            )
            if boxes is not None:
                fmask = box_mask(boxes, spec.height, spec.width)
                area = fmask.sum().clamp_min(1.0) * 3
                if args.face_roi_weight > 0:
                    loss = loss + args.face_roi_weight * ((recon - video).square() * fmask).sum() / area
                if args.face_lowfreq_weight > 0:
                    blur_r = F.avg_pool3d(recon, (1, 5, 5), 1, (0, 2, 2), count_include_pad=False)
                    blur_v = F.avg_pool3d(video, (1, 5, 5), 1, (0, 2, 2), count_include_pad=False)
                    loss = loss + args.face_lowfreq_weight * ((blur_r - blur_v).square() * fmask).sum() / area
                if args.landmark_weight > 0:
                    loss = loss + args.landmark_weight * geometry.heatmap_loss(
                        recon, video, boxes, max_crops=args.landmark_crops)
                if args.face_priority and args.face_loss_weight > 0:
                    loss = loss + args.face_loss_weight * face_crop_loss(
                        recon, video, boxes, geometry, args.mse_weight, args.l1_weight, args.grad_weight)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            rate = step / (time.time() - started)
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e} {rate:.2f} steps/s", flush=True)
            log.write(json.dumps({"step": step, "loss": loss.item(), "lr": lr_at(step)}) + "\n")
            log.flush()

        if step % args.eval_interval == 0 or step == args.kill_step:
            scored = select_score(adapter, select_clips, select_names, faces, lpips_metric, device, args.select_seed0,
                                  geometry)
            base = baseline["per_clip"]
            deltas = {k: paired(scored["per_clip"][k], base[k]) for k in base}
            s = scored["summary"]
            record = {"step": step, **{k: v["mean"] for k, v in s.items()},
                      **{f"d_{k}": v["mean"] for k, v in deltas.items()},
                      **{f"se_{k}": v["se"] for k, v in deltas.items()}}
            history.append(record)
            log.write(json.dumps({"select": record}) + "\n")
            log.flush()
            print(f"select step {step}: mpp12 {s['mpp12']['mean']:.3f} "
                  f"(Δ {deltas['mpp12']['mean']:+.3f} ± {deltas['mpp12']['se']:.3f}) "
                  f"clean Δ {deltas['clean']['mean']:+.3f} lpips_mpp12 Δ {deltas['lpips_mpp12']['mean']:+.4f}", flush=True)
            state = {"mode": "V8", "stage": 2, "step": step, "args": {**adapter_args(args.init), "finetune": vars(args)},
                     "model_state_dict": model.state_dict(), "source_run": str(out), "init_checkpoint": args.init,
                     "select": record, "face_priority_input": bool(args.face_priority)}
            clean_ok = args.kill_clean is None or deltas["clean"]["mean"] >= args.kill_clean
            if args.face_priority:
                print(f"  face-priority: face_lpips_mpp12 {s['face_lpips_mpp12']['mean']:.4f} (Δ "
                      f"{deltas['face_lpips_mpp12']['mean']:+.4f} ± {deltas['face_lpips_mpp12']['se']:.4f})", flush=True)
            if args.face_goal:
                print(f"  face: nme_mpp12 {s['nme_mpp12']['mean']:.3f} (Δ {deltas['nme_mpp12']['mean']:+.3f} ± "
                      f"{deltas['nme_mpp12']['se']:.3f}, n={deltas['nme_mpp12']['n']}) face_mpp12 Δ "
                      f"{deltas['face_mpp12']['mean']:+.3f} ± {deltas['face_mpp12']['se']:.3f}", flush=True)
                guards = deltas["face_mpp12"]["mean"] >= 0.0 and deltas["mpp12"]["mean"] >= args.kill_mpp12_floor
                improved = guards and s["nme_mpp12"]["mean"] < best["nme_mpp12"]
                if args.face_priority:
                    guards = (deltas["nme_mpp12"]["mean"] <= 0.0 and deltas["face_lpips_mpp12"]["mean"] <= 0.0
                              and deltas["mpp12"]["mean"] >= args.whole_frame_stop)
                    improved = guards and s["face_mpp12"]["mean"] > best.get("face_mpp12", -1e9)
            else:
                improved = s["mpp12"]["mean"] > best["mpp12"] and clean_ok
            if improved:
                best = {"step": step, "mpp12": s["mpp12"]["mean"], "nme_mpp12": s["nme_mpp12"]["mean"],
                        "face_mpp12": s["face_mpp12"]["mean"]}
                torch.save(state, out / "best.pt")
                print(f"  new best at step {step}", flush=True)
            torch.save(state, out / "latest.pt")
            if step == args.kill_step:
                ok = deltas["mpp12"]["mean"] >= args.kill_mpp12
                if args.face_priority:
                    def clear(key, sign):
                        d = deltas[key]
                        return sign * d["mean"] > 0 and abs(d["mean"]) > 2 * d["se"]
                    ok = (clear("nme_mpp12", -1) and clear("face_mpp12", 1) and clear("face_lpips_mpp12", -1)
                          and deltas["mpp12"]["mean"] >= args.whole_frame_stop)
                elif args.face_goal:
                    d = deltas["nme_mpp12"]
                    ok = (d["mean"] < 0 and -d["mean"] > 2 * d["se"] and deltas["face_mpp12"]["mean"] >= 0.0
                          and deltas["mpp12"]["mean"] >= args.kill_mpp12_floor)
                elif not args.channel_only:
                    ok = ok and clean_ok and deltas["lpips_mpp12"]["mean"] <= 0.0
                verdict = {"step": step, "passed": ok, "mpp12": deltas["mpp12"], "clean": deltas["clean"],
                           "lpips_mpp12": deltas["lpips_mpp12"], "face_mpp12": deltas["face_mpp12"],
                           "nme_mpp12": deltas["nme_mpp12"], "face_lpips_mpp12": deltas["face_lpips_mpp12"]}
                (out / "kill_check.json").write_text(json.dumps(verdict, indent=1))
                print(f"KILL CHECK {'PASSED' if ok else 'FAILED'} at step {step}", flush=True)
                if not ok or args.stop_after_kill:
                    break
    (out / "history.json").write_text(json.dumps({"history": history, "best": best, "kill_check": verdict}, indent=1))
    print(f"done. best {best}", flush=True)


def adapter_args(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("args", {}) or {})


if __name__ == "__main__":
    main()
