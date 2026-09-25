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


def region_terms(recon, video, mask, boost):
    return (
        region_reconstruction_loss(recon, video, mask, boost) + 0.5 * region_gradient_loss(recon, video, mask, boost),
        region_detail_loss(recon, video, mask, boost),
        region_contrast_loss(recon, video, mask, boost),
    )


def real_channel(wire: np.ndarray, seed: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    profile = rng.choice(REAL_PROFILES)
    received, confidence, _ = modem_exchange(wire, "V8", seed, profile)
    return received, confidence


@torch.no_grad()
def select_score(adapter, clips, names, faces, lpips_metric, device, seed0: int) -> dict:
    adapter.model.eval()
    records = []
    for index in range(clips.shape[0]):
        mask = faces.mask(names[index], clips[index])
        records.append(score_clip(adapter, clips[index], index, device, {"ft": 1.0}, seed0=seed0,
                                  face_mask=mask, lpips_metric=lpips_metric)["ft"])
    out = {"per_clip": {
        "mpp12": [r.rows["mpp12"] for r in records],
        "clean": [r.rows["clean"] for r in records],
        "modem_clean": [r.rows["modem_clean"] for r in records],
        "awgn6": [r.rows["awgn6"] for r in records],
        "lpips_clean": [r.lpips["clean"] for r in records],
        "lpips_mpp12": [r.lpips["mpp12"] for r in records],
        "face_mpp12": [r.face.get("mpp12") for r in records],
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
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")

    pool = training_pool()
    select_paths, train_paths = pool[: args.select_clips], pool[args.select_clips :]
    select_clips = load_clips(select_paths)
    select_names = [p.name for p in select_paths]
    print(f"train clips {len(train_paths)}, selection clips {len(select_paths)} (pool only)", flush=True)
    train_clips = load_clips(train_paths)

    adapter = AutoencoderAdapter("v8-ft", args.init, "V8", device)
    model = adapter.model
    spec = AETV_MODES["V8"]
    gop = spec.gop_frames
    channel = AETVWaveformChannel(band=spec.band, cfg=AETVChannelConfig(
        snr_db_range=(0.0, 16.0), p_fading=0.5, p_measured_path=0.4, p_truncate=0.0, erasure_rate_max=0.0,
    )).to(device)
    vgg = MultiLayerVGGPerceptualLoss().to(device).eval()
    teacher = RegionAttentionTeacher(args.face_model, device, face_score_threshold=0.72)
    faces = FaceMasks(args.face_model)
    import lpips

    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))

    baseline = select_score(adapter, select_clips, select_names, faces, lpips_metric, device, args.select_seed0)
    s = baseline["summary"]
    print(f"step 0 (release): pool mpp12 {s['mpp12']['mean']:.3f} clean {s['clean']['mean']:.3f} "
          f"lpips_mpp12 {s['lpips_mpp12']['mean']:.4f}", flush=True)
    history = [{"step": 0, **{k: v["mean"] for k, v in s.items()}}]
    (out / "select_step0.json").write_text(json.dumps(baseline))
    best = {"step": 0, "mpp12": s["mpp12"]["mean"]}

    rng = random.Random(args.seed)
    pool_exec = ThreadPoolExecutor(max_workers=max(1, args.real_rows))
    started = time.time()
    real_seed = args.train_seed0
    verdict = None
    for step in range(1, args.steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        idx = [rng.randrange(train_clips.shape[0]) for _ in range(args.batch)]
        starts = [rng.randrange(0, 12 - gop + 1) for _ in idx]
        video = torch.stack([train_clips[i, :, s0 : s0 + gop] for i, s0 in zip(idx, starts)]).to(device).float().div(255)
        if rng.random() < 0.5:
            video = video.flip(-1)
        with torch.no_grad():
            attention_mask, face_clips = teacher(video)
        face_grid, face_indices = (None, torch.empty(0, dtype=torch.long, device=device))
        face_target = None
        if bool(face_clips.any()):
            face_grid, face_indices = face_crop_grid(attention_mask, face_clips, crop_size=64)
            face_target = sample_face_crops(video, face_grid, face_indices)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            clean_z = model.encoder(video)
            noisy_z, weights = channel(clean_z.float())
            if args.real_rows > 0:
                wires = clean_z[: args.real_rows].detach().float().cpu().numpy()
                seeds = list(range(real_seed, real_seed + args.real_rows))
                real_seed += args.real_rows
                rngs = [random.Random(sd) for sd in seeds]
                results = list(pool_exec.map(real_channel, wires, seeds, rngs))
                rx = torch.from_numpy(np.stack([r for r, _ in results])).to(device)
                cf = torch.from_numpy(np.stack([c for _, c in results])).to(device)
                real_z = clean_z[: args.real_rows].float() + (rx - clean_z[: args.real_rows].float()).detach()
                noisy_z = torch.cat([real_z, noisy_z[args.real_rows :]], 0)
                weights = torch.cat([cf, weights[args.real_rows :]], 0)
            shape = (gop, spec.height, spec.width)
            recon = model.decoder(noisy_z.to(clean_z.dtype), weights.to(clean_z.dtype), output_shape=shape)
            recon_clean = model.decoder(clean_z, torch.ones_like(clean_z), output_shape=shape)
            with torch.no_grad():
                err = (noisy_z.float() - clean_z.float()).pow(2).mean().clamp_min(1e-8)
                snr = clean_z.float().pow(2).mean() / err
                conf = snr / (1 + snr)

            loss_region, loss_detail, loss_contrast = region_terms(recon, video, attention_mask, args.region_boost)
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
            if face_target is not None:
                loss_face = 0.5 * (
                    vgg(sample_face_crops(recon_clean, face_grid, face_indices), face_target)
                    + vgg(sample_face_crops(recon, face_grid, face_indices), face_target)
                )
            ramp = 0.5 + 0.5 * conf
            loss = (
                args.mse_weight * F.mse_loss(recon, video)
                + args.l1_weight * (recon - video).abs().mean()
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
                + args.consistency_weight * F.l1_loss(recon, recon_clean.detach())
                + args.clean_anchor_weight * anchor
                + args.face_perceptual_weight * loss_face
            )
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
            scored = select_score(adapter, select_clips, select_names, faces, lpips_metric, device, args.select_seed0)
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
                     "select": record}
            if s["mpp12"]["mean"] > best["mpp12"] and deltas["clean"]["mean"] >= args.kill_clean:
                best = {"step": step, "mpp12": s["mpp12"]["mean"]}
                torch.save(state, out / "best.pt")
                print(f"  new best at step {step}", flush=True)
            torch.save(state, out / "latest.pt")
            if step == args.kill_step:
                ok = (deltas["mpp12"]["mean"] >= args.kill_mpp12 and deltas["clean"]["mean"] >= args.kill_clean
                      and deltas["lpips_mpp12"]["mean"] <= 0.0)
                verdict = {"step": step, "passed": ok, "mpp12": deltas["mpp12"], "clean": deltas["clean"],
                           "lpips_mpp12": deltas["lpips_mpp12"], "face_mpp12": deltas["face_mpp12"]}
                (out / "kill_check.json").write_text(json.dumps(verdict, indent=1))
                print(f"KILL CHECK {'PASSED' if ok else 'FAILED'} at step {step}", flush=True)
                if not ok:
                    break
    (out / "history.json").write_text(json.dumps({"history": history, "best": best, "kill_check": verdict}, indent=1))
    print(f"done. best {best}", flush=True)


def adapter_args(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("args", {}) or {})


if __name__ == "__main__":
    main()
