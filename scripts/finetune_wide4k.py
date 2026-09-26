#!/usr/bin/env python3
"""Warm-start the V9 (4 kHz band M) codec from V8 and train it through the V9 modem.

Init: the V8 release checkpoint widened from 3 to 6 latent channels
(``aetv.models.widen_latent_channels``). Before training it decodes exactly as
V8 does from its first 2,808 values; the added channels fill the rest of the
4,800-value GOP and start as repeats of the V8 channels.

Each step, ``--real-rows`` rows go through the real V9 OFDM modem and
``emulate(..., "mpp12")`` with training-only fade seeds; the channel error is
passed straight through to the encoder. Any remaining rows use the
differentiable band-M waveform channel. The loss is the V8 channel-only
fine-tune loss, computed on the channel output only.

Data and selection never touch the 64 eval clips: training uses the training
pool after ``--select-clips``; checkpoints are chosen on the first
``--select-clips`` pool clips through V9 + ``mpp12`` (fade seed base 7300).
The kill check compares against V8 through its own modem on the same clips
and fade seeds.

    python scripts/finetune_wide4k.py --out runs/v9-wide4k --steps 6000
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
    region_gradient_loss,
    region_reconstruction_loss,
    sample_face_crops,
)
from aetv.models import LATENT_CHANNEL_KEYS, AETVAutoencoder, MultiLayerVGGPerceptualLoss, widen_latent_channels  # noqa: E402
from aetv.shared_eval import AutoencoderAdapter, clip_psnr, load_clips, modem_exchange, paired, summarize, training_pool  # noqa: E402
from train import (  # noqa: E402
    dwt3d_loss,
    spatial_gradient_loss,
    temporal_acceleration_loss,
    temporal_cosine_loss,
    temporal_delta_loss,
)

RELEASE_V8 = "models/v8-hf3k-face-gan.pt"
MODE = "V9"


def build_widened(init: str, channels: int, device) -> tuple[AETVAutoencoder, dict]:
    payload = torch.load(init, map_location="cpu", weights_only=False)
    args = dict(payload.get("args", {}) or {})
    spec = AETV_MODES[MODE]
    model = AETVAutoencoder(mode=spec, width=int(args.get("model_width", 128)), latent_channels=channels,
                            compact=bool(args.get("compact", False)), causal=spec.causal)
    state = payload["model_state_dict"]
    if state["encoder.encoder.net.12.conv.weight"].shape[0] != channels:
        state = widen_latent_channels(state, channels)
    model.load_state_dict(state, strict=True)
    args.update({"latent_channels": channels, "mode": MODE})
    return model.to(device), args


@torch.no_grad()
def mpp12_scores(encode, decode, mode_name: str, gop: int, clips: torch.Tensor, seed0: int, device, workers) -> list[float]:
    """Per-clip mpp12 PSNR through ``mode_name``'s modem, the shared-scorer fade seeds."""
    per_clip = []
    for index in range(clips.shape[0]):
        clip = clips[index].float().div(255).unsqueeze(0).to(device)
        gops = [clip[:, :, s : s + gop] for s in range(0, clip.shape[2] - gop + 1, gop)]
        wires = [encode(g) for g in gops]
        seeds = [seed0 + index * len(wires) + g for g in range(len(wires))]
        results = list(workers.map(
            lambda item: modem_exchange(item[0].squeeze(0).float().cpu().numpy(), mode_name, item[1], "mpp12"),
            zip(wires, seeds),
        ))
        rx = [torch.from_numpy(r).to(device).unsqueeze(0) for r, _, _ in results]
        cf = [torch.from_numpy(c).to(device).unsqueeze(0) for _, c, _ in results]
        per_clip.append(clip_psnr(gops, [decode(r, c).clamp(0, 1) for r, c in zip(rx, cf)]))
    return per_clip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/v9-wide4k")
    ap.add_argument("--init", default=RELEASE_V8)
    ap.add_argument("--latent-channels", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--kill-step", type=int, default=1000)
    ap.add_argument("--kill-margin", type=float, default=0.2, help="pool mpp12 gain over V8 required at --kill-step")
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--real-rows", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--new-lr-mult", type=float, default=5.0, help="lr multiplier for the widened latent-channel tensors")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--select-clips", type=int, default=24)
    ap.add_argument("--select-seed0", type=int, default=7300)
    ap.add_argument("--train-seed0", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--l1-weight", type=float, default=0.8)
    ap.add_argument("--dwt-weight", type=float, default=1.0)
    ap.add_argument("--grad-weight", type=float, default=0.5)
    ap.add_argument("--temporal-weight", type=float, default=1.0)
    ap.add_argument("--temporal-accel-weight", type=float, default=0.3)
    ap.add_argument("--temporal-cosine-weight", type=float, default=0.2)
    ap.add_argument("--lpips-weight", type=float, default=0.06)
    ap.add_argument("--region-weight", type=float, default=1.5)
    ap.add_argument("--region-boost", type=float, default=12.0)
    ap.add_argument("--face-perceptual-weight", type=float, default=0.1)
    ap.add_argument("--face-model", default="data/teachers/face_detection_yunet_2023mar.onnx")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")

    pool = training_pool()
    select_paths, train_paths = pool[: args.select_clips], pool[args.select_clips :]
    select_clips = load_clips(select_paths)
    train_clips = load_clips(train_paths)
    print(f"train clips {len(train_paths)}, selection clips {len(select_paths)} (pool only)", flush=True)

    model, model_args = build_widened(args.init, args.latent_channels, device)
    spec = AETV_MODES[MODE]
    gop, shape = spec.gop_frames, (spec.gop_frames, spec.height, spec.width)
    channel = AETVWaveformChannel(band=spec.band, cfg=AETVChannelConfig(
        snr_db_range=(0.0, 16.0), p_fading=0.5, p_measured_path=0.4, p_truncate=0.0, erasure_rate_max=0.0,
    )).to(device)
    vgg = MultiLayerVGGPerceptualLoss().to(device).eval()
    teacher = RegionAttentionTeacher(args.face_model, device, face_score_threshold=0.72)

    widened = {name for name in LATENT_CHANNEL_KEYS}
    new_params = [p for n, p in model.named_parameters() if n in widened]
    old_params = [p for n, p in model.named_parameters() if n not in widened]
    optimizer = torch.optim.AdamW([
        {"params": old_params, "lr_mult": 1.0},
        {"params": new_params, "lr_mult": args.new_lr_mult},
    ], lr=args.lr, weight_decay=1e-4)

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))

    workers = ThreadPoolExecutor(max_workers=args.workers)

    def select(step: int) -> list[float]:
        model.eval()
        scores = mpp12_scores(model.encoder, lambda r, c: model.decoder(r, c, output_shape=shape), MODE, gop,
                              select_clips, args.select_seed0, device, workers)
        model.train()
        return scores

    v8 = AutoencoderAdapter("v8", RELEASE_V8, "V8", device)
    v8_scores = mpp12_scores(v8.model.encoder, lambda r, c: v8.model.decoder(r, c, output_shape=shape), "V8", gop,
                             select_clips, args.select_seed0, device, workers)
    del v8
    torch.cuda.empty_cache()
    init_scores = select(0)
    v8_mean = summarize(v8_scores)["mean"]
    print(f"pool mpp12: V8 {v8_mean:.3f}, widened init on V9 {summarize(init_scores)['mean']:.3f}", flush=True)
    history = [{"step": 0, "mpp12": summarize(init_scores)["mean"], "vs_v8": paired(init_scores, v8_scores)}]
    (out / "select_step0.json").write_text(json.dumps({"v8": v8_scores, "init": init_scores}))
    best = {"step": 0, "mpp12": history[0]["mpp12"]}

    rng = random.Random(args.seed)
    real_seed = args.train_seed0
    started = time.time()
    verdict = None
    model.train()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step) * group["lr_mult"]
        idx = [rng.randrange(train_clips.shape[0]) for _ in range(args.batch)]
        starts = [rng.randrange(0, train_clips.shape[2] - gop + 1) for _ in idx]
        video = torch.stack([train_clips[i, :, s : s + gop] for i, s in zip(idx, starts)]).to(device).float().div(255)
        if rng.random() < 0.5:
            video = video.flip(-1)
        with torch.no_grad():
            attention_mask, face_clips = teacher(video)
        face_grid = face_target = None
        if bool(face_clips.any()):
            face_grid, face_indices = face_crop_grid(attention_mask, face_clips, crop_size=64)
            face_target = sample_face_crops(video, face_grid, face_indices)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            clean_z = model.encoder(video)
            if args.real_rows < args.batch:
                noisy_z, weights = channel(clean_z.float())
            else:
                noisy_z, weights = clean_z.float(), torch.ones_like(clean_z, dtype=torch.float32)
            if args.real_rows > 0:
                rows = clean_z[: args.real_rows].detach().float().cpu().numpy()
                seeds = list(range(real_seed, real_seed + args.real_rows))
                real_seed += args.real_rows
                results = list(workers.map(lambda item: modem_exchange(item[0], MODE, item[1], "mpp12"), zip(rows, seeds)))
                rx = torch.from_numpy(np.stack([r for r, _, _ in results])).to(device)
                cf = torch.from_numpy(np.stack([c for _, c, _ in results])).to(device)
                real = clean_z[: args.real_rows].float()
                noisy_z = torch.cat([real + (rx - real).detach(), noisy_z[args.real_rows :]], 0)
                weights = torch.cat([cf, weights[args.real_rows :]], 0)
            recon = model.decoder(noisy_z.to(clean_z.dtype), weights.to(clean_z.dtype), output_shape=shape)
            region = (region_reconstruction_loss(recon, video, attention_mask, args.region_boost)
                      + 0.5 * region_gradient_loss(recon, video, attention_mask, args.region_boost))
            loss_face = (vgg(sample_face_crops(recon, face_grid, face_indices), face_target)
                         if face_target is not None else torch.zeros((), device=device))
            loss = (
                args.mse_weight * F.mse_loss(recon, video)
                + args.l1_weight * (recon - video).abs().mean()
                + args.dwt_weight * dwt3d_loss(recon, video, levels=3)
                + args.grad_weight * spatial_gradient_loss(recon, video)
                + args.temporal_weight * temporal_delta_loss(recon, video)
                + args.temporal_accel_weight * temporal_acceleration_loss(recon, video)
                + args.temporal_cosine_weight * temporal_cosine_loss(recon, video)
                + args.region_weight * region
                + args.lpips_weight * vgg(recon, video)
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
            scores = select(step)
            vs_v8 = paired(scores, v8_scores)
            record = {"step": step, "mpp12": summarize(scores)["mean"], "vs_v8": vs_v8,
                      "vs_init": paired(scores, init_scores)}
            history.append(record)
            log.write(json.dumps({"select": record}) + "\n")
            log.flush()
            print(f"select step {step}: pool mpp12 {record['mpp12']:.3f} "
                  f"(vs V8 {vs_v8['mean']:+.3f} ± {vs_v8['se']:.3f})", flush=True)
            state = {"mode": MODE, "stage": 2, "step": step,
                     "args": {**model_args, "finetune": vars(args)},
                     "model_state_dict": model.state_dict(), "source_run": str(out),
                     "init_checkpoint": args.init, "select": record}
            if record["mpp12"] > best["mpp12"]:
                best = {"step": step, "mpp12": record["mpp12"]}
                torch.save(state, out / "best.pt")
                print(f"  new best at step {step}", flush=True)
            torch.save(state, out / "latest.pt")
            if step == args.kill_step:
                ok = vs_v8["mean"] >= args.kill_margin
                verdict = {"step": step, "passed": ok, "vs_v8": vs_v8}
                (out / "kill_check.json").write_text(json.dumps(verdict, indent=1))
                print(f"KILL CHECK {'PASSED' if ok else 'FAILED'} at step {step}", flush=True)
                if not ok:
                    break
    (out / "history.json").write_text(json.dumps({"history": history, "best": best, "kill_check": verdict,
                                                  "v8_pool_mpp12": v8_mean}, indent=1))
    print(f"done. best {best}", flush=True)


if __name__ == "__main__":
    main()
