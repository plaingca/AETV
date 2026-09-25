#!/usr/bin/env python3
"""Train the carrier-fill receiver on real V8 + Watterson pairs.

The headline channel is ``mpp12`` (two equal-power Rayleigh paths, 1 Hz
Doppler, 2 ms delay, then AWGN at 12 dB in 2500 Hz, referenced to the
transmitted waveform). Training clips are the seed-2026 shuffle with the
64-clip eval prefix held out. The checkpoint is the fill that raises
reconstructed PSNR on a further held-out slice of that training pool.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetv import AETV_MODES, demodulate_gop_stream, modulate_gop_stream
from aetv.hfchannel import emulate
from aetv.multipath_common import MultipathFill
from aetv.narrowband_jscc import impair_wire, psnr
from scripts.finetune_ac2k_psnr import load_clips


def _modem(wire: torch.Tensor, seed: int, profile: str) -> tuple[torch.Tensor, torch.Tensor]:
    mode = AETV_MODES["V8"]
    audio = modulate_gop_stream(
        [wire.squeeze(0).detach().float().cpu().numpy()], mode_name="V8", callsign="EVAL"
    )
    impaired = audio if profile == "clean" else emulate(audio, profile, seed=seed, fs=mode.geometry.fs)
    demod = demodulate_gop_stream(impaired, band=mode.band, drift_track="off")
    if not demod.gops_latents:
        return torch.zeros_like(wire), torch.zeros_like(wire)
    received = torch.as_tensor(demod.gops_latents[0], dtype=wire.dtype, device=wire.device).unsqueeze(0)
    confidence = (
        torch.as_tensor(demod.gops_weights[0], dtype=wire.dtype, device=wire.device).unsqueeze(0).clamp(0, 1)
    )
    return received, confidence


@torch.no_grad()
def collect_pairs(
    model: MultipathFill,
    clips: torch.Tensor,
    count: int,
    seed0: int,
    profiles: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Encode ``count`` clips and pass every GOP through each named profile.

    Wires stay on CPU. ``clip_id`` groups the three GOPs of one clip so the
    decoder state can be replayed in order.
    """
    device = next(model.parameters()).device
    sent, received, confidence, video, stream_id = [], [], [], [], []
    started = time.time()
    for index in range(count):
        clip = clips[index].float().div(255.0).unsqueeze(0).to(device)
        model.reset()
        wires, gops = [], []
        for start in (0, 4, 8):
            gop = clip[:, :, start : start + 4]
            wires.append(model.encode_gop(gop, retain_state=True))
            gops.append(gop.squeeze(0).cpu())
        for profile_index, profile in enumerate(profiles):
            # One modem draw is one stream: GOP state has to advance in order,
            # and a second fade of the same clip is a different stream.
            stream = index * len(profiles) + profile_index
            for gop_index, (wire, gop) in enumerate(zip(wires, gops)):
                seed = seed0 + index * 30 + gop_index * 4 + profile_index
                rx, weight = _modem(wire, seed, profile)
                sent.append(wire.squeeze(0).cpu())
                received.append(rx.squeeze(0).cpu())
                confidence.append(weight.squeeze(0).cpu())
                video.append(gop)
                stream_id.append(stream)
        if index == 0 or (index + 1) % 16 == 0:
            rate = (index + 1) / max(time.time() - started, 1e-3)
            print(f"  collected {index + 1}/{count} clips ({rate:.2f} clips/s)", flush=True)
    return {
        "sent": torch.stack(sent),
        "received": torch.stack(received),
        "confidence": torch.stack(confidence),
        "video": torch.stack(video),
        "stream_id": torch.tensor(stream_id, dtype=torch.long),
    }


def _groups(stream_id: torch.Tensor) -> list[torch.Tensor]:
    seen = {}
    for position, value in enumerate(stream_id.tolist()):
        seen.setdefault(value, []).append(position)
    return [torch.tensor(seen[value], dtype=torch.long) for value in sorted(seen)]


@torch.no_grad()
def score_collected(model: MultipathFill, pairs: dict[str, torch.Tensor], use_fill: bool) -> float:
    """Stateful PSNR. ``use_fill=False`` decodes the raw modem output with AC2K."""
    device = next(model.parameters()).device
    was = model.training
    model.eval()
    scores = []
    for group in _groups(pairs["stream_id"]):
        model.reset()
        for position in group.tolist():
            received = pairs["received"][position].unsqueeze(0).to(device)
            confidence = pairs["confidence"][position].unsqueeze(0).to(device)
            if use_fill:
                recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
            else:
                recon, _ = model.base.decode_gop(received, confidence=confidence, retain_state=True)
            video = pairs["video"][position].unsqueeze(0).to(device)
            scores.append(psnr(video, recon))
    if was:
        model.train()
    return float(np.mean(scores))


def _batch_indices(count: int, batch: int, rng: random.Random) -> list[int]:
    return [rng.randrange(count) for _ in range(batch)]


def save_checkpoint(model: MultipathFill, path: Path, metrics: dict, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "ac2k_checkpoint": model.ac2k_path,
            "hidden": model.hidden,
            "layers": model.layers,
            "fill": {key: value.detach().cpu() for key, value in model.fill.state_dict().items()},
            "pixels": {key: value.detach().cpu() for key, value in model.pixels.state_dict().items()},
            "confidence_gain": float(model.confidence_gain),
            "metrics": metrics,
            "step": step,
            "budget": 2816,
            "bandwidth_khz": 2.2,
            "modem_mode": "V8",
            "channel": "mpp12",
        },
        path,
    )


def _stream_loss(model: MultipathFill, pairs: dict[str, torch.Tensor], group: torch.Tensor) -> torch.Tensor:
    """Video MSE through frozen AC2K. Latent MSE is the wrong target: a wire can
    move closer to the transmitted vector and still decode worse, because the
    codec trusts a coordinate only in proportion to its modem confidence.
    """
    device = next(model.parameters()).device
    model.reset()
    loss = None
    for position in group.tolist():
        received = pairs["received"][position].unsqueeze(0).to(device)
        confidence = pairs["confidence"][position].unsqueeze(0).to(device)
        video = pairs["video"][position].unsqueeze(0).to(device)
        restored = model.fill(received, confidence)
        recon, _outage = model.base.decode_gop(restored, confidence=confidence, retain_state=True)
        term = F.mse_loss(recon, video)
        loss = term if loss is None else loss + term
    return loss / len(group)


def train_fill(
    model: MultipathFill,
    train: dict[str, torch.Tensor],
    holdout: dict[str, torch.Tensor],
    steps: int,
    batch: int,
    lr: float,
    baseline: float,
    checkpoint: Path,
    log_path: Path,
) -> float:
    device = next(model.parameters()).device
    del device
    optimizer = torch.optim.AdamW(model.fill.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(2026)
    best = baseline
    groups = _groups(train["stream_id"])
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(batch):
            loss = _stream_loss(model, train, groups[rng.randrange(len(groups))])
            (loss / batch).backward()
            loss_value += float(loss.item()) / batch
        grad_norm = torch.nn.utils.clip_grad_norm_(model.fill.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            hold = score_collected(model, holdout, use_fill=True)
            record = {
                "step": step,
                "loss": loss_value,
                "grad_norm": float(grad_norm),
                "holdout_mpp_psnr": hold,
                "baseline": baseline,
            }
            print(
                f"step {step} video mse {loss_value:.5f} grad {float(grad_norm):.3f} "
                f"holdout {hold:.3f} dB (AC2K {baseline:.3f})",
                flush=True,
            )
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if hold > best + 0.02:
                best = hold
                save_checkpoint(model, checkpoint, record, step)
                print(f"  saved {checkpoint}", flush=True)
    return best


def train_pixels(
    model: MultipathFill,
    train: dict[str, torch.Tensor],
    holdout: dict[str, torch.Tensor],
    steps: int,
    lr: float,
    filled_baseline: float,
) -> float:
    """Fit the shared residual plane. Returns holdout PSNR with the head on."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.pixels.parameters(), lr=lr, weight_decay=1e-4)
    groups = _groups(train["stream_id"])
    rng = random.Random(7)
    model.train()
    # One stateful pass caches decoder inputs. The head never sends gradients
    # into AC2K or the fill.
    cache_recon = []
    cache_video = []
    with torch.no_grad():
        for group in groups:
            model.reset()
        for position in group.tolist():
            received = train["received"][position].unsqueeze(0).to(device)
            confidence = train["confidence"][position].unsqueeze(0).to(device)
            # Identity head at this point, so this is the confidence-scaled AC2K decode.
            recon, _ = model.decode_gop(received, confidence=confidence, retain_state=True)
            cache_recon.append(recon.squeeze(0).cpu())
            cache_video.append(train["video"][position])
    # One clean pass per clip (even stream ids are the first fade of that clip).
    for group in groups:
        if int(train["stream_id"][int(group[0])]) % 2 != 0:
            continue
        model.reset()
        for position in group.tolist():
            sent = train["sent"][position].unsqueeze(0).to(device)
            clean_recon, _ = model.base.decode_gop(sent, retain_state=True)
            cache_recon.append(clean_recon.squeeze(0).cpu())
            cache_video.append(train["video"][position])
    recons = torch.stack(cache_recon)
    videos = torch.stack(cache_video)
    for step in range(1, steps + 1):
        chosen = [rng.randrange(recons.shape[0]) for _ in range(8)]
        index = torch.tensor(chosen, dtype=torch.long)
        recon = recons[index].to(device)
        target = videos[index].to(device)
        prediction = model.pixels(recon)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == steps or step % 100 == 0:
            print(f"  pixel step {step} mse {loss.item():.5f}", flush=True)
    return score_collected(model, holdout, use_fill=True)


def choose_gain(model: MultipathFill, holdout: dict[str, torch.Tensor], baseline: float) -> float:
    """Pick a modem-confidence scale on held-out multipath clips."""
    best_gain, best_score = 1.0, baseline
    for gain in (1.0, 1.25, 1.5, 1.75):
        model.confidence_gain = gain
        score = score_collected(model, holdout, use_fill=True)
        print(f"  confidence gain {gain:.2f} -> {score:.3f} dB", flush=True)
        if score > best_score:
            best_gain, best_score = gain, score
    model.confidence_gain = best_gain
    print(f"selected confidence gain {best_gain:.2f} ({best_score:.3f} dB)", flush=True)
    return best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--pixel-steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--train-clips", type=int, default=96)
    parser.add_argument("--holdout-clips", type=int, default=16)
    parser.add_argument("--cache", default="data/openvid_aetv_cache/mode_ac6_192x108_12f")
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ac2k", default="models/ac2k-psnr-2.2khz-best.pt")
    parser.add_argument("--out", default="runs/multipath-fill")
    parser.add_argument("--checkpoint", default="models/multipath-fill-2.2khz-best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_clips, _val, _names = load_clips(Path(args.cache), args.val_clips, args.seed)
    need = args.train_clips + args.holdout_clips
    if train_clips.shape[0] < need:
        raise SystemExit(f"training pool has {train_clips.shape[0]} clips, need {need}")
    print(f"loading AC2K and a {args.layers}-layer carrier fill", flush=True)
    model = MultipathFill(args.ac2k, hidden=args.hidden, layers=args.layers).to(device)
    trainable = sum(parameter.numel() for parameter in model.fill.parameters())
    print(f"fill parameters {trainable / 1e6:.2f}M | train clips {args.train_clips}", flush=True)

    # Real modem pairs. Seeds start at 9000 so they do not collide with the
    # headline seeds 2026 + clip*3 + gop, and the clips themselves are outside
    # the 64-clip eval prefix.
    print("collecting train pairs", flush=True)
    train_pairs = collect_pairs(
        model, train_clips, args.train_clips, seed0=9000, profiles=("mpp12", "mpp12")
    )
    print("collecting holdout pairs", flush=True)
    holdout = collect_pairs(
        model,
        train_clips[args.train_clips :],
        args.holdout_clips,
        seed0=7000,
        profiles=("mpp12",),
    )
    baseline = score_collected(model, holdout, use_fill=False)
    print(f"holdout AC2K mpp12 {baseline:.3f} dB on {args.holdout_clips} clips", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scaled = choose_gain(model, holdout, baseline)
    print("fitting the common residual head", flush=True)
    with_head = train_pixels(
        model, train_pairs, holdout, args.pixel_steps, lr=args.lr, filled_baseline=scaled
    )
    print(f"holdout with common head {with_head:.3f} dB (scaled AC2K {scaled:.3f})", flush=True)
    # Keep the head only when held-out multipath PSNR actually moves.
    if with_head + 1e-6 < scaled:
        print("common head lowered holdout; reverting it", flush=True)
        model.pixels = type(model.pixels)().to(device)
        with_head = scaled
    save_checkpoint(
        model,
        Path(args.checkpoint),
        {
            "holdout_mpp_psnr": with_head,
            "scaled_ac2k": scaled,
            "baseline": baseline,
            "confidence_gain": model.confidence_gain,
        },
        args.pixel_steps,
    )
    print(f"saved {args.checkpoint}", flush=True)

    awgn_scores, raw_scores = [], []
    probe = train_pairs["sent"][:24].to(device)
    for row in range(probe.shape[0]):
        wire = probe[row : row + 1]
        noisy, confidence = impair_wire(wire, 15.0)
        model.reset()
        raw, _ = model.base.decode_gop(noisy, confidence=confidence, retain_state=False)
        model.reset()
        filled_recon, _ = model.decode_gop(noisy, confidence=confidence, retain_state=False)
        video = train_pairs["video"][row].unsqueeze(0).to(device)
        raw_scores.append(psnr(video, raw))
        awgn_scores.append(psnr(video, filled_recon))
    print(
        f"train-slice AWGN 15 dB  AC2K {float(np.mean(raw_scores)):.2f}  model {float(np.mean(awgn_scores)):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
