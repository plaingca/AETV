#!/usr/bin/env python3
"""Train the receiver-side GOP-boundary refiner on cached V8 ``mpp12`` decodes.

The codec is frozen; the wire is unchanged. Inputs are frozen-decoder outputs
through the real V8 modem + ``mpp12`` (``scripts/cache_v8_draws.py``) on
training-pool clips. The loss is the mean per-GOP ``log10`` MSE (the shared
``mpp12`` statistic) plus ``--temporal-weight`` times the mean ``log10``
frame-difference MSE over all transitions.

Selection: best ``mpp12`` PSNR on the 24 pool selection clips (fade seed base
7300). Kill check at ``--kill-step``: selection ``mpp12`` must not fall and
the boundary frame-difference PSNR must rise by more than twice its paired
standard error.

    python scripts/train_gop_refiner.py --train runs/gop-boundary/train4.pt \
        --select runs/gop-boundary/select24.pt --out runs/gop-refiner
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetv.gop_boundary import BoundaryRefiner  # noqa: E402
from eval_gop_boundary import GOP, gop_confidence_frames, score_cache, summarise  # noqa: E402


def refiner_loss(recon: torch.Tensor, video: torch.Tensor, temporal_weight: float) -> torch.Tensor:
    b, c, t, h, w = video.shape
    err = (recon - video).square().reshape(b, c, t // GOP, GOP, h, w)
    gop_mse = err.mean(dim=(1, 3, 4, 5)).clamp_min(1e-8)
    loss = torch.log10(gop_mse).mean()
    if temporal_weight:
        dr = recon[:, :, 1:] - recon[:, :, :-1]
        dv = video[:, :, 1:] - video[:, :, :-1]
        t_mse = (dr - dv).square().mean(dim=(1, 3, 4)).clamp_min(1e-8)
        loss = loss + temporal_weight * torch.log10(t_mse).mean()
    return loss


def frames(video: torch.Tensor) -> torch.Tensor:
    """(B, 3, T, H, W) in [0, 1] -> (B * T, 3, H, W) in [-1, 1]."""
    b, c, t, h, w = video.shape
    return video.transpose(1, 2).reshape(b * t, c, h, w) * 2 - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--per-gop", action="store_true", help="ablation: refine each GOP alone, no cross-GOP view")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--temporal-weight", type=float, default=0.25)
    ap.add_argument("--lpips-weight", type=float, default=0.0,
                    help="perceptual term; when > 0, selection and the kill check also require no LPIPS rise")
    ap.add_argument("--lpips-net", default="vgg", help="training LPIPS network (scoring always uses alex)")
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--kill-step", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    train = torch.load(args.train, map_location="cpu", weights_only=False, mmap=True)
    cache = torch.load(args.select, map_location="cpu", weights_only=False)
    if set(train["names"]) & set(cache["names"]):
        raise SystemExit("selection clips overlap the training clips")
    n, draws = train["decoded"].shape[:2]
    print(f"train {n} clips x {draws} draws, selection {len(cache['names'])} clips", flush=True)

    config = {"gop": GOP, "width": args.width, "blocks": args.blocks, "causal": args.causal, "per_gop": args.per_gop}
    model = BoundaryRefiner(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))

    import lpips

    score_lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    train_lpips = None
    if args.lpips_weight:
        train_lpips = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
        for p in train_lpips.parameters():
            p.requires_grad_(False)
    base_records = score_cache(cache, {}, device, score_lpips)["current"]
    base = summarise({"current": base_records})["current"]
    print(f"step 0: select mpp12 {base['mpp12']['mean']:.3f} boundary {base['boundary_tpsnr']['mean']:.3f} "
          f"lpips {base['lpips_mpp12']['mean']:.4f}", flush=True)
    best = {"step": 0, "mpp12": base["mpp12"]["mean"]}
    history, verdict = [], None
    rng = random.Random(args.seed)
    started = time.time()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        picks = [(rng.randrange(n), rng.randrange(draws)) for _ in range(args.batch)]
        video = torch.stack([train["source"][i] for i, _ in picks]).to(device, non_blocking=True).float().div(255)
        decoded = torch.stack([train["decoded"][i, d] for i, d in picks]).to(device, non_blocking=True).float()
        conf = gop_confidence_frames(torch.stack([train["gop_confidence"][i, d] for i, d in picks]).float()).to(device)
        if rng.random() < 0.5:
            video, decoded = video.flip(-1), decoded.flip(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon = model(decoded, conf)
            perceptual = train_lpips(frames(recon), frames(video)).mean() if train_lpips is not None else 0.0
        loss = refiner_loss(recon.float(), video, args.temporal_weight) + args.lpips_weight * perceptual
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            rate = step / (time.time() - started)
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e} {rate:.2f} steps/s", flush=True)
            log.write(json.dumps({"step": step, "loss": loss.item(), "lr": lr_at(step)}) + "\n")
            log.flush()
        if step % args.eval_interval == 0 or step == args.kill_step:
            records = score_cache(cache, {"refined": model.eval()}, device, score_lpips)
            model.train()
            records["current"] = base_records
            s = summarise(records)["refined"]
            d_mpp, d_bnd = s["mpp12"]["paired_vs_current"], s["boundary_tpsnr"]["paired_vs_current"]
            d_lp = s["lpips_mpp12"]["paired_vs_current"]
            record = {"step": step, "mpp12": s["mpp12"]["mean"], "d_mpp12": d_mpp["mean"], "se_mpp12": d_mpp["se"],
                      "boundary_tpsnr": s["boundary_tpsnr"]["mean"], "d_boundary": d_bnd["mean"],
                      "se_boundary": d_bnd["se"], "boundary_jump": s["boundary_jump"]["mean"],
                      "lpips_mpp12": s["lpips_mpp12"]["mean"], "d_lpips": d_lp["mean"], "se_lpips": d_lp["se"]}
            history.append(record)
            log.write(json.dumps({"select": record}) + "\n")
            log.flush()
            print(f"select step {step}: mpp12 {record['mpp12']:.3f} (Δ {d_mpp['mean']:+.3f} ± {d_mpp['se']:.3f}) "
                  f"boundary tpsnr Δ {d_bnd['mean']:+.3f} ± {d_bnd['se']:.3f} jump {record['boundary_jump']:.2f} "
                  f"lpips Δ {d_lp['mean']:+.4f} ± {d_lp['se']:.4f}", flush=True)
            state = {"config": config, "state_dict": model.state_dict(), "step": step, "select": record,
                     "codec": train["model"], "args": vars(args)}
            lpips_ok = not args.lpips_weight or d_lp["mean"] <= 0
            if record["mpp12"] > best["mpp12"] and lpips_ok:
                best = {"step": step, "mpp12": record["mpp12"]}
                torch.save(state, out / "best.pt")
                print(f"  new best at step {step}", flush=True)
            torch.save(state, out / "latest.pt")
            if step == args.kill_step:
                ok = d_mpp["mean"] >= 0 and d_bnd["mean"] > 2 * d_bnd["se"] and lpips_ok
                verdict = {"step": step, "passed": ok, "mpp12": d_mpp, "boundary_tpsnr": d_bnd, "lpips_mpp12": d_lp}
                (out / "kill_check.json").write_text(json.dumps(verdict, indent=1))
                print(f"KILL CHECK {'PASSED' if ok else 'FAILED'} at step {step}", flush=True)
                if not ok:
                    break
    (out / "history.json").write_text(json.dumps({"history": history, "best": best, "kill_check": verdict}, indent=1))
    print(f"done. best {best}", flush=True)


if __name__ == "__main__":
    main()
