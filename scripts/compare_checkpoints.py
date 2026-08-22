"""Paired comparison of AETV checkpoints over a fixed held-out clip set.

The training loop's own eval reports a mean over 5 clips, which was not able to
resolve the objective changes it was being used to judge: measured across seven
evals, the 6 dB LPIPS cell's eval-to-eval spread (0.016) was larger than any
effect the objective produced. Two things fix that here.

*Pairing.* Every checkpoint sees the same clips and, because
`simulate_transmission` seeds `fading` and `awgn`, the same channel
realizations. So the per-clip difference between two checkpoints is measured
against an identical input, and content variance -- which dominates the spread
across clips -- cancels instead of being averaged over. The reported uncertainty
is the standard error of that paired difference, not of the means.

*Sample count.* All 32 held-out clips rather than the first 5.

Reports each checkpoint against a reference, per channel cell, in PSNR and
LPIPS. A delta is only worth acting on when it exceeds a couple of its own
standard errors.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (  # noqa: E402
    AETV_MODES,
    AETVAutoencoder,
    compute_psnr,
    lpips_metric,
    simulate_transmission,
)

# (label, snr_db, fading_preset) -- None snr means the clean latent loopback.
CELLS: list[tuple[str, float | None, str | None]] = [
    ("clean", None, None),
    ("18 dB", 18.0, None),
    ("12 dB", 12.0, None),
    ("6 dB", 6.0, None),
    ("0 dB", 0.0, None),
    ("fade mpg", 12.0, "mpg"),
    ("fade mpp", 6.0, "mpp"),
]


def load_clips(cache_dir: Path, mode_spec, limit: int) -> list[torch.Tensor]:
    """Same preprocessing as the training script's eval loader, so the numbers
    here are comparable with the figures in the training logs."""
    clips: list[torch.Tensor] = []
    for f in sorted(cache_dir.glob("*.pt"))[:limit]:
        clip = torch.load(f).float()
        if clip.ndim != 4:
            continue
        if clip.shape[1] < mode_spec.gop_frames:
            repeats = math.ceil(mode_spec.gop_frames / clip.shape[1])
            clip = clip.repeat(1, repeats, 1, 1)[:, : mode_spec.gop_frames]
        else:
            clip = clip[:, : mode_spec.gop_frames]
        if clip.shape[-2:] != (mode_spec.height, mode_spec.width):
            h_orig, w_orig = clip.shape[-2:]
            scale = max(mode_spec.height / h_orig, mode_spec.width / w_orig)
            h_new, w_new = int(round(h_orig * scale)), int(round(w_orig * scale))
            scaled = F.interpolate(
                clip.unsqueeze(0),
                size=(mode_spec.gop_frames, h_new, w_new),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            top = (h_new - mode_spec.height) // 2
            left = (w_new - mode_spec.width) // 2
            clip = scaled[:, :, top : top + mode_spec.height, left : left + mode_spec.width]
        if clip.max() > 1.0:
            clip = clip / 255.0
        clips.append(clip.unsqueeze(0))
    return clips


def eval_checkpoint(path: Path, clips, mode_spec, args, device):
    """-> {cell: {'psnr': [per-clip], 'lpips': [per-clip]}}"""
    model = AETVAutoencoder(
        mode=mode_spec,
        width=args.model_width,
        latent_channels=args.latent_channels,
        causal=mode_spec.causal,
    ).to(device)
    model.load_pretrained_weights(str(path), device=device)
    model.eval()

    out = {label: {"psnr": [], "lpips": []} for label, _, _ in CELLS}
    shape = (mode_spec.gop_frames, mode_spec.height, mode_spec.width)

    with torch.no_grad():
        for clip in clips:
            video = clip.to(device)
            z = model.encoder(video)
            lat_np = z[0].cpu().numpy()

            for label, snr_db, preset in CELLS:
                if snr_db is None:
                    recon = model.decoder(z, torch.ones_like(z), shape)
                else:
                    lat, w, _ = simulate_transmission(
                        lat_np,
                        mode_name=mode_spec.name,
                        snr_db=snr_db,
                        fading_preset=preset,
                    )
                    recon = model.decoder(
                        torch.from_numpy(lat).unsqueeze(0).to(device),
                        torch.from_numpy(w).unsqueeze(0).to(device),
                        shape,
                    )
                out[label]["psnr"].append(compute_psnr(recon, video))
                out[label]["lpips"].append(lpips_metric(recon, video, device))

    del model
    torch.cuda.empty_cache()
    return out


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean and standard error of the per-clip difference a - b."""
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return (d[0] if d else 0.0), 0.0
    return st.mean(d), st.stdev(d) / math.sqrt(len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--reference", type=str, required=True, help="label to compare against")
    ap.add_argument("--mode", type=str, default="V7")
    ap.add_argument("--model-width", type=int, default=128)
    ap.add_argument("--latent-channels", type=int, default=3)
    ap.add_argument("--clips", type=int, default=32)
    ap.add_argument("--cache-dir", type=str, default="runs/openvid-cache-5fps-eval-v68")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_spec = AETV_MODES[args.mode]

    entries = []
    for item in args.checkpoints:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {item!r}")
        if not Path(path).exists():
            raise SystemExit(f"no such checkpoint: {path}")
        entries.append((label, Path(path)))
    if args.reference not in [lab for lab, _ in entries]:
        raise SystemExit(f"reference {args.reference!r} is not among the checkpoints")

    clips = load_clips(Path(args.cache_dir), mode_spec, args.clips)
    if len(clips) < 2:
        raise SystemExit(f"need at least 2 clips, found {len(clips)} in {args.cache_dir}")
    print(f"{len(clips)} held-out clips | {len(entries)} checkpoints | device {device}\n", flush=True)

    results = {}
    for label, path in entries:
        t0 = time.time()
        results[label] = eval_checkpoint(path, clips, mode_spec, args, device)
        print(f"  evaluated {label:<18} ({time.time() - t0:.0f}s)", flush=True)

    ref = results[args.reference]
    for metric, better in (("psnr", "higher"), ("lpips", "lower")):
        print(f"\n=== {metric.upper()} ({better} is better) ===")
        print(f"{'cell':>10} {'reference':>10} | " + " | ".join(f"{lab:>18}" for lab, _ in entries if lab != args.reference))
        for label, _, _ in CELLS:
            base = st.mean(ref[label][metric])
            cols = []
            for lab, _ in entries:
                if lab == args.reference:
                    continue
                d, se = paired(results[lab][label][metric], ref[label][metric])
                mark = "*" if se > 0 and abs(d) > 2 * se else " "
                cols.append(f"{d:>+9.4f}+-{se:<6.4f}{mark}")
            print(f"{label:>10} {base:>10.4f} | " + " | ".join(cols))

    print("\n* marks a paired delta larger than twice its own standard error.")
    print(f"reference = {args.reference}")


if __name__ == "__main__":
    main()
