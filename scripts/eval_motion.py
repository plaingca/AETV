"""Measure temporal fidelity of AETV checkpoints on a paired clip set."""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_checkpoints import load_clips  # noqa: E402
from train import AETV_MODES, AETVAutoencoder, compute_psnr, simulate_transmission  # noqa: E402


def load_model(path: Path, mode, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("args", {})
    model = AETVAutoencoder(
        mode=mode,
        width=int(saved.get("model_width", 128)),
        latent_channels=int(saved.get("latent_channels", 3)),
        compact=bool(saved.get("compact", False)),
        causal=mode.causal,
    ).to(device)
    model.load_pretrained_weights(str(path), device=device)
    model.eval()
    return model


def temporal_stats(reference: torch.Tensor, reconstruction: torch.Tensor):
    target = (reference[:, :, 1:] - reference[:, :, :-1]).float().flatten()
    output = (reconstruction[:, :, 1:] - reconstruction[:, :, :-1]).float().flatten()
    target_energy = target.abs().mean().item()
    output_energy = output.abs().mean().item()
    target_centered = target - target.mean()
    output_centered = output - output.mean()
    corr = (
        (target_centered * output_centered).mean()
        / (target_centered.square().mean().sqrt() * output_centered.square().mean().sqrt()).clamp_min(1e-8)
    ).item()
    return target_energy, output_energy, corr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--mode", default="V8")
    ap.add_argument("--clips", type=int, default=32)
    ap.add_argument("--cache-dir", default="runs/openvid-cache-5fps-eval-v68")
    ap.add_argument("--snr", type=float, default=None, help="Evaluate through this AWGN channel SNR")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = AETV_MODES[args.mode]
    clips = load_clips(Path(args.cache_dir), mode, args.clips)
    if len(clips) < 2:
        raise SystemExit("at least two clips are required")

    for item in args.checkpoints:
        label, sep, raw_path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected label=path, got {item!r}")
        model = load_model(Path(raw_path), mode, device)
        rows = []
        with torch.no_grad():
            for clip in clips:
                video = clip.to(device)
                z = model.encoder(video)
                if args.snr is None:
                    received, weights = z, torch.ones_like(z)
                else:
                    lat, w, _ = simulate_transmission(
                        z[0].cpu().numpy(), mode_name=mode.name, snr_db=args.snr
                    )
                    received = torch.from_numpy(lat).unsqueeze(0).to(device)
                    weights = torch.from_numpy(w).unsqueeze(0).to(device)
                recon = model.decoder(received, weights)
                target_e, output_e, corr = temporal_stats(video, recon)
                rows.append((target_e, output_e, corr, compute_psnr(recon, video)))

        threshold = st.median(row[0] for row in rows)
        active = [row for row in rows if row[0] >= threshold]
        ratio = sum(row[1] for row in rows) / sum(row[0] for row in rows)
        active_ratio = sum(row[1] for row in active) / sum(row[0] for row in active)
        print(
            f"{label}: motion={100 * ratio:.2f}% corr={st.mean(r[2] for r in rows):.3f} "
            f"active_motion={100 * active_ratio:.2f}% active_corr={st.mean(r[2] for r in active):.3f} "
            f"psnr={st.mean(r[3] for r in rows):.2f}dB"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
