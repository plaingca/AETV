#!/usr/bin/env python3
"""Render complete held-out comparisons for the strongest boundary methods."""

from pathlib import Path
import hashlib, json
import torch

from aetv.config import AETV_MODES
from scripts.experiment_boundary_frame_predictor import BoundaryFramePredictor, apply as apply_predictor
from scripts.train_symmetric_boundary_corrector import SymmetricBoundaryCorrector, apply as apply_corrector
from scripts.train_whole_gop_scene_corrector import WholeGOPSceneCorrector
from scripts.experiment_gop_boundaries import (
    DEFAULT_CELLS, SequenceCache, decode_cached_sequence, load_model, write_labeled_grid_mp4,
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_predictor(path, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = BoundaryFramePredictor(**p["config"]).to(device)
    m.load_state_dict(p["state_dict"]); return m.eval()


def load_corrector(path, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = SymmetricBoundaryCorrector(**p["config"]).to(device)
    m.load_state_dict(p["state_dict"]); return m.eval(), p


def load_whole(path, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = WholeGOPSceneCorrector(**p["config"]).to(device)
    m.load_state_dict(p["state_dict"]); return m.eval()


def main():
    device = torch.device("cuda")
    mode = AETV_MODES["V8"]
    dataset = SequenceCache(Path("runs/gop-boundary-data/v8_192x108_3gop_eval"))
    rx = torch.load("runs/v8-two-gop-boundary-sweep-lr1e5/eval-runtime-rx.pt", map_location="cpu", weights_only=False)
    base = load_model(Path("models/v8-hf3k-face-gan.pt"), mode, device).eval()
    predictor = load_predictor(Path("runs/v8-boundary-frame-fullres/predictor.pt"), device)
    one_sided, one_meta = load_corrector(Path("runs/v8-one-sided-boundary-measured/predictor.pt") if False else Path("runs/v8-one-sided-boundary-measured/corrector.pt"), device)
    whole = load_whole(Path("runs/v8-whole-gop-scene-corrector/corrector.pt"), device)
    out = Path("runs/v8-boundary-technique-renders"); out.mkdir(parents=True, exist_ok=True)
    manifest = {"frames_per_render": len(dataset) * 12, "sequences": len(dataset), "gui_boundary_blending": False, "files": {}}
    with torch.inference_mode():
        for cell in DEFAULT_CELLS:
            sources, released, learned, symmetric, causal, whole_gop = [], [], [], [], [], []
            for i in range(len(dataset)):
                source = dataset[i].unsqueeze(0).to(device)
                decoded = decode_cached_sequence(base, rx, cell, i, mode, device)
                gops = decoded.reshape(1, 3, 2, 6, 108, 192).permute(0, 2, 1, 3, 4, 5)
                pred = apply_predictor(predictor, gops)
                pred_video = pred.permute(0, 2, 1, 3, 4, 5).reshape(1, 3, 12, 108, 192)
                smoothed = gops.clone()
                f5, f6 = smoothed[:, 0, :, 5].clone(), smoothed[:, 1, :, 0].clone()
                smoothed[:, 0, :, 5] = 0.5 * f5 + 0.5 * f6
                smoothed[:, 1, :, 0] = 0.5 * f6 + 0.5 * f5
                smooth_video = smoothed.permute(0, 2, 1, 3, 4, 5).reshape(1, 3, 12, 108, 192)
                one = apply_corrector(one_sided, gops, False)
                one_video = one.permute(0, 2, 1, 3, 4, 5).reshape(1, 3, 12, 108, 192)
                corrected = gops.clone(); corrected[:, 1] = (corrected[:, 1] + whole(gops[:, 0], gops[:, 1])).clamp(0, 1)
                whole_video = corrected.permute(0, 2, 1, 3, 4, 5).reshape(1, 3, 12, 108, 192)
                sources.append(source.cpu()); released.append(decoded.cpu()); learned.append(pred_video.cpu()); symmetric.append(smooth_video.cpu()); causal.append(one_video.cpu()); whole_gop.append(whole_video.cpu())
            panels = [("Source", torch.cat(sources, dim=2)), ("Released V8", torch.cat(released, dim=2)), ("Full-res predictor", torch.cat(learned, dim=2)), ("Whole-GOP scene", torch.cat(whole_gop, dim=2)), ("Symmetric alpha 0.5", torch.cat(symmetric, dim=2)), ("One-sided trained", torch.cat(causal, dim=2))]
            path = out / f"boundary-techniques-{cell.label}-full-32.mp4"
            write_labeled_grid_mp4(panels, path, fps=mode.fps, columns=2)
            manifest["files"][cell.label] = {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            print(f"Rendered {path}", flush=True)
    (out / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
