"""Export and publish fixed-shape AETV inference graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
import torch

from .config import AETV_MODES
from .models import AETVAutoencoder


class _DecoderForExport(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, output_shape: tuple[int, int, int]):
        super().__init__()
        self.decoder = decoder
        self.output_shape = output_shape

    def forward(self, latents: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents, weights, self.output_shape)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    runtime_name: str | None = None,
) -> Path:
    """Capture encoder/decoder ONNX graphs from a native training checkpoint."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload.get("args", {}) or {}
    mode_name = payload.get("mode") or args.get("mode")
    if mode_name not in AETV_MODES:
        raise ValueError(f"{checkpoint} has unknown mode {mode_name!r}")
    mode = AETV_MODES[mode_name]
    model = AETVAutoencoder(
        mode=mode,
        width=int(args.get("model_width", 128)),
        latent_channels=int(args.get("latent_channels", 3)),
        compact=bool(args.get("compact", False)),
        causal=mode.causal,
    )
    state = payload.get("model_state_dict") or payload.get("model")
    if state is None:
        raise KeyError(f"{checkpoint} has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = runtime_name or checkpoint.stem
    encoder_path = output_dir / f"{stem}.encoder.onnx"
    decoder_path = output_dir / f"{stem}.decoder.onnx"
    metadata_path = output_dir / f"{stem}.runtime.json"
    frames = torch.zeros(1, 3, mode.gop_frames, mode.height, mode.width)
    latents = torch.zeros(1, mode.latents_per_gop)
    weights = torch.ones_like(latents)

    with torch.inference_mode():
        torch.onnx.export(
            model.encoder,
            (frames,),
            encoder_path,
            input_names=["frames"],
            output_names=["latents"],
            opset_version=18,
            dynamo=False,
        )
        torch.onnx.export(
            _DecoderForExport(model.decoder, (mode.gop_frames, mode.height, mode.width)),
            (latents, weights),
            decoder_path,
            input_names=["latents", "weights"],
            output_names=["frames"],
            opset_version=18,
            dynamo=False,
        )

    onnx.checker.check_model(str(encoder_path))
    onnx.checker.check_model(str(decoder_path))
    metadata_path.write_text(
        json.dumps(
            {
                "format": "aetv-onnx-v1",
                "mode": mode_name,
                "step": payload.get("step"),
                "source_checkpoint": checkpoint.name,
                "encoder": encoder_path.name,
                "decoder": decoder_path.name,
                "model_width": int(args.get("model_width", 128)),
                "latent_channels": int(args.get("latent_channels", 3)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    release_path = output_dir / f"{stem}.release.json"
    release_path.write_text(
        json.dumps(
            {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in (metadata_path, encoder_path, decoder_path)
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def publish_runtime_bundles(
    manifests: list[Path],
    *,
    repo_id: str,
    revision: str = "main",
    create_pr: bool = False,
) -> str:
    """Upload runtime bundles to Hugging Face in one atomic Hub commit."""
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError("install huggingface-hub to publish runtime models") from exc

    operations = []
    for manifest in manifests:
        manifest = Path(manifest)
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [
            manifest,
            manifest.with_name(metadata["encoder"]),
            manifest.with_name(metadata["decoder"]),
            manifest.with_name(manifest.name.replace(".runtime.json", ".release.json")),
        ]
        operations.extend(
            CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=str(path))
            for path in paths
        )
    result = HfApi().create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        operations=operations,
        commit_message="Publish AETV ONNX runtime models",
        create_pr=create_pr,
    )
    return result.pr_url or result.commit_url
