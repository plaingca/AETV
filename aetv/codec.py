"""Load the default Flex-8k checkpoint and encode/decode video GOPs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from .config import AETV_MODES, AETVModeSpec
from .models import AETVAutoencoder

DEFAULT_CHECKPOINT = Path("models") / "v8-flex8k-ota-rxfix.pt"
MODE_DEFAULT_CHECKPOINTS = {
    "V8": Path("models") / "v8-hf3k-perceptual.pt",
}
DEFAULT_MODE = "V7"


def resolve_checkpoint(
    path: str | Path | None = None,
    mode: str | None = None,
) -> Path:
    """Return the checkpoint path, or raise with install instructions."""
    if path:
        candidates = [Path(path).expanduser()]
    else:
        candidates = []
        if configured := os.environ.get("AETV_CHECKPOINT"):
            candidates.append(Path(configured).expanduser())
        preferred = MODE_DEFAULT_CHECKPOINTS.get(mode or DEFAULT_MODE, DEFAULT_CHECKPOINT)
        candidates.extend(
            (
                preferred,
                Path(__file__).resolve().parent.parent / preferred,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"AETV checkpoint not found (searched: {searched}).\n"
        f"Install the default {mode or DEFAULT_MODE} weights, set AETV_CHECKPOINT, "
        "or pass --checkpoint."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AETVCodec:
    """Encoder/decoder pair for one AETV mode."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
        mode: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.checkpoint_path = resolve_checkpoint(checkpoint, mode=mode)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        args = payload.get("args", {}) or {}
        mode_name = mode or payload.get("mode") or args.get("mode") or DEFAULT_MODE
        if mode_name not in AETV_MODES:
            raise ValueError(f"unknown AETV mode {mode_name!r}")
        self.mode: AETVModeSpec = AETV_MODES[mode_name]
        self.step = payload.get("step")
        self.args = args
        self.model = AETVAutoencoder(
            mode=self.mode,
            width=int(args.get("model_width", 128)),
            latent_channels=int(args.get("latent_channels", 3)),
            compact=bool(args.get("compact", False)),
            causal=self.mode.causal,
        ).to(self.device)
        state = payload.get("model_state_dict") or payload.get("model")
        if state is None:
            raise KeyError(f"{self.checkpoint_path} has no model_state_dict")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def encode_gop(self, frames: np.ndarray) -> np.ndarray:
        """Encode (T, H, W, 3) uint8 or float frames to a GOP latent vector."""
        video = _to_nchw(frames, self.mode).to(self.device)
        # This is the hot live path; the station model is never mutated.
        # inference_mode also removes autograd's view/version bookkeeping.
        with torch.inference_mode():
            latents = self.model.encoder(video)
        return latents.squeeze(0).float().cpu().numpy()

    def decode_gop(
        self,
        latents: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode one GOP latent vector to (T, H, W, 3) uint8 frames."""
        z = torch.from_numpy(np.asarray(latents, dtype=np.float32))[None].to(self.device)
        if weights is None:
            w = torch.ones_like(z)
        else:
            w = torch.from_numpy(np.asarray(weights, dtype=np.float32))[None].to(self.device)
        with torch.inference_mode():
            recon = self.model.decoder(
                z,
                w,
                output_shape=(self.mode.gop_frames, self.mode.height, self.mode.width),
            )
        return _to_uint8(recon.squeeze(0))


def _to_nchw(frames: np.ndarray, mode: AETVModeSpec) -> torch.Tensor:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) frames, got {array.shape}")
    if array.shape[0] != mode.gop_frames:
        raise ValueError(f"expected {mode.gop_frames} frames, got {array.shape[0]}")
    if array.shape[1:3] != (mode.height, mode.width):
        raise ValueError(
            f"expected {mode.width}x{mode.height} frames, got {array.shape[2]}x{array.shape[1]}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
    return tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def _to_uint8(clip: torch.Tensor) -> np.ndarray:
    array = clip.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 3, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)
