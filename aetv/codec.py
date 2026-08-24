"""Load the default Flex-8k checkpoint and encode/decode video GOPs."""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from .config import AETV_MODES, AETVModeSpec
from .models import AETVAutoencoder

DEFAULT_CHECKPOINT = Path("models") / "v8-hf3k-face-gan.pt"
MODE_DEFAULT_CHECKPOINTS = {
    "V7": Path("models") / "v8-flex8k-ota-rxfix.pt",
}
DEFAULT_MODE = "V8"
HF_MODEL_REPO = "AETV/AETV"
HF_MODEL_REVISION = "7ed90b4a902937248c4408d9e02c29b876b07a75"
RELEASE_CHECKPOINTS = {
    "V7": {
        "filename": "v8-flex8k-ota-rxfix.pt",
        "bytes": 215759999,
        "sha256": "294987591b8ece1cb6fd6ad10349a160192e4e6fefc26d47bbbefd9cce9a778f",
    },
    "V8": {
        "filename": "v8-hf3k-face-gan.pt",
        "bytes": 215759785,
        "sha256": "f218376af9f9916050c9e345353da0c0970c392f58755efaa81d01e7ded8fc40",
    },
}
_DOWNLOAD_LOCK = threading.Lock()


def model_cache_dir() -> Path:
    """Return the writable per-user checkpoint cache directory."""
    if configured := os.environ.get("AETV_MODEL_DIR"):
        return Path(configured).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "AETV" / "models"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "aetv" / "models"


def download_default_checkpoint(
    mode: str,
    *,
    destination: Path | None = None,
) -> Path:
    """Download and verify a mode's release checkpoint from Hugging Face Hub."""
    try:
        release = RELEASE_CHECKPOINTS[mode]
    except KeyError as exc:
        raise FileNotFoundError(f"no downloadable checkpoint is published for mode {mode}") from exc
    target = destination or (model_cache_dir() / release["filename"])
    target = Path(target).expanduser()

    with _DOWNLOAD_LOCK:
        if target.is_file() and target.stat().st_size == release["bytes"]:
            if sha256_file(target) == release["sha256"]:
                return target.resolve()

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{threading.get_ident()}.download"
        )
        url = (
            f"https://huggingface.co/{HF_MODEL_REPO}/resolve/"
            f"{quote(HF_MODEL_REVISION, safe='')}/{quote(release['filename'])}?download=true"
        )
        digest = hashlib.sha256()
        downloaded = 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AETV/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1 << 20):
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
            if downloaded != release["bytes"]:
                raise RuntimeError(
                    f"downloaded {downloaded} bytes for {target.name}; "
                    f"expected {release['bytes']}"
                )
            actual_sha = digest.hexdigest()
            if actual_sha != release["sha256"]:
                raise RuntimeError(
                    f"checksum mismatch for {target.name}: {actual_sha}; "
                    f"expected {release['sha256']}"
                )
            os.replace(temporary, target)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"could not download the {mode} checkpoint from {HF_MODEL_REPO}: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return target.resolve()


def resolve_checkpoint(
    path: str | Path | None = None,
    mode: str | None = None,
) -> Path:
    """Return the checkpoint path, or raise with install instructions."""
    requested_mode = mode or DEFAULT_MODE
    if path:
        candidates = [Path(path).expanduser()]
        allow_download = False
    else:
        candidates = []
        if configured := os.environ.get("AETV_CHECKPOINT"):
            candidates.append(Path(configured).expanduser())
        preferred = MODE_DEFAULT_CHECKPOINTS.get(requested_mode, DEFAULT_CHECKPOINT)
        candidates.extend(
            (
                preferred,
                Path(__file__).resolve().parent.parent / preferred,
                model_cache_dir() / preferred.name,
            )
        )
        allow_download = not bool(os.environ.get("AETV_OFFLINE"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if allow_download:
        return download_default_checkpoint(requested_mode)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"AETV checkpoint not found (searched: {searched}).\n"
        f"Install the default {requested_mode} weights, unset AETV_OFFLINE, "
        "set AETV_CHECKPOINT, or pass --checkpoint."
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
        self.cpu_threads: int | None = None
        if self.device.type == "cpu":
            configured_threads = os.environ.get("AETV_CPU_THREADS")
            logical_threads = (
                int(configured_threads) if configured_threads else (os.cpu_count() or 1)
            )
            if logical_threads > 0 and torch.get_num_threads() != logical_threads:
                torch.set_num_threads(logical_threads)
            self.cpu_threads = torch.get_num_threads()
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
