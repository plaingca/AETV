"""Load the default Flex-8k checkpoint and encode/decode video GOPs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import numpy as np

from .config import AETV_MODES, AETVModeSpec

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
RELEASE_RUNTIME_FILES = {
    "V7": {
        "v8-flex8k-ota-rxfix.runtime.json": {
            "bytes": 265,
            "sha256": "8bc9f5cb5a9330efe7cde6d191a763bfe44fcf9336ec7a7624cc14bffa4a3d26",
        },
        "v8-flex8k-ota-rxfix.encoder.onnx": {
            "bytes": 117083519,
            "sha256": "59acfa659284b84d5ea8ac9928aa8eccec87ae504f3b29b03aebe2ea5c3b0c9e",
        },
        "v8-flex8k-ota-rxfix.decoder.onnx": {
            "bytes": 99051480,
            "sha256": "a59affcc4c832bf6c11cfb3e91cdeab6ac802f3a76d4a531307b4f0ecd2b1997",
        },
    },
    "V8": {
        "v8-hf3k-face-gan.runtime.json": {
            "bytes": 256,
            "sha256": "02e0297d4102eb08e96daec2579f1f5fe2ba45631334f1b48659461420b10890",
        },
        "v8-hf3k-face-gan.encoder.onnx": {
            "bytes": 117083518,
            "sha256": "48659a6caf57cdca848a9b2a2bb475020e2d247bb59452a135f705dedb9d2362",
        },
        "v8-hf3k-face-gan.decoder.onnx": {
            "bytes": 98874127,
            "sha256": "34f881ba7d5095cc01991f70d51c4821f584cba0473ca77f9aa393a2f8ac9d1a",
        },
    },
}
# Immutable Hub commit containing both checksum-pinned runtime bundles.
HF_RUNTIME_REVISION = "a812f2c573fd37da0a4686a03a029c0fd39bb798"
_DOWNLOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReleaseModelStatus:
    """Result of checking one checksum-pinned release model on disk."""

    mode: str
    installed: bool
    path: Path | None = None
    backend: str = ""
    problem: str = ""


DownloadProgress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class RuntimeDevice:
    """Small backend-neutral replacement for ``torch.device`` in the GUI."""

    type: str
    label: str

    def __str__(self) -> str:
        return self.label


def model_cache_dir() -> Path:
    """Return the writable per-user checkpoint cache directory."""
    if configured := os.environ.get("AETV_MODEL_DIR"):
        return Path(configured).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "AETV" / "models"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "aetv" / "models"


def runtime_bundle_bytes(mode: str) -> int:
    """Return the total download size of a mode's ONNX runtime bundle."""
    return sum(int(item["bytes"]) for item in RELEASE_RUNTIME_FILES[mode].values())


def _model_roots() -> list[Path]:
    """Model search order shared by inventory and normal codec resolution."""
    candidates = [
        model_cache_dir(),
        Path("models"),
        Path(__file__).resolve().parent.parent / "models",
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        key = os.path.normcase(str(expanded.absolute()))
        if key not in seen:
            seen.add(key)
            roots.append(expanded)
    return roots


def _valid_release_runtime(mode: str) -> tuple[Path | None, bool]:
    """Return a verified manifest and whether any corrupt/incomplete files exist."""
    files = RELEASE_RUNTIME_FILES[mode]
    manifest_name = next(name for name in files if name.endswith(".runtime.json"))
    found_invalid = False
    for root in _model_roots():
        if not any((root / name).exists() for name in files):
            continue
        valid = True
        for filename, expected in files.items():
            path = root / filename
            if (
                not path.is_file()
                or path.stat().st_size != expected["bytes"]
                or sha256_file(path) != expected["sha256"]
            ):
                valid = False
                found_invalid = True
                break
        if valid:
            return (root / manifest_name).resolve(), found_invalid
    return None, found_invalid


def _valid_release_checkpoint(mode: str) -> tuple[Path | None, bool]:
    """Return a verified native checkpoint when this installation can use Torch."""
    if importlib.util.find_spec("torch") is None:
        return None, False
    release = RELEASE_CHECKPOINTS[mode]
    found_invalid = False
    for root in _model_roots():
        path = root / release["filename"]
        if not path.exists():
            continue
        if (
            path.is_file()
            and path.stat().st_size == release["bytes"]
            and sha256_file(path) == release["sha256"]
        ):
            return path.resolve(), found_invalid
        found_invalid = True
    return None, found_invalid


def inspect_release_model(mode: str) -> ReleaseModelStatus:
    """Check whether a mode has a complete, usable release model installed.

    ONNX is preferred because it is the backend shipped in portable builds.
    Source installations may also use a checksum-valid native checkpoint when
    PyTorch is available.
    """
    if mode not in RELEASE_RUNTIME_FILES or mode not in RELEASE_CHECKPOINTS:
        return ReleaseModelStatus(mode, False, problem="no release model is published")
    runtime, invalid_runtime = _valid_release_runtime(mode)
    if runtime is not None:
        return ReleaseModelStatus(mode, True, runtime, "ONNX Runtime")
    checkpoint, invalid_checkpoint = _valid_release_checkpoint(mode)
    if checkpoint is not None:
        return ReleaseModelStatus(mode, True, checkpoint, "PyTorch")
    problem = (
        "incomplete or failed checksum"
        if invalid_runtime or invalid_checkpoint
        else "not installed"
    )
    return ReleaseModelStatus(mode, False, problem=problem)


def inspect_release_models(
    modes: tuple[str, ...] | list[str],
) -> dict[str, ReleaseModelStatus]:
    """Inventory several release modes without making a network request."""
    return {mode: inspect_release_model(mode) for mode in modes}


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


def download_runtime_bundle(
    mode: str,
    *,
    destination: Path | None = None,
    progress: DownloadProgress | None = None,
) -> Path:
    """Download and verify one mode's ONNX manifest and graph pair."""
    try:
        files = RELEASE_RUNTIME_FILES[mode]
    except KeyError as exc:
        raise FileNotFoundError(f"no downloadable runtime model is published for {mode}") from exc
    target_dir = Path(destination or model_cache_dir()).expanduser()
    manifest_name = next(name for name in files if name.endswith(".runtime.json"))
    total_bytes = runtime_bundle_bytes(mode)

    with _DOWNLOAD_LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        completed_bytes = 0
        if progress is not None:
            progress(completed_bytes, total_bytes, "Preparing download")
        for filename, expected in files.items():
            target = target_dir / filename
            if (
                target.is_file()
                and target.stat().st_size == expected["bytes"]
                and sha256_file(target) == expected["sha256"]
            ):
                completed_bytes += int(expected["bytes"])
                if progress is not None:
                    progress(completed_bytes, total_bytes, f"Verified {filename}")
                continue
            temporary = target.with_name(
                f".{target.name}.{os.getpid()}.{threading.get_ident()}.download"
            )
            url = (
                f"https://huggingface.co/{HF_MODEL_REPO}/resolve/"
                f"{quote(HF_RUNTIME_REVISION, safe='')}/{quote(filename)}?download=true"
            )
            digest = hashlib.sha256()
            downloaded = 0
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "AETV/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
                    "wb"
                ) as output:
                    while chunk := response.read(1 << 20):
                        output.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress(
                                completed_bytes + downloaded,
                                total_bytes,
                                f"Downloading {filename}",
                            )
                if downloaded != expected["bytes"]:
                    raise RuntimeError(
                        f"downloaded {downloaded} bytes for {filename}; "
                        f"expected {expected['bytes']}"
                    )
                actual_sha = digest.hexdigest()
                if actual_sha != expected["sha256"]:
                    raise RuntimeError(
                        f"checksum mismatch for {filename}: {actual_sha}; "
                        f"expected {expected['sha256']}"
                    )
                os.replace(temporary, target)
                completed_bytes += downloaded
                if progress is not None:
                    progress(completed_bytes, total_bytes, f"Verified {filename}")
            except (OSError, urllib.error.URLError) as exc:
                raise RuntimeError(
                    f"could not download the {mode} runtime model from {HF_MODEL_REPO}: {exc}"
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)
        return (target_dir / manifest_name).resolve()


def resolve_checkpoint(
    path: str | Path | None = None,
    mode: str | None = None,
    *,
    allow_download: bool | None = None,
) -> Path:
    """Return the checkpoint path, or raise with install instructions."""
    requested_mode = mode or DEFAULT_MODE
    if path:
        candidates = [Path(path).expanduser()]
        may_download = False
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
        may_download = (
            not bool(os.environ.get("AETV_OFFLINE"))
            if allow_download is None
            else allow_download and not bool(os.environ.get("AETV_OFFLINE"))
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if may_download:
        return download_default_checkpoint(requested_mode)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"AETV checkpoint not found (searched: {searched}).\n"
        f"Install the default {requested_mode} weights, unset AETV_OFFLINE, "
        "set AETV_CHECKPOINT, or pass --checkpoint."
    )


def resolve_runtime_bundle(
    path: str | Path | None = None,
    mode: str | None = None,
    *,
    allow_download: bool = True,
) -> Path | None:
    """Find an exported ONNX runtime manifest without importing PyTorch."""
    requested_mode = mode or DEFAULT_MODE
    candidates: list[Path] = []
    if path:
        selected = Path(path).expanduser()
        if selected.name.endswith(".runtime.json"):
            candidates.append(selected)
        elif selected.suffix.lower() == ".onnx":
            base = selected.name.removesuffix(".encoder.onnx").removesuffix(".decoder.onnx")
            candidates.append(selected.with_name(f"{base}.runtime.json"))
        else:
            return None
    else:
        if configured := os.environ.get("AETV_RUNTIME_MODEL"):
            candidates.append(Path(configured).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if path:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"AETV ONNX runtime manifest not found (searched: {searched})")
    if requested_mode in RELEASE_RUNTIME_FILES:
        verified, _found_invalid = _valid_release_runtime(requested_mode)
        if verified is not None:
            return verified
    if allow_download and not os.environ.get("AETV_OFFLINE"):
        return download_runtime_bundle(requested_mode)
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AETVCodec:
    """Backend-neutral encoder/decoder pair for one AETV mode.

    Exported ONNX models are preferred for the operator GUI. Native ``.pt``
    checkpoints remain supported for training, evaluation, and model export.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | Any | None = None,
        mode: str | None = None,
        *,
        allow_download: bool = True,
    ):
        runtime_manifest = resolve_runtime_bundle(
            checkpoint, mode=mode, allow_download=allow_download
        )
        if runtime_manifest is not None:
            self._init_onnx(runtime_manifest, device=device, requested_mode=mode)
            return
        self._init_torch(
            checkpoint, device=device, mode=mode, allow_download=allow_download
        )

    def _init_onnx(
        self,
        manifest_path: Path,
        *,
        device: str | Any | None,
        requested_mode: str | None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for exported AETV models; install "
                "the 'gui' extra or onnxruntime"
            ) from exc
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "aetv-onnx-v1":
            raise ValueError(f"unsupported AETV runtime format in {manifest_path}")
        mode_name = metadata.get("mode")
        if mode_name not in AETV_MODES:
            raise ValueError(f"unknown AETV mode {mode_name!r}")
        if requested_mode and requested_mode != mode_name:
            raise ValueError(
                f"runtime model is for {mode_name}, not requested mode {requested_mode}"
            )
        encoder_path = manifest_path.with_name(metadata["encoder"])
        decoder_path = manifest_path.with_name(metadata["decoder"])
        for model_path in (encoder_path, decoder_path):
            if not model_path.is_file():
                raise FileNotFoundError(f"runtime model component not found: {model_path}")

        requested = str(device or "auto").lower()
        available = ort.get_available_providers()
        use_dml = requested not in {"cpu", "cpu:0"} and "DmlExecutionProvider" in available
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if use_dml
            else ["CPUExecutionProvider"]
        )
        options = ort.SessionOptions()
        configured_threads = os.environ.get("AETV_CPU_THREADS")
        if configured_threads and not use_dml:
            options.intra_op_num_threads = max(1, int(configured_threads))
        self._encoder_session = ort.InferenceSession(
            str(encoder_path), sess_options=options, providers=providers
        )
        self._decoder_session = ort.InferenceSession(
            str(decoder_path), sess_options=options, providers=providers
        )
        self.backend = "onnxruntime"
        self.backend_version = ort.__version__
        self.device = RuntimeDevice("dml" if use_dml else "cpu", "DirectML" if use_dml else "CPU")
        self.cpu_threads = options.intra_op_num_threads or None
        self.checkpoint_path = manifest_path
        self.mode = AETV_MODES[mode_name]
        self.step = metadata.get("step")
        self.args = metadata
        self.model = None

    def _init_torch(
        self,
        checkpoint: str | Path | None,
        *,
        device: str | Any | None,
        mode: str | None,
        allow_download: bool,
    ) -> None:
        try:
            import torch
            from .models import AETVAutoencoder
        except ImportError as exc:
            raise RuntimeError(
                "This is a native PyTorch checkpoint. Install AETV's 'train' "
                "extra, or select an exported .runtime.json model."
            ) from exc
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.backend = "torch"
        self.backend_version = torch.__version__
        self.cpu_threads: int | None = None
        if self.device.type == "cpu":
            configured_threads = os.environ.get("AETV_CPU_THREADS")
            logical_threads = (
                int(configured_threads) if configured_threads else (os.cpu_count() or 1)
            )
            if logical_threads > 0 and torch.get_num_threads() != logical_threads:
                torch.set_num_threads(logical_threads)
            self.cpu_threads = torch.get_num_threads()
        self.checkpoint_path = resolve_checkpoint(
            checkpoint, mode=mode, allow_download=allow_download
        )
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
        if self.backend == "onnxruntime":
            video = _to_nchw_numpy(frames, self.mode)
            return self._encoder_session.run(None, {"frames": video})[0][0]
        import torch

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
        if self.backend == "onnxruntime":
            z = np.asarray(latents, dtype=np.float32)[None]
            w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=np.float32)[None]
            recon = self._decoder_session.run(
                None, {"latents": np.ascontiguousarray(z), "weights": np.ascontiguousarray(w)}
            )[0][0]
            return _to_uint8_numpy(recon)
        import torch

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

    def synchronize(self) -> None:
        """Wait for asynchronous backend work before timing an operation."""
        if self.backend == "torch" and self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)


def _validate_frames(frames: np.ndarray, mode: AETVModeSpec) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) frames, got {array.shape}")
    if array.shape[0] != mode.gop_frames:
        raise ValueError(f"expected {mode.gop_frames} frames, got {array.shape[0]}")
    if array.shape[1:3] != (mode.height, mode.width):
        raise ValueError(
            f"expected {mode.width}x{mode.height} frames, got {array.shape[2]}x{array.shape[1]}"
        )
    return array


def _to_nchw_numpy(frames: np.ndarray, mode: AETVModeSpec) -> np.ndarray:
    array = _validate_frames(frames, mode)
    converted = np.ascontiguousarray(array, dtype=np.float32)
    if array.dtype == np.uint8:
        converted /= 255.0
    return np.ascontiguousarray(converted.transpose(3, 0, 1, 2)[None])


def _to_nchw(frames: np.ndarray, mode: AETVModeSpec):
    import torch

    array = _validate_frames(frames, mode)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
    return tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def _to_uint8(clip: Any) -> np.ndarray:
    array = clip.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 3, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def _to_uint8_numpy(clip: np.ndarray) -> np.ndarray:
    array = np.clip(clip, 0.0, 1.0).transpose(1, 2, 3, 0)
    return np.rint(array * 255.0).astype(np.uint8)
