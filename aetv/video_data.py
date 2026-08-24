"""Streaming video clips for the experimental HF video codec.

The primary source is the Hugging Face Dataset Viewer API.  It returns
short-lived URLs for video cells, so a training run only downloads the clips
it actually consumes rather than cloning a multi-terabyte dataset.  FFmpeg is
used as the decoder because it handles the mix of codecs found in web video
datasets and is already a normal dependency of video workstations.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


VIEWER_BASE = "https://datasets-server.huggingface.co"


@dataclass(frozen=True)
class VideoClipSpec:
    frames: int = 9
    fps: float = 4.0
    height: int = 128
    width: int = 160

    @property
    def duration_s(self) -> float:
        return (self.frames - 1) / self.fps


def _viewer_json(endpoint: str, retries: int = 5, **params):
    url = f"{VIEWER_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "aetv-training"})
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            last_error = error
            if error.code != 429 and error.code < 500:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 16)
        except URLError as error:
            last_error = error
            delay = min(2**attempt, 16)
        if attempt + 1 == retries:
            raise last_error
        time.sleep(delay + random.random() * 0.25)


def dataset_row_count(dataset: str, config: str, split: str) -> int:
    payload = _viewer_json("size", dataset=dataset)
    for item in payload.get("size", {}).get("splits", []):
        if item["config"] == config and item["split"] == split:
            # Viewer conversions can be partial.  Only rows with generated
            # assets are immediately streamable through /rows.
            return int(item["num_rows"])
    raise ValueError(f"no Dataset Viewer split {dataset}/{config}/{split}")


def _decode_clip(source: str | bytes, spec: VideoClipSpec, start_s: float) -> torch.Tensor:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for streamed video training")
    if isinstance(source, bytes):
        # MP4 files commonly put their index (the moov atom) after the media.
        # FFmpeg cannot seek back to those samples when stdin is a pipe, so
        # otherwise-valid OpenVid rows fail decoding after being downloaded.
        # A temporary seekable file also lets -ss seek before decoding.
        with tempfile.TemporaryDirectory(prefix="aetv-video-") as directory:
            path = Path(directory) / "source.video"
            path.write_bytes(source)
            return _decode_clip(str(path), spec, start_s)
    vf = (
        f"fps={spec.fps},"
        f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=increase,"
        f"crop={spec.width}:{spec.height}"
    )
    input_args = ["-ss", f"{start_s:.3f}", "-i", source]
    cmd = [
        "ffmpeg", "-v", "error", *input_args, "-vf", vf,
        "-frames:v", str(spec.frames), "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=45,
    )
    expected = spec.frames * spec.height * spec.width * 3
    if proc.returncode or len(proc.stdout) != expected:
        detail = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(
            f"ffmpeg returned {proc.returncode}, {len(proc.stdout)}/{expected} bytes: {detail}"
        )
    array = np.frombuffer(proc.stdout, dtype=np.uint8).copy()
    array = array.reshape(spec.frames, spec.height, spec.width, 3)
    return torch.from_numpy(array).permute(3, 0, 1, 2).float().div_(255.0)


class HFDatasetsVideoDataset(IterableDataset):
    """Native Hugging Face `datasets` stream over the repository's Lance shards."""

    def __init__(
        self,
        dataset: str = "lance-format/Openvid-1M",
        split: str = "train",
        video_column: str = "video_blob",
        spec: VideoClipSpec | None = None,
        epoch_size: int = 1000,
        seed: int = 0,
        shuffle_buffer: int = 8,
        shard_index: int = 0,
        num_shards: int = 1,
    ):
        super().__init__()
        try:
            import lance  # noqa: F401
            import datasets  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "native HF video streaming requires datasets and pylance"
            ) from error
        self.dataset = dataset
        self.split = split
        self.video_column = video_column
        self.spec = spec or VideoClipSpec()
        self.epoch_size = epoch_size
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")
        self.shard_index = shard_index
        self.num_shards = num_shards

    def _load_stream(self):
        import lance
        from datasets import Video, load_dataset
        from huggingface_hub import get_token

        # Build the remote fragments inside each Windows worker. Lance
        # fragments reopen their dataset while being pickled, which otherwise
        # causes redundant unauthenticated metadata requests during spawn.
        original_dataset = lance.dataset

        def compatible_dataset(uri, *args, storage_options=None, **kwargs):
            if storage_options is not None:
                storage_options = {
                    key: str(value) for key, value in storage_options.items()
                    if value is not None
                }
            return original_dataset(
                uri, *args, storage_options=storage_options, **kwargs
            )

        lance.dataset = compatible_dataset
        try:
            stream = load_dataset(
                self.dataset, split=self.split, streaming=True,
                columns=[self.video_column, "seconds", "fps"],
                # The Lance builder defaults to batches of 256 rows.  Binary
                # video columns are materialized for the whole builder batch,
                # so that default can download hundreds of videos before the
                # iterable yields its first example.
                batch_size=1,
                token=get_token(),
            )
        finally:
            lance.dataset = original_dataset
        return stream.cast_column(self.video_column, Video(decode=False))

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        stream = self._load_stream()
        total_shards = self.num_shards * workers
        global_shard = self.shard_index * workers + worker_id
        if total_shards > 1:
            stream = stream.shard(num_shards=total_shards, index=global_shard)
        # Partition remote fragments before filling a per-worker shuffle
        # buffer, keeping startup I/O disjoint and bounded.
        stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        rng = random.Random(self.seed + 1009 * worker_id)
        target = (
            self.epoch_size + total_shards - 1 - global_shard
        ) // total_shards
        yielded = 0
        for row in stream:
            seconds = float(row.get("seconds") or 0.0)
            cell = row.get(self.video_column) or {}
            encoded = cell.get("bytes") if isinstance(cell, dict) else None
            if not encoded or seconds < self.spec.duration_s:
                continue
            max_start = max(0.0, seconds - self.spec.duration_s - 0.05)
            try:
                yield _decode_clip(encoded, self.spec, rng.random() * max_start)
            except (RuntimeError, subprocess.TimeoutExpired, OSError):
                continue
            yielded += 1
            if yielded >= target:
                return


class HFViewerVideoDataset(IterableDataset):
    """On-demand clips from a Dataset Viewer video column.

    The current OpenVid Lance conversion exposes an initial streamable Viewer
    window.  More rows become usable automatically as the conversion grows.
    Workers partition row indices, and failed/corrupt web videos are skipped.
    """

    def __init__(
        self,
        dataset: str = "lance-format/Openvid-1M",
        config: str = "default",
        split: str = "train",
        video_column: str = "video_blob",
        spec: VideoClipSpec | None = None,
        epoch_size: int = 1000,
        seed: int = 0,
        page_size: int = 8,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.split = split
        self.video_column = video_column
        self.spec = spec or VideoClipSpec()
        self.epoch_size = epoch_size
        self.seed = seed
        self.page_size = min(max(page_size, 1), 100)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_rows = dataset_row_count(dataset, config, split)

    def _rows(self, offset: int, length: int):
        return _viewer_json(
            "rows", dataset=self.dataset, config=self.config, split=self.split,
            offset=offset, length=length,
        ).get("rows", [])

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        page_rng = random.Random(self.seed)
        rng = random.Random(self.seed + 1009 * worker_id)

        # Shuffle pages, then rows within each page.  This preserves Viewer API
        # batching without forcing a sequential curriculum.
        pages = list(range(0, self.num_rows, self.page_size))
        page_rng.shuffle(pages)
        yielded = 0
        target = (self.epoch_size + workers - 1 - worker_id) // workers
        for page_number, offset in enumerate(pages):
            if page_number % workers != worker_id:
                continue
            try:
                rows = self._rows(offset, min(self.page_size, self.num_rows - offset))
            except (HTTPError, URLError, TimeoutError):
                # A Viewer rate-limit outage should reduce the current
                # iterable epoch, not tear down a long-running trainer.
                continue
            rng.shuffle(rows)
            for row_in_page, wrapped in enumerate(rows):
                row = wrapped.get("row", {})
                row_index = int(wrapped.get("row_idx", offset + row_in_page))
                cell = row.get(self.video_column)
                url = cell.get("src") if isinstance(cell, dict) else None
                seconds = float(row.get("seconds") or 0.0)
                if not url or seconds < self.spec.duration_s:
                    continue
                max_start = max(0.0, seconds - self.spec.duration_s - 0.05)
                clip_rng = random.Random(self.seed + 104729 * row_index)
                cache_path = None
                if self.cache_dir is not None:
                    fps_tag = str(self.spec.fps).replace(".", "p")
                    cache_path = self.cache_dir / (
                        f"row_{row_index:07d}_{self.spec.frames}f_{fps_tag}fps_"
                        f"{self.spec.height}x{self.spec.width}.pt"
                    )
                try:
                    if cache_path is not None and cache_path.exists():
                        clip = torch.load(cache_path, map_location="cpu").float().div_(255.0)
                    else:
                        clip = _decode_clip(url, self.spec, clip_rng.random() * max_start)
                        if cache_path is not None:
                            temporary = cache_path.with_suffix(f".{worker_id}.tmp")
                            torch.save(clip.mul(255).byte(), temporary)
                            temporary.replace(cache_path)
                    yield clip
                except (RuntimeError, subprocess.TimeoutExpired, OSError):
                    continue
                yielded += 1
                if yielded >= target:
                    return


def _hsv_color(rng: random.Random, flat: bool = True) -> tuple[int, int, int]:
    import colorsys

    hue = rng.random()
    saturation = rng.uniform(0.35, 1.0)
    value = rng.uniform(0.45, 1.0) if flat else rng.uniform(0.15, 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def _bounce(value: float, limit: float) -> float:
    """Reflect an unbounded coordinate into [0, limit] (triangle wave)."""
    if limit <= 0:
        return 0.0
    period = 2 * limit
    value = value % period
    return value if value <= limit else period - value


def _sharp_scene(rng: random.Random, width: int, height: int) -> dict:
    """One procedural cartoon-like scene: flat regions, hard edges, text."""
    scene: dict = {"pan": (0.0, 0.0)}
    background_kind = rng.random()
    if background_kind < 0.5:
        scene["background"] = [(0, 0, width, height, _hsv_color(rng))]
    elif background_kind < 0.85:
        split = int(height * rng.uniform(0.3, 0.7))
        scene["background"] = [
            (0, 0, width, split, _hsv_color(rng)),
            (0, split, width, height, _hsv_color(rng)),
        ]
    else:
        bands = rng.randint(2, 4)
        edges = sorted({0, width} | {int(width * rng.random()) for _ in range(bands - 1)})
        scene["background"] = [
            (edges[i], 0, edges[i + 1], height, _hsv_color(rng))
            for i in range(len(edges) - 1)
        ]
    if rng.random() < 0.3:
        scene["pan"] = (rng.uniform(-4.0, 4.0), rng.uniform(-2.0, 2.0))

    objects = []
    for _ in range(rng.randint(2, 6)):
        kind = rng.choice(("rect", "rect", "ellipse", "polygon", "line", "stripes"))
        size = rng.uniform(0.10, 0.45) * min(width, height)
        item = {
            "kind": kind,
            "size": size,
            "aspect": rng.uniform(0.5, 2.0),
            "x": rng.uniform(0, width),
            "y": rng.uniform(0, height),
            "vx": rng.uniform(-6.0, 6.0),
            "vy": rng.uniform(-4.0, 4.0),
            "color": _hsv_color(rng),
            "outline": (0, 0, 0) if rng.random() < 0.5 else None,
            "angle": rng.uniform(0, math.tau),
            "spin": rng.uniform(-0.25, 0.25) if rng.random() < 0.5 else 0.0,
            "sides": rng.randint(3, 6),
            "stripes": rng.randint(3, 7),
            "color2": _hsv_color(rng),
        }
        objects.append(item)
    scene["objects"] = objects

    texts = []
    for _ in range(rng.randint(0, 2)):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        words = " ".join(
            "".join(rng.choice(letters) for _ in range(rng.randint(2, 8)))
            for _ in range(rng.randint(1, 3))
        )
        mode = rng.choice(("static", "ticker", "credits"))
        bright = rng.random() < 0.6
        texts.append({
            "text": words,
            "size": rng.randint(14, 40),
            "x": rng.uniform(0, width * 0.7),
            "y": rng.uniform(0, height * 0.85),
            "vx": rng.uniform(4.0, 14.0) * rng.choice((-1, 1)) if mode == "ticker" else 0.0,
            "vy": rng.uniform(2.0, 6.0) * rng.choice((-1, 1)) if mode == "credits" else 0.0,
            "color": (255, 255, 255) if bright else (0, 0, 0),
            "stroke": ((0, 0, 0) if bright else (255, 255, 255)) if rng.random() < 0.7 else None,
        })
    scene["texts"] = texts
    return scene


def _render_sharp_frame(scene: dict, t: int, width: int, height: int):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), scene["background"][0][4])
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1, color in scene["background"]:
        draw.rectangle((x0, y0, x1, y1), fill=color)
    pan_x, pan_y = scene["pan"]
    for item in scene["objects"]:
        cx = _bounce(item["x"] + (item["vx"] + pan_x) * t, width)
        cy = _bounce(item["y"] + (item["vy"] + pan_y) * t, height)
        half_w = item["size"] * item["aspect"] / 2
        half_h = item["size"] / 2
        box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        kind = item["kind"]
        if kind == "rect":
            draw.rectangle(box, fill=item["color"], outline=item["outline"], width=2)
        elif kind == "ellipse":
            draw.ellipse(box, fill=item["color"], outline=item["outline"], width=2)
        elif kind == "polygon":
            angle = item["angle"] + item["spin"] * t
            radius = item["size"] / 2
            points = [
                (
                    cx + radius * math.cos(angle + k * math.tau / item["sides"]),
                    cy + radius * math.sin(angle + k * math.tau / item["sides"]),
                )
                for k in range(item["sides"])
            ]
            draw.polygon(points, fill=item["color"], outline=item["outline"])
        elif kind == "line":
            angle = item["angle"] + item["spin"] * t
            radius = item["size"] / 2
            draw.line(
                (
                    cx - radius * math.cos(angle), cy - radius * math.sin(angle),
                    cx + radius * math.cos(angle), cy + radius * math.sin(angle),
                ),
                fill=item["color"], width=2,
            )
        elif kind == "stripes":
            count = item["stripes"]
            stripe = max((box[2] - box[0]) / count, 2.0)
            for k in range(count):
                draw.rectangle(
                    (box[0] + k * stripe, box[1], box[0] + (k + 1) * stripe, box[3]),
                    fill=item["color"] if k % 2 == 0 else item["color2"],
                )
    for item in scene["texts"]:
        try:
            font = ImageFont.load_default(size=item["size"])
        except TypeError:
            font = ImageFont.load_default()
        x = (item["x"] + (item["vx"] + pan_x) * t) % (width * 1.4) - width * 0.2
        y = (item["y"] + (item["vy"] + pan_y) * t) % (height * 1.2) - height * 0.1
        stroke = item["stroke"]
        draw.text(
            (x, y), item["text"], fill=item["color"], font=font,
            stroke_width=2 if stroke else 0, stroke_fill=stroke,
        )
    return image


class SharpSyntheticVideoDataset(torch.utils.data.Dataset):
    """Procedural sharp-content clips: flat colors, hard edges, text, pans.

    OpenVid natural video leaves animation, subtitles, thin lines, and flat
    color regions out of distribution, which is exactly where the codec was
    measured softest (the 60-second animation OTA test).  These clips are
    rendered at 2x and box-downsampled so edges carry the same half-pixel
    antialiasing that real downscaled animation does.  Deterministic per
    (seed, index), so a fixed range doubles as a regression suite.
    """

    def __init__(self, n: int = 100_000, spec: VideoClipSpec | None = None, seed: int = 0):
        self.n = n
        self.spec = spec or VideoClipSpec()
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        from PIL import Image

        s = self.spec
        rng = random.Random(self.seed * 1_000_003 + index)
        scale = 2
        render_width, render_height = s.width * scale, s.height * scale
        scene = _sharp_scene(rng, render_width, render_height)
        cut_at = None
        second_scene = None
        if s.frames >= 7 and rng.random() < 0.25:
            cut_at = rng.randint(3, s.frames - 3)
            second_scene = _sharp_scene(rng, render_width, render_height)
        flash_frame = rng.randrange(s.frames) if rng.random() < 0.08 else None
        frames = []
        for t in range(s.frames):
            if cut_at is not None and t >= cut_at:
                image = _render_sharp_frame(second_scene, t - cut_at, render_width, render_height)
            else:
                image = _render_sharp_frame(scene, t, render_width, render_height)
            image = image.resize((s.width, s.height), Image.BOX)
            if t == flash_frame:
                image = Image.blend(image, Image.new("RGB", image.size, (255, 255, 255)), 0.6)
            frames.append(torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()))
        clip = torch.stack(frames, dim=0)  # (T, H, W, 3)
        return clip.permute(3, 0, 1, 2).float().div_(255.0)


class MixedStreamDataset(IterableDataset):
    """Insert clips from a map-style dataset into a streamed epoch.

    Synthetic clips are inserted *between* base clips rather than replacing
    them, so no downloaded base clip is wasted.  ``fraction`` is the share of
    yielded clips that come from ``extra``; the epoch grows accordingly.
    """

    def __init__(self, base: IterableDataset, extra, fraction: float, seed: int = 0):
        super().__init__()
        if not 0.0 < fraction < 0.9:
            raise ValueError("mixture fraction must be in (0, 0.9)")
        self.base = base
        self.extra = extra
        self.fraction = fraction
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + 7919 * worker_id)
        # Geometric insertion: continuing with probability f yields an
        # expected f/(1-f) extras per base clip, i.e. a fraction f of all
        # yielded clips, without discarding any downloaded base clip.
        for clip in self.base:
            while rng.random() < self.fraction:
                yield self.extra[rng.randrange(len(self.extra))]
            yield clip


class SyntheticVideoDataset(torch.utils.data.Dataset):
    """Small moving-pattern dataset for shape, memory and training smoke tests."""

    def __init__(self, n: int = 32, spec: VideoClipSpec | None = None):
        self.n = n
        self.spec = spec or VideoClipSpec()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        s = self.spec
        y = torch.linspace(0, 1, s.height)[None, :, None]
        x = torch.linspace(0, 1, s.width)[None, None, :]
        frames = []
        for t in range(s.frames):
            phase = (index * 0.037 + t / max(s.frames - 1, 1)) % 1.0
            r = (x + phase).remainder(1.0).expand(1, s.height, s.width)
            g = (y + phase * 0.5).remainder(1.0).expand(1, s.height, s.width)
            b = (((x + y) * 0.5 + phase).remainder(1.0))
            frames.append(torch.cat([r, g, b], dim=0))
        return torch.stack(frames, dim=1)


class CachedVideoDataset(torch.utils.data.Dataset):
    """Fixed uint8 clip cache used for repeatable learning-curve runs."""

    def __init__(self, root, spec: VideoClipSpec | None = None):
        from pathlib import Path

        root = Path(root)
        self.files = sorted(root.glob("clip_*.pt"))
        if not self.files:
            self.files = sorted(root.glob("row_*.pt"))
        self.spec = spec
        if not self.files:
            raise ValueError(f"no clip_*.pt files in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        clip = torch.load(self.files[index], map_location="cpu").float().div_(255.0)
        if self.spec is not None:
            if clip.shape[1] != self.spec.frames:
                if clip.shape[1] < self.spec.frames:
                    raise ValueError(
                        f"cached clip has {clip.shape[1]} frames, requested {self.spec.frames}"
                    )
                # Reuse a higher-rate cache for semantic low-frame-rate modes.
                # Nearest integer indices preserve real source frames instead
                # of manufacturing intermediate motion through interpolation.
                indices = torch.linspace(
                    0, clip.shape[1] - 1, self.spec.frames
                ).round().long()
                clip = clip.index_select(1, indices)
            if clip.shape[-2:] != (self.spec.height, self.spec.width):
                # Treat time as a batch so resizing cannot blend neighboring
                # frames or manufacture motion.
                frames = clip.permute(1, 0, 2, 3)
                frames = torch.nn.functional.interpolate(
                    frames, size=(self.spec.height, self.spec.width),
                    mode="bilinear", align_corners=False,
                )
                clip = frames.permute(1, 0, 2, 3).contiguous()
        return clip
