"""Ham-station TX/RX engines: encode, key, play, unkey; capture, demodulate, decode.

The one transmit rule: PTT always comes back down. A cancelled send, an
exception, or a wedged audio callback must unkey. The keyed region is
wrapped in try/finally, and a watchdog thread drops PTT if the
transmission overruns its known duration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .audio_io import open_input_stream, play_cancellable
from .cat import CatConfig, NullPtt, open_ptt
from .codec import AETVCodec, resolve_checkpoint
from .config import AETV_MODES
from .kiwi import KiwiCapture
from .modem import demodulate_gop_stream, modulate_gop_stream
from .ringbuffer import RingBuffer
from .settings import StationSettings
from .source import collect_gops, iter_video_file, iter_webcam, write_mp4
from .sync import SyncError

WATCHDOG_MARGIN_S = 15.0


class TxPhase(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    ENCODING = "encoding"
    MODULATING = "modulating"
    KEYING = "keying"
    SENDING = "sending"
    UNKEYING = "unkeying"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TxState:
    phase: TxPhase = TxPhase.IDLE
    progress: float = 0.0
    message: str = ""

    @property
    def active(self) -> bool:
        return self.phase not in {TxPhase.IDLE, TxPhase.DONE, TxPhase.CANCELLED, TxPhase.FAILED}


@dataclass
class RxState:
    listening: bool = False
    source: str = "soundcard"
    gops: int = 0
    frames: int = 0
    freq_offset: float = 0.0
    sync_metric: float = 0.0
    snr_db: float = float("nan")
    callsign: str = ""
    message: str = ""


class Station:
    """Shared codec plus the current operator settings."""

    def __init__(self, settings: StationSettings | None = None):
        self.settings = settings or StationSettings()
        self.codec: AETVCodec | None = None
        self.codec_lock = threading.Lock()
        self._on_log = lambda _msg: None

    def set_logger(self, callback) -> None:
        self._on_log = callback

    def log(self, message: str) -> None:
        self._on_log(message)

    def checkpoint_path(self) -> Path:
        return resolve_checkpoint(self.settings.checkpoint or None)

    def load_codec(self) -> AETVCodec:
        device = self.settings.torch_device or None
        codec = AETVCodec(
            checkpoint=self.settings.checkpoint or None,
            device=device,
            mode=self.settings.mode,
        )
        with self.codec_lock:
            self.codec = codec
        return codec

    def require_codec(self) -> AETVCodec:
        if self.codec is None:
            raise RuntimeError("the V7 checkpoint is still loading")
        return self.codec

    def cat_config(self) -> CatConfig:
        geom = AETV_MODES[self.settings.mode].geometry
        return CatConfig(
            backend="none" if self.settings.audio_only else self.settings.cat_backend,
            rigctld_host=self.settings.rigctld_host,
            rigctld_port=self.settings.rigctld_port,
            flex_host=self.settings.flex_host,
            flex_power=self.settings.flex_power,
            flex_filter_low=int(geom.tx_bandpass[0]),
            flex_filter_high=int(geom.tx_bandpass[1]),
            freq_mhz=self.settings.freq_mhz,
            require_mode=self.settings.require_mode or None,
            serial_port=self.settings.serial_port,
            serial_line=self.settings.cat_backend if self.settings.cat_backend in {"rts", "dtr"} else "rts",
        )


class TxEngine:
    def __init__(self, station: Station, on_state=None, on_error=None, on_preview=None, ptt=None, player=None):
        self.station = station
        self._on_state = on_state or (lambda _state: None)
        self._on_error = on_error or (lambda _msg: None)
        self._on_preview = on_preview or (lambda _frames: None)
        self._ptt_override = ptt
        self._player = player
        self._cancel = threading.Event()
        self.state = TxState()
        self.last_wav: np.ndarray | None = None
        self.last_frames: np.ndarray | None = None

    def cancel(self) -> None:
        self._cancel.set()

    def _set(self, phase: TxPhase, progress: float | None = None, message: str = "") -> None:
        self.state = TxState(
            phase=phase,
            progress=self.state.progress if progress is None else progress,
            message=message,
        )
        self._on_state(self.state)

    def transmit(self, source: str) -> bool:
        self._cancel.clear()
        settings = self.station.settings
        try:
            codec = self.station.require_codec()
            frames = self._capture(source, codec)
            if frames is None:
                self._set(TxPhase.CANCELLED, 0.0, "cancelled")
                return False
            self.last_frames = frames
            self._on_preview(frames)
            latents = self._encode(frames, codec)
            if latents is None:
                self._set(TxPhase.CANCELLED, 0.0, "cancelled")
                return False
            self._set(TxPhase.MODULATING, 0.0, "modulating Flex-8k waveform")
            audio = modulate_gop_stream(latents, mode_name=codec.mode.name, callsign=settings.callsign)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0:
                audio = audio * (settings.tx_level / peak)
            self.last_wav = audio
            if self._cancel.is_set():
                self._set(TxPhase.CANCELLED, 0.0, "cancelled")
                return False
            return self._keyed_send(audio, codec.mode.geometry.fs)
        except Exception as error:
            self._on_error(str(error))
            self._set(TxPhase.FAILED, self.state.progress, str(error))
            return False

    def _capture(self, source: str, codec) -> np.ndarray | None:
        mode = codec.mode
        settings = self.station.settings
        if source.lower() in {"webcam", "cam", "camera"}:
            self._set(TxPhase.CAPTURING, 0.0, f"capturing {settings.gops} s from webcam")
            frames = []
            for index, frame in enumerate(iter_webcam(mode, camera=settings.camera_index, duration_s=settings.gops)):
                if self._cancel.is_set():
                    return None
                frames.append(frame)
                if (index + 1) % mode.gop_frames == 0:
                    self._on_preview(np.stack(frames[-mode.gop_frames :], axis=0))
                    self._set(TxPhase.CAPTURING, (index + 1) / (settings.gops * mode.gop_frames), f"captured GOP {(index + 1) // mode.gop_frames}/{settings.gops}")
            return collect_gops(np.stack(frames, axis=0), mode)
        self._set(TxPhase.CAPTURING, 0.0, f"reading {source}")
        return collect_gops(iter_video_file(source, mode, frames=settings.gops * mode.gop_frames), mode)

    def _encode(self, frames: np.ndarray, codec) -> list[np.ndarray] | None:
        mode = codec.mode
        n_gops = frames.shape[0] // mode.gop_frames
        latents = []
        with self.station.codec_lock:
            for index in range(n_gops):
                if self._cancel.is_set():
                    return None
                self._set(TxPhase.ENCODING, index / max(1, n_gops), f"encoding GOP {index + 1}/{n_gops}")
                gop = frames[index * mode.gop_frames : (index + 1) * mode.gop_frames]
                latents.append(codec.encode_gop(gop))
        return latents

    def _keyed_send(self, wave: np.ndarray, fs: int) -> bool:
        settings = self.station.settings
        duration_s = len(wave) / fs
        if self._ptt_override is not None:
            ptt = self._ptt_override
        elif settings.audio_only:
            ptt = NullPtt()
        else:
            ptt = open_ptt(self.station.cat_config())
        player = self._player or play_cancellable
        watchdog = _PttWatchdog(
            ptt,
            timeout_s=settings.ptt_lead_s + duration_s + settings.ptt_tail_s + WATCHDOG_MARGIN_S,
            on_fire=lambda: self._on_error(
                "PTT watchdog fired: transmission overran its expected duration; forcing receive"
            ),
        )
        try:
            self._set(TxPhase.KEYING, 0.0, ptt.describe())
            self._key(ptt, True)
            watchdog.start()
            if self._cancel.wait(settings.ptt_lead_s):
                self._set(TxPhase.CANCELLED, 0.0, "cancelled")
                return False
            self._set(TxPhase.SENDING, 0.0, f"sending {duration_s:.1f} s")
            completed = player(
                wave,
                fs,
                device=settings.audio_output or None,
                should_stop=self._cancel.is_set,
                on_progress=self._report_progress,
            )
            if not completed:
                self._set(TxPhase.CANCELLED, self.state.progress, "cancelled")
                return False
            self._set(TxPhase.UNKEYING, 1.0, "unkeying")
            self._cancel.wait(settings.ptt_tail_s)
        except Exception as error:
            self._on_error(str(error))
            self._set(TxPhase.FAILED, self.state.progress, str(error))
            return False
        finally:
            watchdog.cancel()
            self._key(ptt, False)
            if hasattr(ptt, "close"):
                try:
                    ptt.close()
                except Exception:
                    pass
        self._set(TxPhase.DONE, 1.0, "sent")
        return True

    def _key(self, ptt, on: bool) -> None:
        try:
            ptt.set_ptt(on)
        except Exception as error:
            if on:
                self._on_error(f"PTT on failed: {error}")
            else:
                self._on_error(f"PTT OFF FAILED: {error} — the rig may still be transmitting")

    def _report_progress(self, frac: float) -> None:
        self.state.progress = float(frac)
        self._on_state(self.state)


class _PttWatchdog:
    def __init__(self, ptt, timeout_s: float, on_fire=None):
        self._ptt = ptt
        self._timeout_s = timeout_s
        self._on_fire = on_fire or (lambda: None)
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if isinstance(self._ptt, NullPtt):
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="ptt-watchdog")
        self._thread.start()

    def _run(self) -> None:
        if self._done.wait(self._timeout_s):
            return
        try:
            self._ptt.set_ptt(False)
        except Exception:
            pass
        self._on_fire()

    def cancel(self) -> None:
        self._done.set()


class RxEngine:
    def __init__(self, station: Station, on_state=None, on_error=None, on_video=None, on_ring=None):
        self.station = station
        self._on_state = on_state or (lambda _state: None)
        self._on_error = on_error or (lambda _msg: None)
        self._on_video = on_video or (lambda _video, _info: None)
        self._on_ring = on_ring or (lambda _ring: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._kiwi: KiwiCapture | None = None
        self.ring: RingBuffer | None = None
        self.state = RxState()
        self.last_video: np.ndarray | None = None
        self._shown_gops = 0

    @property
    def listening(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        codec = self.station.require_codec()
        settings = self.station.settings
        self._stop.clear()
        self._shown_gops = 0
        self.ring = RingBuffer(settings.buffer_seconds, codec.mode.geometry.fs)
        self._on_ring(self.ring)
        self.state = RxState(listening=True, source=settings.rx_source, message="starting")
        self._on_state(self.state)
        if settings.rx_source == "kiwi":
            self._kiwi = KiwiCapture(
                host=settings.kiwi_host,
                dial_mhz=settings.kiwi_dial_mhz,
                fcenter_hz=codec.mode.geometry.fcenter_hz,
                dst_rate=codec.mode.geometry.fs,
                ring=self.ring,
                user=settings.kiwi_user or settings.callsign,
                password=settings.kiwi_password,
                on_status=self._on_kiwi_status,
                on_error=self._on_error,
            )
            self._kiwi.start()
        else:
            self._stream, _rate = open_input_stream(
                settings.audio_input or None,
                self.ring,
                codec.mode.geometry.fs,
                on_error=self._on_error,
            )
        self._thread = threading.Thread(target=self._loop, name="aetv-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._kiwi is not None:
            self._kiwi.stop()
            self._kiwi = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        thread = self._thread
        if thread is not None:
            thread.join(timeout=4.0)
        self._thread = None
        self.ring = None
        self._on_ring(None)
        self.state = RxState(listening=False, message="stopped")
        self._on_state(self.state)

    def _on_kiwi_status(self, status) -> None:
        self.state.message = status.message
        self.state.source = "kiwi"
        self._on_state(self.state)

    def _loop(self) -> None:
        codec = self.station.require_codec()
        while not self._stop.wait(self.station.settings.decode_every_s):
            ring = self.ring
            if ring is None:
                break
            audio, _total = ring.snapshot()
            if audio.size < codec.mode.geometry.fs:
                self.state.message = "listening"
                self._on_state(self.state)
                continue
            try:
                with self.station.codec_lock:
                    result = demodulate_gop_stream(audio, band=codec.mode.band, drift_track="off")
                    if len(result.gops_latents) <= self._shown_gops:
                        self._update_from_result(result, decoded=None)
                        continue
                    decoded = []
                    for latents, weights in zip(result.gops_latents, result.gops_weights):
                        decoded.append(codec.decode_gop(latents, weights))
            except SyncError as error:
                self.state.message = str(error)
                self._on_state(self.state)
                continue
            except Exception as error:
                self._on_error(str(error))
                continue
            video = np.concatenate(decoded, axis=0)
            self.last_video = video
            self._shown_gops = len(result.gops_latents)
            self._update_from_result(result, video)
            if self.station.settings.autosave:
                self._autosave(video, result)

    def _update_from_result(self, result, decoded) -> None:
        self.state = RxState(
            listening=True,
            source=self.station.settings.rx_source,
            gops=len(result.gops_latents),
            frames=result.frames_received,
            freq_offset=result.freq_offset,
            sync_metric=result.sync_metric,
            snr_db=result.snr_db,
            callsign=result.callsign,
            message=f"de {result.callsign or '????'}  {len(result.gops_latents)} GOP  SNR {result.snr_db:.1f} dB",
        )
        self._on_state(self.state)
        if decoded is not None:
            self._on_video(decoded, self.state)

    def _autosave(self, video: np.ndarray, result) -> None:
        folder = self.station.settings.receive_path()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        call = result.callsign or "unknown"
        path = folder / f"{stamp}_{call}_{self.station.settings.mode}.mp4"
        try:
            write_mp4(video, path, self.station.require_codec().mode.fps)
            self.station.log(f"saved {path}")
        except Exception as error:
            self._on_error(f"autosave failed: {error}")

    def save_current(self, path: Path | None = None) -> Path | None:
        if self.last_video is None:
            return None
        folder = self.station.settings.receive_path()
        folder.mkdir(parents=True, exist_ok=True)
        target = path or folder / time.strftime("aetv_%Y%m%d-%H%M%S.mp4")
        write_mp4(self.last_video, Path(target), self.station.require_codec().mode.fps)
        return Path(target)
