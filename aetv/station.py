"""Ham-station TX/RX engines: encode, key, play, unkey; capture, demodulate, decode.

The one transmit rule: PTT always comes back down. A cancelled send, an
exception, or a wedged audio callback must unkey. The keyed region is
wrapped in try/finally, and a watchdog thread drops PTT if the
transmission overruns its known duration.
"""

from __future__ import annotations

import json
import threading
import time
import queue
import re
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .audio_io import (
    StreamResampler,
    open_input_stream,
    play_cancellable,
    play_chunk_stream,
    resample_ratio,
)
from .cat import CatConfig, NullPtt, open_ptt
from .codec import AETVCodec, resolve_checkpoint
from .config import AETV_MODES, AETVModeSpec
from .kiwi import KiwiCapture
from .flex import FlexVitaSession
from .hfchannel import CHANNEL_PROFILES, StreamingChannelEmulator
from .modem import (
    StreamingDemodulator,
    modulate_continuous_chunks,
    modulate_gop_chunks,
    modulate_gop_stream,
)
from .ringbuffer import RingBuffer
from .settings import StationSettings
from .source import collect_gops, iter_video_file, iter_webcam, write_mp4
from .sync import SyncError

WATCHDOG_MARGIN_S = 15.0


def _debug_prefix(settings: StationSettings, label: str) -> Path:
    folder = settings.receive_path() / "debug"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    millis = (time.time_ns() // 1_000_000) % 1000
    return folder / f"{stamp}-{millis:03d}_{safe}"


class _PcmWaveRecorder:
    """Incrementally save float audio without retaining a transmission in RAM."""

    def __init__(self, path: Path, rate: int, channels: int = 1):
        self.path = Path(path)
        self.rate = int(rate)
        self.channels = int(channels)
        self.samples = 0
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(self.channels)
        self._wave.setsampwidth(2)
        self._wave.setframerate(self.rate)

    def write(self, values: np.ndarray) -> None:
        array = np.asarray(values)
        if self.channels == 1:
            array = array.reshape(-1)
            self.samples += len(array)
        else:
            array = array.reshape(-1, self.channels)
            self.samples += len(array)
        pcm = np.rint(np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
        self._wave.writeframesraw(pcm.tobytes())

    def close(self) -> None:
        self._wave.close()


class _JsonlRecorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: dict) -> None:
        with self._lock:
            self._file.write(json.dumps(event, allow_nan=True, sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


class _KiwiIqRecorder:
    def __init__(self, prefix: Path, metadata: dict):
        self.prefix = Path(prefix)
        self.metadata = dict(metadata)
        self.writer: _PcmWaveRecorder | None = None
        self.blocks = 0
        self.discontinuities: list[dict] = []

    def write(self, iq: np.ndarray, rate: float, sequence: int) -> None:
        if self.writer is None:
            self.writer = _PcmWaveRecorder(
                self.prefix.with_suffix(".iq.wav"), round(rate), channels=2
            )
            self.metadata["kiwi_sample_rate_exact"] = float(rate)
            self.metadata["wav_sample_rate"] = round(rate)
        interleaved = np.column_stack((np.real(iq), np.imag(iq)))
        self.writer.write(interleaved)
        self.blocks += 1
        self.metadata["last_sequence"] = int(sequence)

    def discontinuity(self, reason: str) -> None:
        samples = self.writer.samples if self.writer is not None else 0
        self.discontinuities.append(
            {"sample": samples, "time": time.time(), "reason": reason}
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.metadata["iq_samples"] = self.writer.samples
            self.metadata["iq_wav"] = str(self.writer.path)
        self.metadata["blocks"] = self.blocks
        self.metadata["discontinuities"] = self.discontinuities
        self.metadata["stopped_at"] = time.time()
        self.prefix.with_suffix(".iq.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )


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
        return resolve_checkpoint(
            self.settings.checkpoint or None,
            mode=self.settings.mode,
        )

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
            raise RuntimeError(f"the {self.settings.mode} checkpoint is still loading")
        if self.codec.mode.name != self.settings.mode:
            raise RuntimeError(
                f"the {self.settings.mode} checkpoint is still loading "
                f"(currently loaded: {self.codec.mode.name})"
            )
        return self.codec

    def cat_config(self) -> CatConfig:
        geom = AETV_MODES[self.settings.mode].geometry
        return CatConfig(
            backend="none" if self.settings.audio_only else self.settings.cat_backend,
            rigctld_host=self.settings.rigctld_host,
            rigctld_port=self.settings.rigctld_port,
            hamlib_model=self.settings.hamlib_model,
            hamlib_device=self.settings.hamlib_device,
            hamlib_baud=self.settings.hamlib_baud,
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
    def __init__(
        self,
        station: Station,
        on_state=None,
        on_error=None,
        on_preview=None,
        ptt=None,
        player=None,
        camera_frames=None,
        on_loopback=None,
    ):
        self.station = station
        self._on_state = on_state or (lambda _state: None)
        self._on_error = on_error or (lambda _msg: None)
        self._on_preview = on_preview or (lambda _frames: None)
        self._ptt_override = ptt
        self._player = player
        self._camera_frames = camera_frames
        self._on_loopback = on_loopback or (lambda _video, _state: None)
        self._cancel = threading.Event()
        self.state = TxState()
        self.last_wav: np.ndarray | None = None
        self.last_frames: np.ndarray | None = None
        self.gop_timings: list[dict] = []

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
        self.gop_timings = []
        settings = self.station.settings
        tx_recorder: _PcmWaveRecorder | None = None
        tx_metadata: dict = {}
        tx_prefix: Path | None = None
        try:
            codec = self.station.require_codec()
            n_gops = settings.gops
            if source.lower() in {"webcam", "cam", "camera"}:
                encoded_gops = self._live_webcam_gops(codec, n_gops)
            else:
                frames = self._capture(source, codec)
                if frames is None:
                    self._set(TxPhase.CANCELLED, 0.0, "cancelled")
                    return False
                self.last_frames = frames
                self._on_preview(frames)
                n_gops = frames.shape[0] // codec.mode.gop_frames

                def encoded_gops():
                    for index in range(n_gops):
                        if self._cancel.is_set():
                            return
                        phase = (
                            TxPhase.SENDING
                            if self.state.phase in {TxPhase.KEYING, TxPhase.SENDING}
                            else TxPhase.ENCODING
                        )
                        self._set(
                            phase,
                            index / max(1, n_gops),
                            f"encoding GOP {index + 1}/{n_gops}",
                        )
                        start = index * codec.mode.gop_frames
                        with self.station.codec_lock:
                            encode_started = time.perf_counter()
                            latent = codec.encode_gop(
                                frames[start : start + codec.mode.gop_frames]
                            )
                        self.gop_timings.append(
                            {
                                "gop": index + 1,
                                "device": str(getattr(codec, "device", "unknown")),
                                "encode_s": time.perf_counter() - encode_started,
                            }
                        )
                        yield latent

                encoded_gops = encoded_gops()

            chunks = modulate_continuous_chunks(
                encoded_gops, mode_name=codec.mode.name, callsign=settings.callsign
            )
            channel_profile = settings.tx_channel_profile

            if settings.debug_capture:
                tx_prefix = _debug_prefix(settings, f"tx_{codec.mode.name}_{settings.callsign}")
                tx_recorder = _PcmWaveRecorder(
                    tx_prefix.with_suffix(".tx.wav"), codec.mode.geometry.fs
                )
                tx_metadata = {
                    "started_at": time.time(),
                    "source": source,
                    "mode": codec.mode.name,
                    "callsign": settings.callsign,
                    "sample_rate": codec.mode.geometry.fs,
                    "requested_gops": n_gops,
                    "framing": "continuous",
                    "payload_gop_seconds": 1.0,
                    "expected_waveform_seconds": n_gops + 0.65,
                    "tx_level": settings.tx_level,
                    "frequency_mhz": settings.freq_mhz,
                    "flex_host": settings.flex_host,
                    "waveform": str(tx_recorder.path),
                    "gop_timings": self.gop_timings,
                    "channel_profile": channel_profile,
                }
                self.station.log(f"TX waveform debug: {tx_recorder.path}")

            def leveled_chunks():
                for index, audio in enumerate(chunks):
                    phase = (
                        TxPhase.SENDING
                        if self.state.phase in {TxPhase.KEYING, TxPhase.SENDING}
                        else TxPhase.MODULATING
                    )
                    self._set(
                        phase,
                        index / max(1, n_gops),
                        f"streaming GOP {index + 1}/{n_gops}",
                    )
                    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                    if peak > 0:
                        audio = audio * (settings.tx_level / peak)
                    self.last_wav = audio
                    if tx_recorder is not None and channel_profile == "radio":
                        tx_recorder.write(audio)
                    yield audio

            if channel_profile != "radio":
                return self._emulated_send_stream(
                    leveled_chunks(), codec.mode.geometry.fs, n_gops, codec,
                    channel_profile, tx_recorder,
                )
            return self._keyed_send_stream(leveled_chunks(), codec.mode.geometry.fs, n_gops)
        except Exception as error:
            self._on_error(str(error))
            self._set(TxPhase.FAILED, self.state.progress, str(error))
            return False
        finally:
            if tx_recorder is not None:
                tx_recorder.close()
                tx_metadata["samples"] = tx_recorder.samples
                tx_metadata["duration_s"] = tx_recorder.samples / tx_recorder.rate
                tx_metadata["stopped_at"] = time.time()
                assert tx_prefix is not None
                tx_prefix.with_suffix(".tx.json").write_text(
                    json.dumps(tx_metadata, indent=2, allow_nan=True) + "\n",
                    encoding="utf-8",
                )

    def _emulated_send_stream(
        self,
        chunks,
        fs: int,
        n_gops: int,
        codec,
        profile_key: str,
        recorder: _PcmWaveRecorder | None = None,
    ) -> bool:
        """Incrementally run TX through the channel and production RX modem."""
        profile = CHANNEL_PROFILES[profile_key]
        self._set(TxPhase.SENDING, 0.0, f"streaming {profile.label} into receiver")
        channel = StreamingChannelEmulator(profile, fs=fs)
        demodulator = StreamingDemodulator(
            codec.mode.band,
            continuous=True,
            mode_name=codec.mode.name,
        )
        decoded_count = 0
        transmitted_chunks = 0
        block_samples = max(1, fs // 10)
        stream_started: float | None = None
        delivered_samples = 0
        for clean in chunks:
            if self._cancel.is_set():
                self._set(TxPhase.CANCELLED, self.state.progress, "cancelled")
                return False
            impaired = channel.process(clean)
            peak = float(np.max(np.abs(impaired))) if impaired.size else 0.0
            if peak > 0:
                impaired *= self.station.settings.tx_level / peak
            self.last_wav = impaired
            if stream_started is None:
                # Encoding happens before the first audio buffer exists; the
                # emulated on-air clock starts only once that buffer is ready.
                stream_started = time.monotonic()
            if recorder is not None:
                recorder.write(impaired)
            transmitted_chunks += 1
            for start in range(0, len(impaired), block_samples):
                block = impaired[start : start + block_samples]
                # A local channel has no soundcard to impose wall-clock
                # timing. Deliver each block when its final sample would have
                # arrived over the air so decoded GOPs cannot overrun the
                # GUI's real-time playout queue.
                delivered_samples += len(block)
                deadline = stream_started + delivered_samples / fs
                delay = deadline - time.monotonic()
                if delay > 0 and self._cancel.wait(delay):
                    self._set(TxPhase.CANCELLED, self.state.progress, "cancelled")
                    return False
                results = demodulator.feed(block)
                for result in results:
                    for latents, weights in zip(result.gops_latents, result.gops_weights):
                        with self.station.codec_lock:
                            decoded = codec.decode_gop(latents, weights)
                        decoded_count += 1
                        state = RxState(
                            listening=False,
                            source="emulator",
                            gops=decoded_count,
                            frames=decoded_count * codec.mode.gop_frames,
                            freq_offset=result.freq_offset,
                            sync_metric=result.sync_metric,
                            snr_db=result.snr_db,
                            callsign=result.callsign,
                            message=(
                                f"{profile.label} loopback  {decoded_count}/{n_gops} GOP  "
                                f"SNR {result.snr_db:.1f} dB"
                            ),
                        )
                        self._on_loopback(decoded, state)
                        self._set(
                            TxPhase.SENDING,
                            decoded_count / max(1, n_gops),
                            state.message,
                        )
        if transmitted_chunks == 0:
            raise SyncError("modulator produced no loopback audio")
        if decoded_count == 0:
            raise SyncError(f"{profile.label} loopback recovered no GOPs")
        self._set(
            TxPhase.DONE,
            1.0,
            f"{profile.label} loopback decoded {decoded_count}/{n_gops} GOPs",
        )
        return True

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
            if not self._key(ptt, True):
                self._set(TxPhase.FAILED, 0.0, "PTT on failed")
                return False
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

    def _keyed_send_stream(self, chunks, fs: int, n_gops: int) -> bool:
        """Key once while GOPs are encoded, modulated, and written incrementally."""
        settings = self.station.settings
        # Build a small rolling buffer before keying. The previous path first
        # pulled the lazy generator *after* xmit=1, leaving the transmitter
        # keyed with no audio while a CPU encoded the first GOP. A producer
        # keeps later GOPs moving while the consumer paces radio audio.
        # Reserve queue capacity *before* advancing ``chunks``.  A regular
        # bounded Queue alone is not sufficient here: ``for chunk in chunks``
        # encodes the next GOP before put() discovers that the queue is full.
        # During startup that let the producer encode several seconds of video
        # back-to-back while the camera and radio were being handed over,
        # starving the GUI even though this work runs on a Python thread.
        #
        # A single look-ahead GOP is enough to overlap encoding with the
        # one-second audio consumer without creating that startup burst.
        ready: queue.Queue = queue.Queue(maxsize=1)
        free_slots = threading.Semaphore(1)
        producer_stop = threading.Event()
        sentinel = object()

        def produce() -> None:
            iterator = iter(chunks)
            while not producer_stop.is_set():
                if not free_slots.acquire(timeout=0.2):
                    continue
                if producer_stop.is_set():
                    free_slots.release()
                    return
                try:
                    item = next(iterator)
                except StopIteration:
                    item = sentinel
                except Exception as error:
                    item = error
                ready.put(item)
                if item is sentinel or isinstance(item, Exception):
                    return

        self._set(TxPhase.ENCODING, 0.0, "preparing first live GOP before PTT")
        producer = threading.Thread(target=produce, name="aetv-tx-producer", daemon=True)
        producer.start()
        first = ready.get()
        free_slots.release()
        if isinstance(first, Exception):
            self._on_error(str(first))
            self._set(TxPhase.FAILED, 0.0, str(first))
            producer_stop.set()
            producer.join(timeout=5.0)
            return False
        if first is sentinel:
            self._set(TxPhase.FAILED, 0.0, "encoder produced no GOP audio")
            producer_stop.set()
            producer.join(timeout=5.0)
            return False

        def prepared_chunks():
            yield first
            index = 1
            while True:
                wait_started = time.perf_counter()
                item = ready.get()
                free_slots.release()
                wait_s = time.perf_counter() - wait_started
                if item is sentinel:
                    return
                if isinstance(item, Exception):
                    raise item
                if index < len(self.gop_timings):
                    self.gop_timings[index]["tx_buffer_wait_s"] = wait_s
                if wait_s >= 0.02:
                    self.station.log(
                        f"TX buffer warning: waited {wait_s * 1000:.0f} ms for GOP {index + 1}"
                    )
                index += 1
                yield item

        chunks = prepared_chunks()
        flex_session = None
        if (
            self._ptt_override is None
            and not settings.audio_only
            and settings.cat_backend == "flex"
            and settings.flex_native_audio
        ):
            tx_geometry = AETV_MODES[settings.mode].geometry
            self.station.log(
                f"Flex {settings.mode} TX mask: "
                f"{int(tx_geometry.tx_bandpass[0])}-"
                f"{int(tx_geometry.tx_bandpass[1])} Hz"
            )
            flex_session = FlexVitaSession(
                settings.flex_host,
                frequency_mhz=settings.freq_mhz,
                mode=settings.require_mode or "DIGU",
                power=settings.flex_power,
                filter_low=int(tx_geometry.tx_bandpass[0]),
                filter_high=int(tx_geometry.tx_bandpass[1]),
            )
            # Stream creation and DAX ownership must complete while receiving;
            # otherwise the first VITA packets can arrive before the radio has
            # associated this client with its transmit-audio stream.
            flex_session.prepare_tx()
            ptt = flex_session
        elif self._ptt_override is not None:
            ptt = self._ptt_override
        elif settings.audio_only:
            ptt = NullPtt()
        else:
            ptt = open_ptt(self.station.cat_config())
        # Each synchronized chunk is about 1.34 seconds. Encoding time does not
        # count against this generous watchdog allowance.
        watchdog = _PttWatchdog(
            ptt,
            timeout_s=settings.ptt_lead_s + n_gops * 3.0 + settings.ptt_tail_s + WATCHDOG_MARGIN_S,
            on_fire=lambda: self._on_error("PTT watchdog fired; forcing receive"),
        )
        try:
            self._set(TxPhase.KEYING, 0.0, ptt.describe())
            if not self._key(ptt, True):
                self._set(TxPhase.FAILED, 0.0, "PTT on failed")
                return False
            watchdog.start()
            if self._cancel.wait(settings.ptt_lead_s):
                return False
            self._set(TxPhase.SENDING, 0.0, "encoding and sending live GOPs")
            if flex_session is not None:
                flex_chunks = chunks
                flex_rate = fs
                if fs != 24000:
                    flex_resampler = StreamResampler(*resample_ratio(fs, 24000))
                    flex_chunks = (flex_resampler(chunk) for chunk in chunks)
                    flex_rate = 24000
                completed = flex_session.send_audio_stream(
                    flex_chunks,
                    flex_rate,
                    should_stop=self._cancel.is_set,
                    on_chunk=lambda count: self._report_progress(
                        count / max(1, n_gops)
                    ),
                )
            else:
                completed = play_chunk_stream(
                    chunks,
                    fs,
                    device=settings.audio_output or None,
                    should_stop=self._cancel.is_set,
                    on_chunk=lambda count: self._report_progress(count / max(1, n_gops)),
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
            producer_stop.set()
            watchdog.cancel()
            self._key(ptt, False)
            try:
                ptt.close()
            except Exception:
                pass
            producer.join(timeout=30.0)
        self._set(TxPhase.DONE, 1.0, "sent")
        return True

    def _live_webcam_gops(self, codec, n_gops: int):
        """Capture, encode, and yield camera GOPs while the prior GOP transmits."""
        mode = codec.mode
        if self._camera_frames is None:
            camera_frames = iter_webcam(
                mode, camera=self.station.settings.camera_index
            )
        else:
            camera_frames = self._camera_frames(
                mode,
                camera=self.station.settings.camera_index,
                should_stop=self._cancel.is_set,
            )
        try:
            for index in range(n_gops):
                capture_started = time.perf_counter()
                frames = []
                for _ in range(mode.gop_frames):
                    if self._cancel.is_set():
                        return
                    frames.append(next(camera_frames))
                gop = np.stack(frames, axis=0)
                self.last_frames = gop
                self._on_preview(gop)
                phase = (
                    TxPhase.SENDING
                    if self.state.phase in {TxPhase.KEYING, TxPhase.SENDING}
                    else TxPhase.CAPTURING
                )
                self._set(
                    phase,
                    index / max(1, n_gops),
                    f"live GOP {index + 1}/{n_gops}: encoding",
                )
                with self.station.codec_lock:
                    encode_started = time.perf_counter()
                    latent = codec.encode_gop(gop)
                self.gop_timings.append(
                    {
                        "gop": index + 1,
                        "device": str(getattr(codec, "device", "unknown")),
                        "capture_s": encode_started - capture_started,
                        "encode_s": time.perf_counter() - encode_started,
                        "prepare_s": time.perf_counter() - capture_started,
                    }
                )
                yield latent
        finally:
            close = getattr(camera_frames, "close", None)
            if close is not None:
                close()

    def _key(self, ptt, on: bool) -> bool:
        try:
            ptt.set_ptt(on)
            return True
        except Exception as error:
            if on:
                self._on_error(f"PTT on failed: {error}")
            else:
                self._on_error(f"PTT OFF FAILED: {error} — the rig may still be transmitting")
            return False

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
        self._flex: FlexVitaSession | None = None
        self.ring: RingBuffer | None = None
        self.state = RxState()
        self.last_video: np.ndarray | None = None
        self._shown_gops = 0
        self._stream_decoder: StreamingDemodulator | None = None
        self._kiwi_discontinuity = threading.Event()
        self._read_cursor = 0
        self._last_result = None
        self._debug_log: _JsonlRecorder | None = None
        self._iq_recorder: _KiwiIqRecorder | None = None

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
        self.last_video = None
        self._last_result = None
        self.ring = RingBuffer(settings.buffer_seconds, codec.mode.geometry.fs)
        if settings.debug_capture:
            prefix = _debug_prefix(settings, f"rx_{settings.rx_source}")
            self._debug_log = _JsonlRecorder(prefix.with_suffix(".modem.jsonl"))
            self.station.log(f"RX modem debug: {self._debug_log.path}")
            if settings.rx_source == "kiwi":
                self._iq_recorder = _KiwiIqRecorder(
                    prefix,
                    {
                        "started_at": time.time(),
                        "host": settings.kiwi_host,
                        "dial_mhz": settings.kiwi_dial_mhz,
                        "iq_center_khz": codec.mode.geometry.fcenter_hz / 1000.0
                        + settings.kiwi_dial_mhz * 1000.0,
                        "mode": codec.mode.name,
                        "destination_rate": codec.mode.geometry.fs,
                    },
                )
                self.station.log(f"Kiwi IQ debug: {prefix.with_suffix('.iq.wav')}")
        self._stream_decoder = self._new_demodulator(codec.mode)
        self._kiwi_discontinuity.clear()
        self._read_cursor = 0
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
                on_discontinuity=self._on_kiwi_discontinuity,
                on_iq=self._record_kiwi_iq,
            )
            self._kiwi.start()
        elif settings.rx_source == "flex":
            self._flex = FlexVitaSession(
                settings.flex_host,
                frequency_mhz=settings.freq_mhz,
                mode=settings.require_mode or "DIGU",
                power=settings.flex_power,
                filter_low=int(codec.mode.geometry.tx_bandpass[0]),
                filter_high=int(codec.mode.geometry.tx_bandpass[1]),
            )
            if codec.mode.geometry.fs == 24000:
                write_flex = self.ring.write
            else:
                resample_flex = StreamResampler(*resample_ratio(24000, codec.mode.geometry.fs))
                write_flex = lambda chunk: self.ring.write(resample_flex(chunk))
            self._flex.start_rx(write_flex)
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
        if self._flex is not None:
            self._flex.close()
            self._flex = None
        thread = self._thread
        if thread is not None:
            thread.join(timeout=4.0)
        self._thread = None
        if self.station.settings.autosave and self.last_video is not None and self._last_result is not None:
            self._autosave(self.last_video, self._last_result)
        self.ring = None
        if self._iq_recorder is not None:
            self._iq_recorder.close()
            self._iq_recorder = None
        if self._debug_log is not None:
            self._debug_log.close()
            self._debug_log = None
        self._on_ring(None)
        self.state = RxState(listening=False, message="stopped")
        self._on_state(self.state)

    def _on_kiwi_status(self, status) -> None:
        self.state.message = status.message
        self.state.source = "kiwi"
        self._on_state(self.state)

    def _record_kiwi_iq(self, iq: np.ndarray, rate: float, sequence: int) -> None:
        if self._iq_recorder is not None:
            self._iq_recorder.write(iq, rate, sequence)

    def _on_kiwi_discontinuity(self) -> None:
        if self._iq_recorder is not None:
            self._iq_recorder.discontinuity("Kiwi sequence gap or reconnect")
        self._kiwi_discontinuity.set()

    def _record_modem_debug(self, event: dict) -> None:
        if self._debug_log is not None:
            self._debug_log.write(event)

    def _new_demodulator(self, mode: AETVModeSpec) -> StreamingDemodulator:
        return StreamingDemodulator(
            mode.band,
            on_debug=self._record_modem_debug,
            continuous=True,
            mode_name=mode.name,
        )

    def _loop(self) -> None:
        codec = self.station.require_codec()
        interval = max(0.05, float(self.station.settings.decode_every_s))
        next_poll = time.monotonic()
        while True:
            if self._stop.wait(max(0.0, next_poll - time.monotonic())):
                break
            next_poll += interval
            ring = self.ring
            if ring is None:
                break
            if self._kiwi_discontinuity.is_set():
                self._kiwi_discontinuity.clear()
                self._stream_decoder = self._new_demodulator(codec.mode)
                # Drop every sample written before the reset. The interrupted
                # GOP cannot be repaired live; the next independently framed
                # GOP will provide a fresh preamble and mode header.
                _discarded, self._read_cursor, _overrun = ring.read_since(2**63 - 1)
                self.state.message = "Kiwi stream gap; reacquiring next GOP"
                self._on_state(self.state)
                continue
            audio, self._read_cursor, overrun = ring.read_since(self._read_cursor)
            if overrun:
                self._stream_decoder = self._new_demodulator(codec.mode)
                self.state.message = "receive buffer overrun; reacquiring"
            if audio.size == 0:
                continue
            try:
                demodulator = self._stream_decoder
                if demodulator is None:
                    demodulator = self._stream_decoder = self._new_demodulator(codec.mode)
                demod_started = time.perf_counter()
                results = demodulator.feed(audio)
                demod_s = time.perf_counter() - demod_started
                for result in results:
                    for latents, weights in zip(result.gops_latents, result.gops_weights):
                        decode_started = time.perf_counter()
                        with self.station.codec_lock:
                            decoded = codec.decode_gop(latents, weights)
                        decode_s = time.perf_counter() - decode_started
                        with ring.lock:
                            backlog_s = max(
                                0.0,
                                (ring.total_written - self._read_cursor) / ring.fs,
                            )
                        self._record_modem_debug(
                            {
                                "event": "gop_decoded",
                                "time": time.time(),
                                "device": str(codec.device),
                                "decode_ms": 1000.0 * decode_s,
                                "demod_batch_ms": 1000.0 * demod_s,
                                "rx_backlog_s": backlog_s,
                            }
                        )
                        self._shown_gops += 1
                        self._last_result = result
                        if self.last_video is None:
                            self.last_video = decoded
                        else:
                            self.last_video = np.concatenate([self.last_video, decoded], axis=0)
                            max_frames = codec.mode.gop_frames * 300
                            self.last_video = self.last_video[-max_frames:]
                        self._update_from_result(result, decoded)
            except SyncError as error:
                self.state.message = str(error)
                self._on_state(self.state)
                continue
            except Exception as error:
                self._on_error(str(error))
                continue
            if next_poll < time.monotonic():
                next_poll = time.monotonic()

    def _update_from_result(self, result, decoded) -> None:
        identity = f"de {result.callsign}" if result.callsign else "beacon acquiring"
        self.state = RxState(
            listening=True,
            source=self.station.settings.rx_source,
            gops=self._shown_gops,
            frames=self._shown_gops * AETV_MODES[self.station.settings.mode].gop_frames,
            freq_offset=result.freq_offset,
            sync_metric=result.sync_metric,
            snr_db=result.snr_db,
            callsign=result.callsign,
            message=f"{identity}  {self._shown_gops} GOP  SNR {result.snr_db:.1f} dB",
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
