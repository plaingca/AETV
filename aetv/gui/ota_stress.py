"""Opt-in observation/fault-injection harness for the ordinary live GUI.

Fixtures replace only camera/microphone inputs. RF, codecs, receiver, audio
output and GUI timers run normally. No source references reach acquisition.
"""

import hashlib
import gc
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer

from aetv.codec import sha256_file
from aetv.config import AETV_MODES
from aetv.recording import BufferedWriter


class _TraceFile:
    """Observation must not block the thread being measured on slow storage."""

    def __init__(self, path):
        self._file = path.open("wb")
        self._writer = BufferedWriter(self._file.write, self._file.close)
        self.health = self._writer.health

    def write(self, data):
        return self._writer.write(data.encode("utf-8") if isinstance(data, str) else data)

    def close(self):
        self._writer.close()


def install_stress(window, app, config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    source = np.load(config["source"], mmap_mode="r")
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB video fixture")
    mode = AETV_MODES[window.settings.mode]
    count = window.settings.gops
    # The normal picker remains unchanged; this explicit harness can run longer.
    window.tx.gops.setMaximum(max(300, count))
    window.tx.gops.setValue(count)
    source_fps = float(config.get("source_fps", 10))
    cancel = threading.Event()
    started = time.monotonic()
    state = dict(config=config, mode=mode.name, packaged=bool(getattr(sys, "frozen", False)),
                 source_sha256=sha256_file(Path(config["source"])), started=started,
                 input_frames=[], consumed_frames=[], source_dropped_frames=0, encode=[], decode=[], displayed=[], delivered=[],
                 audio_writes=[], errors=[], logs=[], faults=[], tx_io=[])
    state["gc_events"] = []

    def garbage_collection(phase, info):
        state["gc_events"].append(dict(time=time.monotonic(), phase=phase, **info))

    gc.callbacks.append(garbage_collection)
    status = dict(started=False, finished=False, tx_done=None, epoch=None,
                  rx_started=False, actions=set(), rf_start=None, last_health=0)
    telemetry = _TraceFile(output / "health.jsonl")
    raw_file = (output / "rtl.cu8").open("wb") if config.get("save_iq", True) else None
    raw_queue = queue.Queue(80)
    recorder_health = dict(queue_high_water=0, dropped_blocks=0, write_max_ms=0.)
    raw_end = object()

    def record_iq():
        while True:
            item = raw_queue.get()
            if item is raw_end:
                return
            before = time.monotonic()
            raw_file.write(item)
            recorder_health["write_max_ms"] = max(recorder_health["write_max_ms"], 1000*(time.monotonic()-before))

    raw_thread = threading.Thread(target=record_iq, daemon=True, name="stress-iq-recorder") if raw_file else None
    if raw_thread is not None:
        raw_thread.start()
    audio_trace = _TraceFile(output / "audio-output.jsonl")
    sent, received = [], []
    rendered = _TraceFile(output / "decoded.rgb")
    frame_ids = {}
    shown = window.rx.preview._show_frame

    def fingerprint(frame):
        return hashlib.blake2s(memoryview(np.ascontiguousarray(frame)), digest_size=12).hexdigest()

    def show(frame):
        now = time.monotonic()
        state["displayed"].append(dict(time=now, decoded_frame=frame_ids.get(fingerprint(frame))))
        shown(frame)

    window.rx.preview._show_frame = show
    original_log = window.station.log

    def log(message):
        state["logs"].append(dict(time=time.monotonic(), message=str(message)))
        original_log(message)

    window.station.log = log
    original_error = window.rx.engine._on_error

    def error(message):
        state["errors"].append(dict(time=time.monotonic(), message=str(message)))
        original_error(message)

    window.rx.engine._on_error = error
    original_deliver = window.rx.engine._deliver_received

    def deliver(result, decoded, audio):
        row = dict(time=time.monotonic(), stream_start_sample=result.stream_start_sample,
                   freq_offset=float(result.freq_offset), snr_db=float(result.snr_db),
                   frame_counter=result.stream_frame_counter,
                   audio_samples=0 if audio is None else len(audio))
        if audio is not None:
            audio_recording.write(np.asarray(audio, np.float32).tobytes())
        state["delivered"].append(row)
        return original_deliver(result, decoded, audio)

    audio_recording = _TraceFile(output / "paired-audio.f32")
    window.rx.engine._deliver_received = deliver

    # Observe original PortAudio writes, including its returned underflow flag.
    import sounddevice as sd
    original_audio_write = sd.OutputStream.write

    def audio_write(stream, samples):
        begin = time.monotonic()
        underflow = original_audio_write(stream, samples)
        state["audio_writes"].append(dict(start=begin, end=time.monotonic(), samples=len(samples),
                                           rate=stream.samplerate, underflow=bool(underflow)))
        audio_trace.write(json.dumps(state["audio_writes"][-1]) + "\n")
        return underflow

    sd.OutputStream.write = audio_write

    # Observe the actual IIO writes without replacing driver behavior.
    import aetv.sdr as sdr
    original_open = sdr.open_pluto

    def open_radio(uri):
        radio = original_open(uri)
        original_tx = radio.tx

        def write_tx(samples):
            before = time.monotonic()
            if status["rf_start"] is None:
                status["rf_start"] = before
            result = original_tx(samples)
            state["tx_io"].append(dict(start=before, end=time.monotonic(), samples=len(samples)))
            return result

        radio.tx = write_tx
        status["tx_radio"] = radio
        return radio

    sdr.open_pluto = open_radio

    def frames(_mode, **_kwargs):
        pending = queue.Queue(max(2, round(2 * mode.fps)))
        epoch = time.monotonic()
        status["epoch"] = epoch

        producer_stop = threading.Event()

        def feed():
            index = 0
            while not producer_stop.is_set() and not cancel.is_set():
                deadline = epoch + index / mode.fps
                if producer_stop.wait(max(0, deadline - time.monotonic())):
                    break
                source_index = round(index / mode.fps * source_fps) % len(source)
                frame = source[source_index]
                if frame.shape[:2] != (mode.height, mode.width):
                    frame = np.asarray(Image.fromarray(frame).resize((mode.width, mode.height), Image.Resampling.LANCZOS))
                provenance = dict(index=index, source_index=source_index,
                                  available=time.monotonic(), scheduled=deadline)
                state["input_frames"].append(provenance)
                item = (np.array(frame), provenance)
                try:
                    pending.put_nowait(item)
                except queue.Full:
                    # Live capture keeps recent frames when inference stalls.
                    # Account for the skipped source time in latency/quality
                    # scoring instead of terminating an injected-fault trial.
                    try:
                        pending.get_nowait()
                        state["source_dropped_frames"] += 1
                    except queue.Empty:
                        pass
                    pending.put_nowait(item)
                index += 1

        producer = threading.Thread(target=feed, daemon=True, name="stress-rgb")
        producer.start()
        try:
            for _ in range(count * mode.gop_frames):
                frame, provenance = pending.get(timeout=5)
                state["consumed_frames"].append(dict(provenance, consumed=time.monotonic()))
                yield frame
        finally:
            producer_stop.set()
            producer.join(timeout=3)

    # A different tone identifies every source second within a 60-second cycle.
    original_composite = window.tx.engine._composite_chunks

    def composite(chunks, _voice, n_gops, **_kwargs):
        t = np.arange(8000) / 8000
        voice = np.concatenate([.2 * np.sin(2*np.pi*(400 + 23*(i % 60))*t) for i in range(n_gops)])
        return original_composite(chunks, voice, n_gops, capture_microphone=False)

    window.tx.engine._composite_chunks = composite

    def start_rx():
        if not window.rx.start():
            error(window.rx.status.text())
            return
        status["rx_started"] = True
        cap = window.rx.engine._sdr
        if cap is not None:
            original_get = cap._queue.get

            def get_iq(*args, **kwargs):
                elapsed = time.monotonic() - (status["rf_start"] or time.monotonic())
                for index, fault in enumerate(config.get("converter_stalls", [])):
                    key = ("converter", index)
                    if key not in status["actions"] and elapsed >= fault["at_s"]:
                        status["actions"].add(key)
                        state["faults"].append(dict(kind="converter_stall", time=time.monotonic(), **fault))
                        cap._stop.wait(float(fault["seconds"]))
                return original_get(*args, **kwargs)

            cap._queue.get = get_iq
        if cap is not None and raw_file is not None:
            original_iq = cap.preview.write

            def save_iq(iq):
                original_iq(iq)
                values = np.empty(2 * len(iq), np.uint8)
                values[::2] = np.clip(np.rint(iq.real * 128 + 127.5), 0, 255)
                values[1::2] = np.clip(np.rint(iq.imag * 128 + 127.5), 0, 255)
                try:
                    raw_queue.put_nowait(values.tobytes())
                    recorder_health["queue_high_water"] = max(recorder_health["queue_high_water"], raw_queue.qsize())
                except queue.Full:
                    recorder_health["dropped_blocks"] += 1

            cap.preview.write = save_iq

    def health():
        rx = window.rx.engine
        row = dict(time=time.monotonic(), tx_phase=window.tx.engine.state.phase.value,
                   decoded=len(received), delivered=len(state["delivered"]),
                   displayed=len(state["displayed"]), video_queue=len(window.rx.preview._frames),
                   video_timer=window.rx.preview._timer.isActive(), message=rx.state.message)
        row["video_health"] = dict(window.rx.preview.health) if hasattr(window.rx.preview, "health") else {}
        row["rx_health"] = dict(rx.health) if hasattr(rx, "health") else {}
        row["tx_health"] = dict(getattr(window.tx.engine, "sdr_health", {}))
        row["raw_recorder"] = dict(recorder_health)
        row["source_dropped_frames"] = state["source_dropped_frames"]
        if rx.ring is not None:
            with rx.ring.lock:
                row.update(rx_samples=rx.ring.total_written,
                           rx_backlog_s=(rx.ring.total_written-rx._read_cursor)/rx.ring.fs)
        if rx._sdr is not None:
            row.update(iq_queue=rx._sdr._queue.qsize(), iq_error=rx._sdr.error,
                       iq_blocks=rx._sdr.preview.sequence)
            row["sdr_health"] = dict(getattr(rx._sdr, "health", {}))
        if rx._audio_playback is not None:
            row.update(audio_queue=rx._audio_playback._queue.qsize(),
                       audio_error=str(rx._audio_playback._error or ""))
            row["audio_health"] = dict(getattr(rx._audio_playback, "health", {}))
        if rx._av_playout is not None:
            row.update(av_queue=len(rx._av_playout), av_dropped=rx._av_playout.dropped)
        if rx._av_program is not None:
            row.update(av_pending=len(rx._av_program.pending))
        if sys.platform.startswith("linux"):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    row["rss_mb"] = int(line.split()[1]) / 1024
        telemetry.write(json.dumps(row) + "\n")

    def finish(reason=""):
        if status["finished"]:
            return
        status["finished"] = True
        timer.stop()
        cancel.set()
        if reason:
            error(reason)
        window.tx.cancel()
        if window.tx._thread is not None:
            window.tx._thread.join(timeout=8)
        health()
        state.update(tx_phase=window.tx.engine.state.phase.value, tx_timings=window.tx.engine.gop_timings,
                     decoded_gops=len(received), displayed_frames=len(state["displayed"]),
                     rf_start=status["rf_start"], frame_epoch=status["epoch"], finished=time.monotonic())
        window.grab().save(str(output / "gui.png"))
        window.close()
        telemetry.close()
        if raw_file is not None:
            raw_queue.put(raw_end, timeout=5)
            raw_thread.join(timeout=5)
            if raw_thread.is_alive():
                state["errors"].append(dict(message="Raw-IQ recorder did not drain"))
            raw_file.close()
        audio_trace.close()
        state["raw_recorder"] = dict(recorder_health)
        if recorder_health["dropped_blocks"]:
            state["errors"].append(dict(message="Raw-IQ recording overflowed; excluded from paired replay"))
        audio_recording.close()
        rendered.close()
        state["trace_recorders"] = {
            name: dict(recorder.health) for name, recorder in
            (("health", telemetry), ("audio_output", audio_trace),
             ("decoded_rgb", rendered), ("paired_audio", audio_recording))
        }
        for name, record in state["trace_recorders"].items():
            if record["error"]:
                state["errors"].append(dict(message=f"{name}: {record['error']}"))
        np.save(output / "sent-latents.npy", np.asarray(sent))
        np.save(output / "received-latents.npy", np.asarray(received))
        state["completed"] = state["tx_phase"] == "done" and not state["errors"]
        gc.callbacks.remove(garbage_collection)
        (output / "validation.json").write_text(json.dumps(state, indent=2) + "\n")
        app.exit(0 if state["completed"] else 1)

    def poll():
        now = time.monotonic()
        if now - started > count + 150:
            finish("GUI stress watchdog expired")
            return
        if not status["started"]:
            codec = window.station.codec
            if codec is None:
                if window._last_codec_error:
                    finish(window._last_codec_error)
                return
            state.update(backend=codec.backend, device=str(codec.device))
            encode, decode = codec.encode_gop, codec.decode_gop

            def record_encode(values):
                before = time.monotonic()
                latent = encode(values)
                sent.append(latent.copy())
                state["encode"].append(dict(start=before, end=time.monotonic()))
                return latent

            def record_decode(latent, weights=None):
                before = time.monotonic()
                index = len(received)
                elapsed = before - (status["rf_start"] or before)
                for number, fault in enumerate(config.get("decode_stalls", [])):
                    key = ("decode", number)
                    if key not in status["actions"] and elapsed >= fault["at_s"]:
                        status["actions"].add(key)
                        state["faults"].append(dict(kind="decode_stall", time=before, **fault))
                        cancel.wait(float(fault["seconds"]))
                result = decode(latent, weights)
                received.append(latent.copy())
                state["decode"].append(dict(start=before, end=time.monotonic()))
                rendered.write(np.asarray(result, np.uint8).tobytes())
                for i, frame in enumerate(result):
                    frame_ids[fingerprint(frame)] = index * mode.gop_frames + i
                return result

            codec.encode_gop, codec.decode_gop = record_encode, record_decode
            status["started"] = True
            window.tx.engine._camera_frames = frames
            window.tx.cam_radio.setChecked(True)
            if float(config.get("join_s", 0)) == 0:
                start_rx()
            window.tx.send()
            return
        if not status["rx_started"] and status["rf_start"] is not None:
            if now-status["rf_start"] >= float(config.get("join_s", 0)):
                start_rx()
        if status["rf_start"] is not None:
            elapsed = now - status["rf_start"]
            for index, phase in enumerate(config.get("gain_schedule", [])):
                key = ("gain", index)
                if key not in status["actions"] and elapsed >= phase["at_s"]:
                    status["actions"].add(key)
                    gain = float(phase["tx_gain"])
                    if not -89.75 <= gain <= 0:
                        finish("Invalid stress gain schedule")
                        return
                    status["tx_radio"].tx_hardwaregain_chan0 = gain
                    state["faults"].append(dict(kind="tx_gain", time=now, **phase))
            for index, fault in enumerate(config.get("gui_stalls", [])):
                key = ("gui", index)
                if key not in status["actions"] and elapsed >= fault["at_s"]:
                    status["actions"].add(key)
                    state["faults"].append(dict(kind="gui_stall", time=now, **fault))
                    cancel.wait(float(fault["seconds"]))
        health()
        if not window.tx.transmitting() and window.tx.engine.state.phase.value in {"done", "failed", "cancelled"}:
            if status["tx_done"] is None:
                status["tx_done"] = now
            if now-status["tx_done"] > 8 and not window.rx.preview._frames:
                finish()

    timer = QTimer(window)
    timer.timeout.connect(poll)
    timer.start(100)
    window._validation_timer = timer
