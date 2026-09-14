"""Opt-in, bounded hardware validation of the actual packaged GUI.

Invoked only with --ota-validation CONFIG.json and transmit=true in that file.
The camera boundary is fed paced RGB fixtures; the GUI, codec, radio transport,
streaming receiver and displayed-frame callback are the ordinary application.
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer

from aetv.codec import sha256_file
from aetv.settings import StationSettings


def load_validation(path):
    config = json.loads(Path(path).read_text())
    if config.get("transmit") is not True:
        raise ValueError("OTA validation requires explicit transmit=true")
    gops = int(config.get("gops", 10))
    if not 1 <= gops <= 60:
        raise ValueError("OTA validation is limited to 1–60 GOPs")
    settings = StationSettings(
        mode="AC16",
        tx_backend=config.get("transmitter", "pluto"),
        rx_source=config.get("receiver", "rtlsdr"),
        gops=gops,
        torch_device=config.get("device", "cpu"),
        pluto_uri=config.get("pluto_uri", "ip:192.168.2.1"),
        rtl_serial=config.get("rtl_serial", "1001"),
        hackrf_serial=config.get("hackrf_serial", ""),
        hackrf_tx_gain=int(config.get("hackrf_tx_gain", 0)),
        hackrf_rx_lna_gain=int(config.get("hackrf_rx_lna_gain", 16)),
        hackrf_rx_vga_gain=int(config.get("hackrf_rx_vga_gain", 16)),
        pluto_tx_gain=float(config.get("tx_gain", -30)),
        pluto_rx_gain=float(config.get("rx_gain", 30)),
        rtl_rx_gain=float(config.get("rx_gain", 37.2)),
        sdr_frequency_mhz=float(config.get("frequency_mhz", 439)),
        sdr_auto_correct=config.get("auto_correct", True),
        sdr_rx_correction_hz=float(config.get("rx_correction_hz", 0)),
        receive_dir=str(Path(config["output"]) / "received"),
        buffer_seconds=8,
        autosave=True,
        debug_capture=True,
    )
    problems = settings.validate()
    if settings.tx_backend not in {"pluto", "hackrf"}:
        problems.append("OTA validation requires a Pluto or HackRF transmitter")
    if settings.tx_backend == "hackrf" and settings.rx_source == "hackrf":
        problems.append("HackRF is half duplex; OTA validation needs a separate Pluto or RTL receiver")
    if problems:
        raise ValueError("; ".join(problems))
    return config, settings


def install_validation(window, app, config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    source_path = Path(config["source"])
    source = np.load(source_path, mmap_mode="r")[: window.settings.gops * 10]
    if (
        source.shape != (window.settings.gops * 10, 144, 256, 3)
        or source.dtype != np.uint8
    ):
        raise ValueError("Expected AC16 RGB source frames")
    started = time.monotonic()
    state = dict(
        source_sha256=sha256_file(source_path),
        packaged=bool(getattr(sys, "frozen", False)),
        radio={
            key: getattr(window.settings, key)
            for key in (
                "rx_source",
                "tx_backend",
                "sdr_frequency_mhz",
                "pluto_uri",
                "rtl_serial",
                "hackrf_serial",
                "hackrf_tx_gain",
                "hackrf_rx_lna_gain",
                "hackrf_rx_vga_gain",
                "pluto_tx_gain",
                "pluto_rx_gain",
                "rtl_rx_gain",
                "sdr_auto_correct",
                "sdr_rx_correction_hz",
            )
        },
        backend=None,
        input_frames=[],
        encode=[],
        decode=[],
        displayed=[],
        errors=[],
    )
    active = {
        "started": False,
        "finished": False,
        "tx_done": None,
        "screenshot_gop": None,
    }
    cancel = threading.Event()
    sent, received = [], []
    original_show = window.rx.preview._show_frame

    def shown(frame):
        original_show(frame)
        state["displayed"].append(time.monotonic())

    window.rx.preview._show_frame = shown

    def frames(_mode, **_kwargs):
        pending = queue.Queue(20)
        epoch = time.monotonic()

        def feed():
            for index, frame in enumerate(source):
                deadline = epoch + index / 10
                if cancel.wait(max(0, deadline - time.monotonic())):
                    return
                state["input_frames"].append(
                    dict(index=index, scheduled=deadline, available=time.monotonic())
                )
                try:
                    pending.put_nowait(np.array(frame))
                except queue.Full:
                    state["errors"].append(
                        "Paced RGB input overflowed its two-second queue"
                    )
                    return

        producer = threading.Thread(
            target=feed, daemon=True, name="validation-rgb-source"
        )
        producer.start()
        try:
            for _ in source:
                yield pending.get(timeout=3)
        finally:
            cancel.set()
            producer.join(timeout=3)

    def finish(reason=""):
        if active["finished"]:
            return
        active["finished"] = True
        timer.stop()
        cancel.set()
        if reason:
            state["errors"].append(reason)
        window.tx.cancel()
        if window.tx._thread is not None:
            window.tx._thread.join(timeout=8)
        video = window.rx.engine.last_video
        state.update(
            gops=window.rx.engine.state.gops,
            tx_phase=window.tx.engine.state.phase.value,
            tx_message=window.tx.engine.state.message,
            requested_gops=window.settings.gops,
            displayed_frames=len(state["displayed"]),
            tx_timings=window.tx.engine.gop_timings,
        )
        if video is not None:
            np.save(output / "received.npy", video)
        np.save(output / "sent-latents.npy", np.asarray(sent))
        np.save(output / "received-latents.npy", np.asarray(received))
        window.grab().save(str(output / "gui.png"))
        window.close()
        if len(sent) == len(received) == window.settings.gops:
            tx, rx = np.asarray(sent), np.asarray(received)
            similarity = (
                rx / np.maximum(np.linalg.norm(rx, axis=1, keepdims=True), 1e-12)
            ) @ (tx / np.maximum(np.linalg.norm(tx, axis=1, keepdims=True), 1e-12)).T
            state["latent_cosines"] = np.diag(similarity).tolist()
            state["correct_chronology"] = bool(
                np.array_equal(similarity.argmax(1), np.arange(len(tx)))
            )
        state["passed"] = bool(
            not state["errors"]
            and state["gops"] == window.settings.gops
            and state["tx_phase"] == "done"
            and state["displayed_frames"] == len(source)
            and state.get("correct_chronology", False)
            and min(state.get("latent_cosines", [0])) > 0.8
        )
        (output / "validation.json").write_text(json.dumps(state, indent=2) + "\n")
        app.exit(0 if state["passed"] else 1)

    def poll():
        if time.monotonic() - started > window.settings.gops + 120:
            finish("GUI hardware validation watchdog expired")
            return
        if not active["started"]:
            codec = window.station.codec
            if codec is None:
                if window._last_codec_error:
                    finish(window._last_codec_error)
                return
            state["backend"], state["device"] = codec.backend, str(codec.device)
            state["providers"] = (
                codec._encoder_session.get_providers()
                if codec.backend == "onnxruntime"
                else []
            )
            encode, decode = codec.encode_gop, codec.decode_gop

            def record_encode(values):
                before = time.monotonic()
                z = encode(values)
                state["encode"].append(dict(start=before, end=time.monotonic()))
                sent.append(z.copy())
                return z

            def record_decode(z, weights=None):
                before = time.monotonic()
                result = decode(z, weights)
                state["decode"].append(dict(start=before, end=time.monotonic()))
                received.append(z.copy())
                return result

            codec.encode_gop, codec.decode_gop = record_encode, record_decode
            active["started"] = True
            window.tx.engine._camera_frames = frames
            window.tx.cam_radio.setChecked(True)
            if not window.rx.start():
                finish(window.rx.status.text())
                return
            window.tx.send()
            return
        if not window.tx.transmitting() and window.tx.engine.state.phase.value in {
            "done",
            "failed",
            "cancelled",
        }:
            if active["tx_done"] is None:
                active["tx_done"] = time.monotonic()
            if (
                time.monotonic() - active["tx_done"] > 4
                and not window.rx.preview._frames
            ):
                finish()
        gop = window.rx.engine.state.gops
        if (
            gop
            and gop % 5 == 0
            and active["screenshot_gop"] != gop
            and window.tx.transmitting()
        ):
            active["screenshot_gop"] = gop
            window.grab().save(str(output / "gui-live.png"))

    timer = QTimer(window)
    timer.timeout.connect(poll)
    timer.start(100)
    window._validation_timer = timer
