"""Deterministic virtual soundcards for offline production audio-path validation.

No device is opened. The only replaced component is sounddevice's native endpoint:
production TX resampling/writes and RX callbacks/resampling/ring buffer still run.
The HF channel is applied to the complete endpoint trace, preserving fading and
filter history across endpoint blocks. Virtual time tests ordering and bounded
buffers; it is deliberately not a claim about OS/driver real-time scheduling.
Run this harness separately from a live station (the backend patch is process-wide).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

from . import audio_io
from .hfchannel import ChannelProfile, emulate
from .ringbuffer import RingBuffer


@dataclass(frozen=True)
class AudioPathScenario:
    name: str
    tx_rate: int = 48000
    rx_rate: int = 48000
    snr_db: float | None = None
    fading: str = "none"
    clock_ppm: int = 0
    gain: float = 1.0
    dropout_start_s: float | None = None
    dropout_duration_s: float = 0.0
    capture_overflow_s: float | None = None
    consumer_stall_s: float = 0.0
    ring_seconds: float = 4.0
    cancel_after_s: float | None = None


SCENARIOS = (
    AudioPathScenario("soundcard_clean"),
    AudioPathScenario("mixed_native_rates", rx_rate=44100),
    AudioPathScenario("awgn12", snr_db=12),
    AudioPathScenario("awgn6", snr_db=6),
    AudioPathScenario("mpp12", snr_db=12, fading="mpp"),
    AudioPathScenario("mpp6", snr_db=6, fading="mpp"),
    AudioPathScenario("mpd12", snr_db=12, fading="mpd"),
    AudioPathScenario("clock_plus100", clock_ppm=100),
    AudioPathScenario("clock_minus100", clock_ppm=-100),
    AudioPathScenario("clipping", gain=4),
    AudioPathScenario("dropout100ms", dropout_start_s=1.5, dropout_duration_s=.1),
    AudioPathScenario("capture_overflow", capture_overflow_s=1.5),
    AudioPathScenario("slow_consumer", consumer_stall_s=1.2, ring_seconds=.25),
    AudioPathScenario("cancel", cancel_after_s=.35),
)


def virtual_audio_path(chunks, rate: int, scenario: AudioPathScenario, *, seed=0,
                       on_samples=None, on_gap=None, send_chunks=None, on_output_write=None):
    """Return captured samples and telemetry through the actual audio adapters.

    ``on_samples`` runs from the virtual consumer schedule after RingBuffer reads,
    never directly from the channel. Optional on_gap mirrors station reacquisition.
    Native capture overflow is a lost-sample event, while dropout is RF silence.
    Optional ``send_chunks(chunks, rate) -> bool`` may wrap the real player in
    TxEngine CAT/PTT orchestration. ``on_output_write(samples, state)`` observes
    each virtual write after counters advance, and can cancel or raise a fault.
    """
    if min(rate, scenario.tx_rate, scenario.rx_rate) <= 0:
        raise ValueError("audio rates must be positive")
    output, captured, errors, gaps = [], [], [], []
    state = dict(output_samples=0, max_write_frames=0, aborted=False,
                 output_closed=False, input_closed=False, input_stopped=False,
                 callbacks=0, ring_overruns=0, max_backlog_samples=0,
                 clipped_samples=0, dropped_rf_samples=0, capture_lost_samples=0)
    ring = RingBuffer(seconds=scenario.ring_seconds, fs=rate)
    callback = None

    class OutputStream:
        def __init__(self, **kwargs):
            assert kwargs["samplerate"] == scenario.tx_rate
            assert kwargs["channels"] == 1
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            state["output_closed"] = True
        def write(self, samples):
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            output.append(samples.copy())
            state["output_samples"] += len(samples)
            state["max_write_frames"] = max(state["max_write_frames"], len(samples))
            if on_output_write:
                on_output_write(samples, state)
            return False
        def abort(self):
            state["aborted"] = True

    class InputStream:
        def __init__(self, **kwargs):
            nonlocal callback
            assert kwargs["samplerate"] == scenario.rx_rate
            callback = kwargs["callback"]
        def start(self):
            pass
        def stop(self):
            state["input_stopped"] = True
        def close(self):
            state["input_closed"] = True

    def report_gap():
        gaps.append(state["callbacks"])
        if on_gap:
            on_gap()

    backend = SimpleNamespace(
        OutputStream=OutputStream, InputStream=InputStream,
        query_devices=lambda device, kind: {"default_samplerate":
            scenario.tx_rate if kind == "output" else scenario.rx_rate})
    stop = lambda: (scenario.cancel_after_s is not None and
                    state["output_samples"] / scenario.tx_rate >= scenario.cancel_after_s)
    with patch.object(audio_io, "_sd", return_value=backend):
        stream, _ = audio_io._open_input_stream_direct(
            None, ring, rate, on_error=errors.append, on_discontinuity=report_gap)
        try:
            complete = (send_chunks(chunks, rate) if send_chunks is not None else
                        audio_io._play_chunk_stream_direct(chunks, rate, should_stop=stop))
            tx = np.concatenate(output) if output else np.empty(0, dtype=np.float32)
            driven = tx * scenario.gain
            state["clipped_samples"] = int(np.count_nonzero(np.abs(driven) > 1))
            driven = np.clip(driven, -1, 1)
            if len(driven):
                channel = emulate(driven, ChannelProfile(scenario.name, scenario.name,
                                  scenario.snr_db, scenario.fading), seed=seed, fs=scenario.tx_rate)
                if scenario.dropout_start_s is not None:
                    start = int(scenario.dropout_start_s * scenario.tx_rate)
                    end = min(len(channel), start + int(scenario.dropout_duration_s * scenario.tx_rate))
                    channel[start:end] = 0
                    state["dropped_rf_samples"] = max(0, end - start)
                up = scenario.rx_rate * (1_000_000 + scenario.clock_ppm)
                down = scenario.tx_rate * 1_000_000
                factor = gcd(up, down)
                # Simulate independent endpoint clocks in the virtual cable;
                # each application's rate conversion still happens in audio_io.
                cable = (resample_poly(channel, up // factor, down // factor)
                         if up != down else channel)
                # Continue capture briefly after TX to drain the actual RX FIR.
                cable = np.concatenate([cable, np.zeros(int(.1 * scenario.rx_rate), np.float32)])
            else:
                cable = np.empty(0, np.float32)
            sizes = (127, 511, 64, 960, 257)
            cursor = offset = count = 0
            overflow_sent = False
            next_read_s = 0.0
            while offset < len(cable):
                size = min(sizes[count % len(sizes)], len(cable) - offset)
                status = None
                if (not overflow_sent and scenario.capture_overflow_s is not None and
                    offset / scenario.rx_rate >= scenario.capture_overflow_s):
                    overflow_sent = True
                    lost = min(int(.02 * scenario.rx_rate), len(cable) - offset)
                    offset += lost
                    state["capture_lost_samples"] += lost
                    size = min(size, len(cable) - offset)
                    status = "input overflow"
                callback(cable[offset:offset + size, None], size, None, status)
                state["callbacks"] += 1
                state["max_backlog_samples"] = max(state["max_backlog_samples"], ring.total_written - cursor)
                offset += size
                count += 1
                now = offset / scenario.rx_rate
                if now >= next_read_s and now >= scenario.consumer_stall_s:
                    values, cursor, overrun = ring.read_since(cursor)
                    if overrun:
                        state["ring_overruns"] += 1
                        report_gap()
                    captured.append(values)
                    if on_samples:
                        on_samples(values)
                    next_read_s = now + .02
            values, cursor, overrun = ring.read_since(cursor)
            if overrun:
                state["ring_overruns"] += 1
                report_gap()
            captured.append(values)
            if on_samples:
                on_samples(values)
        finally:
            stream.stop()
            stream.close()
    result = np.concatenate(captured) if captured else np.empty(0, np.float32)
    telemetry = dict(state, scenario=asdict(scenario), completed=complete,
                     captured_samples=len(result), discontinuities=len(gaps), errors=errors,
                     output_seconds=state["output_samples"] / scenario.tx_rate,
                     max_write_ms=1000 * state["max_write_frames"] / scenario.tx_rate,
                     ring_capacity_samples=ring.n)
    return result, telemetry
