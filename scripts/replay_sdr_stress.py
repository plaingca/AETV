#!/usr/bin/env python3
"""Replay unsigned-byte RTL IQ through production tuning, modem and A/V pairing.

No source frames, transmitted latents, start time, or transmitter frequency
measurement enter the receiver. Optional manual correction is an explicit
control arm. See docs/all-modes-sdr-stress.md for the capture configuration.
"""

import argparse
import json
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aetv.analog_av import (
    AC16CompositeSeparator, ProgramAudio, StreamingCompositeSeparator,
    waveform_sample_rate,
)
from aetv.audio_io import StreamResampler, resample_ratio
from aetv.config import AETV_MODES
from aetv.modem import StreamingDemodulator
from aetv.sdr import SDRCapture
from aetv.settings import StationSettings


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iq", type=Path)
    parser.add_argument("--mode", choices=["V8", "V7", "AC16", "V8_AV", "AC16_AV"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix (.json/.npz)")
    parser.add_argument("--join", type=float, default=0, help="Seconds discarded before starting a fresh receiver")
    parser.add_argument("--limit", type=float, default=0)
    parser.add_argument("--correction", type=float, help="Manual Hz correction; default is received-only automatic tuning")
    args = parser.parse_args()
    if args.join < 0 or args.limit < 0:
        parser.error("Join and duration must be nonnegative")
    mode = AETV_MODES[args.mode.removesuffix("_AV")]
    av = args.mode.endswith("_AV")
    settings = StationSettings(mode=mode.name, waveform_mode="analog_av" if av else "video",
                               rx_source="rtlsdr", sdr_auto_correct=args.correction is None,
                               sdr_rx_correction_hz=args.correction or 0)
    raw = np.memmap(args.iq, np.uint8, mode="r")[round(args.join * 1920000):]
    if len(raw) % 2:
        raise ValueError("RTL IQ must contain complete I/Q byte pairs")
    if args.limit:
        raw = raw[:round(args.limit * 1920000)]
    rows, events, latents, weights, paired, pair_indices = [], [], [], [], [], []
    input_time = [0.]

    def event(value):
        events.append(dict(input_s=input_time[0], **value))

    demod = StreamingDemodulator(mode.band, continuous=True, mode_name=mode.name,
                                 boundary_tracking=True, verify_gap_phase=True, on_debug=event)
    separator = (AC16CompositeSeparator() if mode.name == "AC16" else StreamingCompositeSeparator()) if av else None
    rate = waveform_sample_rate(settings)
    video_resampler = StreamResampler(*resample_ratio(rate, mode.geometry.fs))
    voice_resampler = StreamResampler(*resample_ratio(rate, 8000))
    program = ProgramAudio(mode.name) if av else None

    def receive(audio):
        if separator is not None:
            voice, audio = separator.process(audio)
            program.voice.write(voice_resampler(voice))
            audio = video_resampler(audio)
        for result in demod.feed(audio):
            index = len(rows)
            rows.append(dict(input_s=input_time[0], sample=result.stream_start_sample,
                             frame_counter=result.stream_frame_counter,
                             freq_offset=result.freq_offset, snr_db=result.snr_db))
            latents.append(result.gops_latents[0])
            weights.append(result.gops_weights[0])
            if program is not None:
                program.add(result, index)
        if program is not None:
            for index, voice in program.ready():
                pair_indices.append(index)
                paired.append(voice)

    capture = SDRCapture(settings, mode, SimpleNamespace(write=receive),
                         on_error=lambda message: event(dict(event="error", message=message)),
                         on_status=lambda message: event(dict(event="sdr_status", message=message)))

    class Input:
        position = 0

        def get(self, timeout):
            if self.position >= len(raw):
                capture._stop.set()
                raise queue.Empty
            end = min(self.position + 192000, len(raw))
            values = raw[self.position:end].astype(np.float32)
            self.position = end
            input_time[0] = end / 1920000
            return ((values[::2] - 127.5) + 1j * (values[1::2] - 127.5)) / 128

    capture._queue = Input()
    started = time.perf_counter()
    capture._convert()
    result = dict(mode=args.mode, iq=str(args.iq), join=args.join,
                  input_s=len(raw) / 1920000, processing_s=time.perf_counter() - started,
                  manual_correction_hz=args.correction, rows=rows, events=events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(finite(result), indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), latents=latents, weights=weights,
             audio=paired, audio_row_indices=pair_indices)
    print(json.dumps(dict(mode=args.mode, decoded=len(rows), paired=len(paired),
                          processing_s=result["processing_s"])), flush=True)


if __name__ == "__main__":
    main()
