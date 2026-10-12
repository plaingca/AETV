#!/usr/bin/env python3
"""Exercise real PortAudio/ALSA file-PCM streams without opening any hardware.

Linux only. Requires sounddevice, libportaudio and ALSA. The isolated ALSA
configuration is installed in the environment before sounddevice is imported.
The file PCM has no sample clock: this is NOT a real-time latency/xrun test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aetv import audio_io
from aetv.analog_validation import unit_rms_latents
from aetv.config import AETV_MODES, RELEASE_MODES
from aetv.hfchannel import emulate
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.ringbuffer import RingBuffer

DEVICE = 'aetv_file_pcm'


def configure_fixture(root):
    """No defaults, host config, devices, or persistent services are modified."""
    if sys.platform != 'linux':
        raise RuntimeError('This fixture requires Linux ALSA; it is not a Windows test')
    base = Path('/usr/share/alsa/alsa.conf')
    if not base.is_file():
        raise RuntimeError(f'ALSA base configuration is missing: {base}')
    if 'sounddevice' in sys.modules:
        raise RuntimeError('Run this fixture in a fresh process before sounddevice initializes')
    root = Path(root).resolve()
    if any(c in str(root) for c in ('"', '\n', '\\')):
        raise ValueError('fixture path contains unsupported ALSA syntax')
    config = root / 'alsa.conf'
    config.write_text(f'''<{base}>
pcm.{DEVICE} {{
 type file
 slave.pcm "null"
 file "{root / 'output.raw'}"
 infile "{root / 'input.raw'}"
 format "raw"
 hint {{ show on description "AETV isolated file PCM (no hardware)" }}
}}
''')
    os.environ['ALSA_CONFIG_PATH'] = str(config)
    (root / 'input.raw').write_bytes(b'\0' * 4096)


def capture(root, samples, native_rate, rate, *, small_ring=False):
    # Two seconds of trailing silence drain the RX resampler and decoder.
    np.r_[samples, np.zeros(native_rate * 2)].astype('<f4').tofile(root / 'input.raw')
    target = int(np.ceil(len(samples) * rate / native_rate)) + rate
    ready = threading.Event()
    class Collector(RingBuffer):
        callbacks = 0
        def write(self, values):
            super().write(values)
            self.callbacks += 1
            if self.total_written >= target:
                ready.set()
    ring = Collector(.025 if small_ring else target / rate + 20, rate)
    errors, gaps = [], []
    stream, actual_rate = audio_io.open_input_stream(
        DEVICE, ring, rate, on_error=errors.append,
        on_discontinuity=lambda: gaps.append(ring.total_written))
    try:
        if not ready.wait(20):
            raise TimeoutError('native capture did not reach its sample target')
    finally:
        stream.stop()
        stream.close()
    data, cursor, overrun = ring.read_since(0)
    return data[:target], dict(native_rate=actual_rate, callbacks=ring.callbacks,
        total_written=cursor, ring_overrun=overrun, errors=errors,
        discontinuities=len(gaps), closed=stream.closed)


def modem_trial(root, sd, mode_name, profile, seed, gops):
    mode = AETV_MODES[mode_name]
    rate = mode.geometry.fs
    originals = [unit_rms_latents(mode_name, 13000 + seed * 100 + n) for n in range(gops)]
    chunks = [chunk * (.5 / max(1e-12, float(np.max(np.abs(chunk)))))
              for chunk in modulate_continuous_chunks(originals, mode_name, 'N0CALL')]
    native = int(sd.query_devices(DEVICE, 'output')['default_samplerate'])
    assert audio_io.play_chunk_stream(chunks, rate, DEVICE)
    transmitted = np.fromfile(root / 'output.raw', dtype='<f4')
    # Compare the actual ALSA file bytes against the intended native waveform.
    resampler = audio_io.StreamResampler(*audio_io.resample_ratio(rate, native))
    expected = np.concatenate([*(resampler(chunk) for chunk in chunks), resampler.flush()])
    np.testing.assert_allclose(transmitted, expected, atol=1e-6)
    impaired = emulate(transmitted, profile, seed=seed, fs=native)
    received, health = capture(root, impaired, native, rate)
    receiver = StreamingDemodulator(mode.band, mode_name=mode_name, continuous=True,
        timing_tracking=True, boundary_tracking=True)
    results = []
    for start in range(0, len(received), rate // 10):
        results.extend(receiver.feed(received[start:start + rate // 10]))
    matches = []
    for result in results:
        for latent in result.gops_latents:
            scores = [float(np.corrcoef(source, latent)[0, 1]) for source in originals]
            index = int(np.argmax(scores))
            nmse = float(np.mean((latent.astype(float) - originals[index]) ** 2) /
                         np.mean(originals[index].astype(float) ** 2))
            matches.append(dict(source_gop=index, correlation=scores[index], nmse=nmse))
    valid = [item for item in matches if item['correlation'] > .25]
    unique = len({item['source_gop'] for item in valid})
    baseline = (unique == gops and len(matches) == gops and
                min((m['correlation'] for m in matches), default=0) > .9 and
                max((m['nmse'] for m in matches), default=1) < .2 and
                not health['errors'] and not health['ring_overrun']) if profile == 'clean' else None
    return dict(mode=mode_name, profile=profile, seed=seed, modem_rate=rate,
        sent_gops=gops, received_gops=len(matches), matched_unique_gops=unique,
        tx_native_samples=len(transmitted), tx_max_error=float(np.max(abs(transmitted - expected))),
        capture=health, measurements=matches, baseline_pass=baseline)


def lifecycle_tests(root, sd):
    rows = []
    for rate in (8000, 12000, 24000, 44100, 48000):
        sd.check_input_settings(DEVICE, channels=1, samplerate=rate, dtype='float32')
        sd.check_output_settings(DEVICE, channels=1, samplerate=rate, dtype='float32')
        with sd.OutputStream(device=DEVICE, channels=1, samplerate=rate, dtype='float32') as s:
            underflow = s.write(np.full(rate // 10, .125, np.float32))
        actual = np.fromfile(root / 'output.raw', '<f4')
        np.testing.assert_array_equal(actual, np.full(rate // 10, .125, np.float32))
        np.full(rate, .125, np.float32).tofile(root / 'input.raw')
        with sd.InputStream(device=DEVICE, channels=1, samplerate=rate, dtype='float32') as s:
            recorded, overflow = s.read(rate // 10)
        np.testing.assert_array_equal(recorded[:, 0], np.full(rate // 10, .125, np.float32))
        rows.append(dict(rate=rate, native_open_read_write_close_pass=True,
                         output_underflow=underflow, input_overflow=overflow))
    callback_complete = audio_io.play_cancellable(np.ones(8000, np.float32) * .1, 8000, DEVICE)
    progress = []
    callback_cancel = audio_io.play_cancellable(np.ones(80000, np.float32) * .1, 8000, DEVICE,
        should_stop=lambda: bool(progress), on_progress=progress.append)
    checks = []
    def stop():
        checks.append(1)
        return len(checks) >= 8
    blocking_cancel = audio_io.play_chunk_stream([np.ones(80000, np.float32) * .1], 8000,
        DEVICE, should_stop=stop)
    cancel_samples = (root / 'output.raw').stat().st_size // 4
    native = int(sd.query_devices(DEVICE, 'input')['default_samplerate'])
    _, ring_health = capture(root, np.zeros(native // 4), native, 8000, small_ring=True)
    reopened = []
    for _ in range(3):
        data, health = capture(root, np.ones(native // 10) * .125, native, 8000)
        reopened.append(bool(health['closed'] and health['callbacks'] > 0 and np.max(data) > .1))
    failed = []
    for kind in ('input', 'output'):
        try:
            if kind == 'input':
                audio_io.open_input_stream(999999, RingBuffer(1, 8000), 8000)
            else:
                audio_io.play_chunk_stream([np.zeros(800)], 8000, 999999)
        except (sd.PortAudioError, ValueError) as error:
            failed.append(dict(kind=kind, exception=type(error).__name__, message=str(error)))
        else:
            raise AssertionError('invalid endpoint unexpectedly opened')
    assert callback_complete and not callback_cancel and not blocking_cancel
    assert ring_health['ring_overrun'] and all(reopened) and len(failed) == 2
    return dict(rate_probes=rows, callback_completion=callback_complete,
        callback_cancel_return=callback_cancel, blocking_cancel_return=blocking_cancel,
        blocking_cancel_native_samples=cancel_samples, small_ring_capture=ring_health,
        reopen_passes=reopened, invalid_endpoint_failures=failed,
        actual_driver_xrun='Not induced: ALSA null/file PCM has no real-time device clock.',
        disconnect='Reopen only; physical unplug and backend disconnect were not tested.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/validation/portaudio-linux-20261012.json'))
    parser.add_argument('--modes', nargs='+', choices=RELEASE_MODES, default=list(RELEASE_MODES))
    parser.add_argument('--profiles', nargs='+', choices=('clean', 'awgn6', 'mpp12'),
                        default=['clean', 'awgn6', 'mpp12'])
    parser.add_argument('--gops', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.gops <= 100:
        parser.error('gops must be 1..100')
    with tempfile.TemporaryDirectory(prefix='aetv-native-pcm-') as directory:
        root = Path(directory)
        configure_fixture(root)
        sd = audio_io._sd()
        inventory = {kind: [asdict(item) for item in audio_io.list_audio_devices(kind)]
                     for kind in ('input', 'output')}
        # Explicitly selected file PCM only. No default/physical endpoint is opened.
        assert any(item['name'] == DEVICE for item in inventory['input'])
        report = dict(platform=platform.platform(), python=platform.python_version(),
            sounddevice=sd.__version__, portaudio=sd.get_portaudio_version(),
            scope='Real sounddevice -> PortAudio -> ALSA file/null PCM, production TX writes/RX callbacks, streaming resamplers and ring, HF emulator, modem. No endpoint monkeypatch.',
            limits='File-backed accelerated replay, not simultaneous real-time loopback. No physical soundcard, microphone, loudspeaker, RF, Station engine, CAT/PTT, Windows WASAPI, callback deadlines or native xrun/unplug certification.',
            inventory=inventory, hostapis=list(sd.query_hostapis()), lifecycle=lifecycle_tests(root, sd),
            trials=[])
        for mode in args.modes:
            for profile in args.profiles:
                row = modem_trial(root, sd, mode, profile, 0, args.gops)
                report['trials'].append(row)
                print(f"{mode} {profile}: {row['matched_unique_gops']}/{args.gops}; baseline={row['baseline_pass']}", flush=True)
        clean = [r['baseline_pass'] for r in report['trials'] if r['profile'] == 'clean']
        report['all_clean_pass'] = all(clean) if clean else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        if report['all_clean_pass'] is False:
            raise SystemExit('Clean baseline failed; see JSON report')


if __name__ == '__main__':
    main()
