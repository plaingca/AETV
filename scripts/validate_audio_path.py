#!/usr/bin/env python3
"""Hardware-free audio/HF bench: real adapters, callbacks, resamplers and modem."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aetv.analog_validation import unit_rms_latents
from aetv.audio_validation import AudioPathScenario, SCENARIOS, virtual_audio_path
from aetv.config import AETV_MODES, RELEASE_MODES
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks


def run_trial(mode_name, scenario, seed=0, gops=4, originals=None, restart_after_gops=None):
    mode = AETV_MODES[mode_name]
    originals = originals if originals is not None else [
        unit_rms_latents(mode_name, 13000 + seed * 100 + n) for n in range(gops)]
    chunks = []
    groups = ([originals] if restart_after_gops is None else
              [originals[:restart_after_gops], originals[restart_after_gops:]])
    for group_index, group in enumerate(groups):
        if group_index:
            chunks.append(np.zeros(mode.geometry.fs // 2, np.float32))
        for chunk in modulate_continuous_chunks(group, mode_name, "N0CALL"):
            maximum = float(np.max(np.abs(chunk))) if chunk.size else 0
            chunks.append(chunk * (.65 / maximum) if maximum else chunk)
    results, timing = [], []
    def make_receiver():
        return StreamingDemodulator(mode.band, mode_name=mode_name, continuous=True,
                                    timing_tracking=True, boundary_tracking=True)
    receiver = make_receiver()
    def receive(values):
        started = time.perf_counter()
        results.extend(receiver.feed(values))
        timing.append(time.perf_counter() - started)
    def gap():
        nonlocal receiver
        receiver = make_receiver()
    started = time.perf_counter()
    _capture, health = virtual_audio_path(chunks, mode.geometry.fs, scenario, seed=seed,
                                         on_samples=receive, on_gap=gap)
    matches, recovered = [], []
    for result in results:
        for latent, weight in zip(result.gops_latents, result.gops_weights):
            if not np.isfinite(latent).all() or not np.isfinite(weight).all():
                raise AssertionError("nonfinite modem output")
            correlations = [float(np.corrcoef(source, latent)[0, 1]) for source in originals]
            index = int(np.argmax(correlations))
            nmse = float(np.mean((latent.astype(float) - originals[index]) ** 2) /
                         max(1e-12, float(np.mean(originals[index] ** 2))))
            matches.append(dict(source_gop=index, correlation=correlations[index], nmse=nmse,
                                snr_db=float(result.snr_db), confidence=float(np.mean(weight))))
            recovered.append((index, latent, weight))
    valid = [m for m in matches if m['correlation'] > .25]
    row = dict(mode=mode_name, fs=mode.geometry.fs, seed=seed, sent_gops=len(originals),
               received_gops=len(matches), matched_unique_gops=len({m['source_gop'] for m in valid}),
               unmatched_outputs=len(matches) - len(valid), measurements=matches,
               mean_correlation=float(np.mean([m['correlation'] for m in valid])) if valid else None,
               mean_nmse=float(np.mean([m['nmse'] for m in valid])) if valid else None,
               audio=health, elapsed_compute_s=time.perf_counter() - started,
               max_decode_feed_ms=1000 * max(timing, default=0))
    if scenario.name in ('soundcard_clean', 'mixed_native_rates', 'native_clean'):
        row['baseline_pass'] = (row['matched_unique_gops'] == len(originals) and
                                row['received_gops'] == len(originals) and
                                row['mean_correlation'] > .9 and row['mean_nmse'] < .2 and
                                health['ring_overruns'] == 0 and not health['errors'])
    elif scenario.cancel_after_s is not None:
        row['baseline_pass'] = (not health['completed'] and health['aborted'] and
            scenario.cancel_after_s <= health['output_seconds'] <= scenario.cancel_after_s + .021)
    else:
        row['baseline_pass'] = None  # Impairment curves are measurements, not guaranteed decodes.
    return row, recovered


def recovery_trials():
    """Distinguish short-stream reset loss from long-stream/fresh-preamble recovery."""
    rows = []
    for scenario in [AudioPathScenario('capture_overflow', capture_overflow_s=1.5),
                     AudioPathScenario('slow_consumer', consumer_stall_s=1.2, ring_seconds=.25)]:
        row, _ = run_trial('V8', scenario, gops=16)
        row['recovery_pass'] = any(m['source_gop'] >= 12 and m['correlation'] > .9
                                   for m in row['measurements'])
        rows.append(row)
    restart, _ = run_trial('V8', AudioPathScenario('fresh_transmission', capture_overflow_s=1.5),
                           gops=4, restart_after_gops=2)
    restart['recovery_pass'] = {2, 3}.issubset(
        {m['source_gop'] for m in restart['measurements'] if m['correlation'] > .9})
    rows.append(restart)
    return rows


def make_chart(rows, destination):
    modes = list(dict.fromkeys(row['mode'] for row in rows))
    scenarios = list(dict.fromkeys(row['audio']['scenario']['name'] for row in rows))
    width, height = 850, 85 + len(scenarios) * 32
    canvas = Image.new('RGB', (width, height), '#101923')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), 'Virtual soundcard + HF channel: unique matched GOPs / sent', fill='white')
    draw.text((16, 32), 'Aggregate matches across seeds; cancellation is an expected abort; no physical RF', fill='#bdc9d4')
    for j, mode in enumerate(modes):
        draw.text((215 + j * 155, 58), mode, fill='white')
    for i, scenario in enumerate(scenarios):
        y = 83 + i * 32
        draw.text((16, y + 4), 'cancel (expected abort)' if scenario == 'cancel' else scenario, fill='white')
        for j, mode in enumerate(modes):
            group = [r for r in rows if r['mode'] == mode and r['audio']['scenario']['name'] == scenario]
            if not group:
                continue
            numerator = sum(r['matched_unique_gops'] for r in group)
            denominator = sum(r['sent_gops'] for r in group)
            x = 210 + j * 155
            draw.rectangle((x, y, x + 140, y + 24), fill='#344453')
            draw.rectangle((x, y, x + max(1, int(140 * numerator / denominator)), y + 24), fill='#277b81')
            draw.text((x + 5, y + 5), f'{numerator}/{denominator}', fill='white')
    canvas.save(destination)


def learned_trial(manifest, destination):
    from aetv.codec import AETVCodec, RELEASE_RUNTIME_FILES
    hashes = {}
    for name, expected in RELEASE_RUNTIME_FILES['V8'].items():
        file = Path(manifest).parent / name
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        hashes[name] = dict(sha256=actual, matches_release=actual == expected['sha256'])
        if actual != expected['sha256']:
            raise ValueError(f"V8 release checksum mismatch: {file}")
    if Path(manifest).name not in RELEASE_RUNTIME_FILES['V8'] or not str(manifest).endswith('.runtime.json'):
        raise ValueError('Use the pinned V8 runtime manifest installed by Model Manager')
    codec = AETVCodec(checkpoint=manifest, mode='V8', device='cpu', allow_download=False)
    scenes = ('interview', 'presenter', 'canopy')
    sources = []
    source_paths = []
    root = Path(__file__).resolve().parent.parent
    for scene in scenes:
        source_path = root / 'docs' / 'demos' / f'standard-{scene}-motion.gif'
        source_paths.append(str(source_path.relative_to(root)))
        with Image.open(source_path) as image:
            frames = []
            for index in range(6):
                image.seek(index)
                frames.append(np.asarray(image.convert('RGB').crop((0, 0, 192, 108))).copy())
            sources.append(np.stack(frames))
    encode_start = time.perf_counter()
    latents = [codec.encode_gop(source) for source in sources]
    encode_seconds = time.perf_counter() - encode_start
    profiles = [AudioPathScenario('soundcard_clean'), AudioPathScenario('awgn6', snr_db=6),
                AudioPathScenario('mpp12', snr_db=12, fading='mpp')]
    decoded, rows = {}, []
    for profile in profiles:
        row, recovered = run_trial('V8', profile, originals=latents)
        decoded[profile.name] = {}
        metrics = []
        for index, latent, weights in recovered:
            start = time.perf_counter()
            frames = codec.decode_gop(latent, weights)
            decode_ms = 1000 * (time.perf_counter() - start)
            decoded[profile.name][index] = frames
            mse = float(np.mean((frames.astype(float) - sources[index].astype(float)) ** 2))
            metrics.append(dict(source_gop=index, psnr_db=10 * np.log10(255**2 / max(mse, 1e-12)),
                                decode_ms=decode_ms, shape=list(frames.shape)))
        row['video'] = metrics
        rows.append(row)
    panels = [('Source (GIF crop)', None)] + [(p.name, p.name) for p in profiles]
    canvas = Image.new('RGB', (len(panels) * 200, 3 * 140 + 38), '#101923')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), 'V8 ONNX -> production audio callbacks -> HF -> modem -> ONNX (seed 0)', fill='white')
    for col, (title, key) in enumerate(panels):
        draw.text((col * 200 + 4, 27), title, fill='white')
        for row, source in enumerate(sources):
            frames = source if key is None else decoded[key].get(row)
            y = 48 + row * 136
            if frames is not None:
                canvas.paste(Image.fromarray(frames[3]), (col * 200 + 4, y))
            else:
                draw.text((col * 200 + 8, y + 44), 'No matched GOP', fill='white')
            draw.text((col * 200 + 4, y + 112), scenes[row], fill='#c2d1e0')
    canvas.save(destination / 'audio-learned-v8.png')
    return dict(backend=codec.backend, mode='V8', source=source_paths,
                source_note='Six-frame source cells cropped from existing lossy GIF demo grids; includes source label overlay. Three distinct scenes, not independent natural video benchmarks.',
                encode_seconds=encode_seconds, runtime_files=hashes, trials=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/validation/audio-path'))
    parser.add_argument('--modes', nargs='+', choices=RELEASE_MODES, default=list(RELEASE_MODES))
    parser.add_argument('--scenarios', nargs='+', choices=[s.name for s in SCENARIOS] + ['native_clean'])
    parser.add_argument('--gops', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--onnx-manifest', type=Path, help='Optional existing V8 runtime manifest; no download or Torch installation')
    args = parser.parse_args()
    if not 1 <= args.gops <= 100 or not 1 <= args.seeds <= 20:
        parser.error('gops must be 1..100 and seeds 1..20')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes:
        fs = AETV_MODES[mode].geometry.fs
        scenarios = (AudioPathScenario('native_clean', tx_rate=fs, rx_rate=fs),) + SCENARIOS
        for scenario in scenarios:
            if args.scenarios and scenario.name not in args.scenarios and scenario.name != 'native_clean':
                continue
            for seed in range(args.seeds):
                row, _ = run_trial(mode, scenario, seed, args.gops)
                rows.append(row)
                print(f"{mode} {scenario.name} seed={seed}: {row['matched_unique_gops']}/{args.gops}, NMSE={row['mean_nmse']}", flush=True)
    report = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        scope='Accelerated virtual native endpoints; actual audio_io TX/RX adapters, streaming resamplers, callbacks, ring buffer and modem. No Station engine, real PortAudio/WASAPI driver, physical soundcard, CAT/PTT or RF in this harness.',
        channel='Whole native TX trace through Watterson/AWGN, then independent RX clock conversion. Noise referenced to 2500 Hz. RF dropout is zeroed interval; capture overflow removes samples and reports discontinuity.',
        timing='Deterministic virtual consumer schedule; callback deadlines and driver latency are not measured. Compute timing is machine-dependent.',
        matching='Maximum latent correlation > 0.25 is a measurement oracle only; never given to the receiver.',
        learned_codec=None, trials=rows, recovery_checks=recovery_trials() if 'V8' in args.modes else [])
    if args.onnx_manifest:
        report['learned_codec'] = learned_trial(args.onnx_manifest, args.output_dir)
    baseline = [r['baseline_pass'] for r in rows if r['baseline_pass'] is not None]
    report['baseline_pass'] = bool(baseline) and all(baseline) and all(r['recovery_pass'] for r in report['recovery_checks'])
    if report['learned_codec']:
        clean = report['learned_codec']['trials'][0]
        report['baseline_pass'] &= clean['matched_unique_gops'] == clean['sent_gops']
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    make_chart(rows, args.output_dir / 'audio-path-delivery.png')
    print(f"Baseline {'PASS' if report['baseline_pass'] else 'FAIL'}; results: {args.output_dir}")
    return 0 if report['baseline_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
