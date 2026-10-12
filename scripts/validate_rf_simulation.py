#!/usr/bin/env python3
"""Bounded seeded production real-audio modem sweep (no hardware/models)."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aetv.analog_validation import unit_rms_latents
from aetv.config import AETV_MODES, RELEASE_MODES, SNR_REF_BW_HZ
from aetv.hfchannel import ChannelProfile, emulate, freq_shift
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks

@dataclass(frozen=True)
class Scenario:
    name: str
    snr_db: float | None = None
    fading: str = 'none'
    frequency_offset_hz: float = 0.
    clock_ppm: float = 0.
    clip_rms: float | None = None
    tone_sir_db: float | None = None

SCENARIOS = (
    Scenario('clean'), Scenario('awgn12',12), Scenario('awgn6',6),
    Scenario('awgn0',0), Scenario('mpp12',12,'mpp'), Scenario('mpp6',6,'mpp'),
    Scenario('mpd12',12,'mpd'), Scenario('gaussian_mpp12',12,'mpp-gauss'),
    Scenario('offset_plus25',12,frequency_offset_hz=25),
    Scenario('offset_minus25',12,frequency_offset_hz=-25),
    Scenario('clock_plus100',12,clock_ppm=100),
    Scenario('clock_minus100',12,clock_ppm=-100),
    Scenario('clip_1rms',12,clip_rms=1.), Scenario('tone_sir6',12,tone_sir_db=6),
    Scenario('combined',12,'mpp',25,100),
)

def impair(audio, scenario, fs, center_hz, seed):
    """Clipping -> fading/noise -> CFO -> tone -> sample clock; no RX AGC.

    Noise is referenced to pre-fade signal power in 2500 Hz. Resampling uses
    rational polyphase filtering, representing clock skew to one ppm.
    """
    audio = np.asarray(audio,dtype=np.float64)
    rms = float(np.sqrt(np.mean(audio**2)))
    if scenario.clip_rms is not None:
        level = scenario.clip_rms*rms
        audio = np.clip(audio,-level,level)
    audio = emulate(audio,ChannelProfile(scenario.name,scenario.name,scenario.snr_db,scenario.fading),seed=seed,fs=fs)
    if scenario.frequency_offset_hz:
        audio = freq_shift(audio,scenario.frequency_offset_hz,fs=fs)
    if scenario.tone_sir_db is not None:
        amplitude = np.sqrt(2)*rms*10**(-scenario.tone_sir_db/20)
        audio = audio+amplitude*np.cos(2*np.pi*(center_hz+137)*np.arange(len(audio))/fs)
    if scenario.clock_ppm:
        from math import gcd
        up,down = 1_000_000+round(scenario.clock_ppm),1_000_000
        factor = gcd(up,down)
        audio = signal.resample_poly(audio,up//factor,down//factor)
    return audio.astype(np.float32)

def run_trial(mode_name,scenario,seed,gops=4):
    if gops < 1:
        raise ValueError('gops must be positive')
    mode = AETV_MODES[mode_name]
    fs = mode.geometry.fs
    originals = [unit_rms_latents(mode_name,10_000+seed*1000+i) for i in range(gops)]
    tx = np.concatenate(list(modulate_continuous_chunks(originals,mode_name,'N0CALL')))
    rx = impair(tx,scenario,fs,mode.geometry.fcenter_hz,seed)
    receiver = StreamingDemodulator(mode.band,mode_name=mode_name,continuous=True,
                                   timing_tracking=True,boundary_tracking=True)
    results = []
    sizes = [max(1,fs//23),max(1,fs//11),max(1,fs//7)]
    start = tick = 0
    began = time.monotonic()
    while start < len(rx):
        end = start+sizes[tick%len(sizes)]
        results.extend(receiver.feed(rx[start:end]))
        start,tick = end,tick+1
    measurements = []
    for result in results:
        for latent,weights in zip(result.gops_latents,result.gops_weights):
            if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(weights)):
                raise RuntimeError('nonfinite receiver output')
            correlations = [float(np.corrcoef(original,latent)[0,1]) for original in originals]
            best = int(np.argmax(correlations))
            measurements.append({'source_gop':best,'correlation':correlations[best],
                'nmse':float(np.mean((latent.astype(np.float64)-originals[best])**2)),
                'mean_confidence':float(np.mean(weights)),'estimated_snr_db':float(result.snr_db)})
    # Measurement oracle only, never supplied to acquisition/equalization.
    valid = [r for r in measurements if r['correlation']>0.25]
    return {'mode':mode_name,'sample_rate_hz':fs,'scenario':asdict(scenario),'seed':seed,
            'sent_gops':gops,'received_gops':len(measurements),
            'matched_unique_gops':len({r['source_gop'] for r in valid}),
            'unmatched_outputs':len(measurements)-len(valid),
            'mean_nmse':float(np.mean([r['nmse'] for r in valid])) if valid else None,
            'mean_correlation':float(np.mean([r['correlation'] for r in valid])) if valid else None,
            'receive_seconds':time.monotonic()-began,'measurements':measurements}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes',nargs='+',choices=list(AETV_MODES),default=list(RELEASE_MODES))
    parser.add_argument('--scenarios',nargs='+',choices=[s.name for s in SCENARIOS])
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--gops',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 1<=args.seeds<=100 or not 1<=args.gops<=100:
        parser.error('seeds and gops must each be between 1 and 100')
    rows = []
    for mode in args.modes:
        for scenario in SCENARIOS:
            if args.scenarios and scenario.name not in args.scenarios:
                continue
            for seed in range(args.seeds):
                row = run_trial(mode,scenario,seed,args.gops)
                rows.append(row)
                print(f"{mode} {scenario.name} seed={seed}: {row['matched_unique_gops']}/{args.gops} GOPs, NMSE={row['mean_nmse']}",flush=True)
    report = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'snr_reference_bandwidth_hz':SNR_REF_BW_HZ,
              'scope':'Production analog modem, synthetic unit-RMS latents; not video or physical RF',
              'source_matching':'Post-reception maximum correlation > 0.25; not used by receiver',
              'trials':rows}
    def finite(value):
        if isinstance(value,float) and not np.isfinite(value):
            return None
        if isinstance(value,dict):
            return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):
            return [finite(v) for v in value]
        return value
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(finite(report),indent=2,allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
