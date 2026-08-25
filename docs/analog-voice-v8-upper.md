# Analog voice plus upper-slice V8 AETV experiment

Date: 2026-08-25

## Waveform

This prototype abandons the learned 500-value audio latent and frequency-stacks ordinary analog speech with the released V8/W video waveform:

- composite sample rate: 12 kHz
- analog voice: 0--2,200 Hz, raised-cosine roll-off from 2,100 Hz
- guard: 2,200--2,500 Hz (300 Hz)
- V8/W carrier centers: 2,600--4,800 Hz
- filtered AETV skirt: 2,500--4,900 Hz
- source V8/W waveform: 450--2,650 Hz, shifted upward by 2,150 Hz
- model: `models/v8-hf3k-face-gan.pt`, 192x108 at 6 fps

The transmit mask is important. Frequency translation without it leaves OFDM symbol sidelobes well beyond the nominal edge. With the explicit upper-slice mask, measured guard power is 45.8 dB below voice-band power and 50.4 dB below AETV-band power.

This composite needs an approximately 5 kHz radio/DAX passband. It will not pass through the normal 2.7 kHz SSB filtering used by the existing W mode.

## Synchronization

The speech stream is delayed by exactly one one-second GOP. During transmit interval `n`, the upper slice carries video GOP `n`, while the lower slice carries speech GOP `n-1`. At the receiver, decoded video GOP `n-1` is already available when its delayed analog speech arrives.

This is the relative modem/playout delay. A live capture implementation may add another GOP of glass-to-glass latency if it must first buffer a complete source GOP before encoding.

## Ideal composite results

| Clip | Latent NMSE | Latent corr. | Video clean PSNR | Composite PSNR | Transport delta | Voice SI-SDR | STOI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.90e-4 | 0.999905 | 23.989 dB | 23.982 dB | -0.007 dB | 44.11 dB | 0.942 |
| 1 | 3.21e-4 | 0.999840 | 30.960 dB | 30.942 dB | -0.017 dB | 38.79 dB | 0.945 |
| 2 | 3.74e-4 | 0.999802 | 39.907 dB | 39.882 dB | -0.025 dB | 44.51 dB | 0.981 |

The remaining video softness is the V8 autoencoder reconstruction itself; the composite transport adds negligible degradation in the ideal loopback. Unlike the learned audio latent, the analog speech path remains plainly intelligible and nearly transparent.

## Artifacts

Everything is under `runs/analog-voice-v8-upper`:

- `all_samples_comparison.mp4`: three browser-compatible comparisons; source audio is left and recovered analog audio is right
- `sample_00_comparison.mp4` through `sample_02_comparison.mp4`: individual clips
- `composite_transmit.wav`: four-second 12 kHz frequency-stacked transmit waveform
- `voice_component_delayed.wav`: isolated one-GOP-delayed speech component
- `aetv_component_upper.wav`: isolated upper AETV component
- `composite_spectrum.png`: measured composite PSD
- `metrics.json`: per-clip and spectral metrics

## Reproduction

```bash
.venv/bin/python scripts/experiment_analog_voice_aetv.py --clips 3
```

The experiment uses `aetv/analog_av.py`. Nine focused analog/composite tests pass. The generated MP4 is H.264 Constrained Baseline with `yuv420p`, AAC-LC at 48 kHz, fast-start layout, and was fully decoded with FFmpeg after generation.

## Next validation

The current result proves ideal spectral coexistence and synchronization. Before integrating it into the station GUI, evaluate the composite through:

1. the actual 5 kHz radio/DAX filter response and transmitter linearity;
2. composite peak limiting and unequal voice/video power allocations;
3. AWGN, selective fading, and frequency offset with separate voice and V8 quality measurements;
4. causal streaming filters, whose group delay must be included in the audio playout delay.
