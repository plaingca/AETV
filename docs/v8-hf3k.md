# V8 experimental HF-3k mode

V8 is a narrow-channel derivative of the Flex-8k V7 work. It is intended for
a conventional SSB transmit/receive filter rather than an 8 kHz DIGU slice.

| Field | V8 |
|---|---|
| Video | 192x108 color, 6 frames/s |
| GOP | 6 frames in exactly 1 RF second |
| Audio sample rate | 8 kHz |
| OFDM carriers | 45 at 50 Hz spacing |
| Carrier centers | 450 through 2650 Hz |
| TX bandpass | 350 through 2750 Hz |
| Analog latent budget | 2,816 real values/s |

The carrier span is 2.20 kHz center-to-center (about 2.25 kHz occupied), with
the transmit conditioning kept below 2.75 kHz. This leaves practical margin
inside a nominal 3 kHz USB channel; actual radios should still be checked with
the waterfall before transmitting.

## Checkpoints

The primary checkpoint is `models/v8-hf3k-perceptual.pt`. Leaving the
checkpoint field empty selects it automatically when V8 is selected. The
alternate `models/v8-hf3k-robust.pt` trades some clean-channel fidelity for
better recovery around 0 dB and under severe MPP fading.

```powershell
uv run aetv send --mode V8 --checkpoint models/v8-hf3k-perceptual.pt `
  --source clip.mp4 --callsign YOURCALL --gops 10 --out v8-test.wav

uv run aetv receive --mode V8 --checkpoint models/v8-hf3k-perceptual.pt `
  --wav v8-test.wav --out v8-test.mp4
```

Both ends must use the same checkpoint.

## Held-out results

Checkpoint selection used 32 real, disjoint OpenVid clips with identical modem
and channel realizations for every model. LPIPS is lower-is-better and PSNR is
higher-is-better.

| Condition | Zero-shot PSNR | V8 PSNR | Zero-shot LPIPS | V8 LPIPS |
|---|---:|---:|---:|---:|
| Clean | 15.85 | 23.35 | 0.531 | 0.250 |
| 12 dB | 15.88 | 21.93 | 0.526 | 0.259 |
| 6 dB | 15.86 | 21.22 | 0.527 | 0.306 |
| MPG fading | 15.85 | 21.43 | 0.516 | 0.286 |
| MPP fading | 15.94 | 18.55 | 0.523 | 0.433 |

The motion-aware fine-tune intentionally spends about 0.2 dB clean and 0.5 dB
through the modem to avoid the nearly static minimum. Against the previous V8
checkpoint, its clean LPIPS improves by 0.017 and its 0 dB LPIPS by 0.020;
both paired changes exceed twice their standard errors. The robust alternate
remains available for maximum PSNR under severe fading.

| Temporal metric (32 held-out clips) | Previous V8 | Motion-aware V8 |
|---|---:|---:|
| Clean motion energy retained | 19.6% | 69.1% |
| Clean temporal correlation | 0.078 | 0.151 |
| 6 dB motion energy retained | 21.1% | 94.5% |
| 6 dB temporal correlation | 0.051 | 0.100 |

Motion energy alone can be inflated by channel noise, so checkpoint selection
also required temporal correlation to improve and LPIPS not to regress.

## Warm-starting a V8 fine-tune

Stage 2 now retains the broad HF channel distribution while drawing 40% of
faded examples from a measured 40 m mixture. The mixture is centered on the
K9CZI-1 capture: 0.45-0.75 ms differential delay, 0.18-0.30 Hz Doppler, a
strong -2 to 0 dB echo, and -1 to 10.5 dB SNR. `ota40m` at 0.6 ms, 0.24 Hz and
5 dB is included in every periodic modem evaluation so a run cannot improve
the training average while silently regressing this path.

```powershell
uv run aetv train -- --mode V8 --stage 2 --out runs/v8-hf3k `
  --init-checkpoint models/v7-flex8k-severe.pt --reset-steps --steps 10000 --amp
```

For a short path-focused adaptation from the current V8 model, keep generic
flat and fading examples in the batch rather than training solely on the new
capture family:

```powershell
uv run aetv train -- --mode V8 --stage 2 --out runs/v8-hf3k-ota40m `
  --init-checkpoint models/v8-hf3k-perceptual.pt --reset-steps `
  --model-width 128 --latent-channels 3 --compact --steps 1000 `
  --snr-min -2 --snr-max 12 --p-fading 0.60 --p-measured-path 0.60 --amp
```

In that recipe, 36% of all stage-2 examples use the measured-path family, 24%
use the general Watterson distribution, and 40% remain non-fading. Set
`--p-measured-path 0` to reproduce the earlier sampler.

V8 has protocol mode index 8, so a receiver can distinguish it from the older
W-band modes.
