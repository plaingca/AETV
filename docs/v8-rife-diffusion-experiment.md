# V8 receiver post-processing experiment

This experiment leaves the released `models/v8-hf3k-face-gan.pt`, V8 latent
budget, modem, and wire format unchanged. It tests receiver-only processing in
separate arms:

1. frame hold (the honest 6-to-12 fps control),
2. linear interpolation,
3. pretrained Practical-RIFE 4.25-lite interpolation,
4. SNR-conditioned residual diffusion, evaluated separately first and only
   combined with interpolation if it passes the restoration gate.

## Why interpolation uses a separate source path

The existing V8 `.pt` caches contain only 6 fps pictures. They cannot measure
the accuracy of an invented midpoint. `evaluate-rife` therefore decodes the
source at 12 fps, transmits only even frames through two independent V8 GOPs,
and scores all 23 output frames against the retained 12 fps source. Scene cuts
are detected before RIFE so unrelated shots are held rather than blended.

## Commands

The downloaded Practical-RIFE checkout and 4.25-lite weights used in this run
are kept under `runs/v8-rife-diffusion-20260826/external/`.

```bash
.venv/bin/python scripts/experiment_v8_receiver_postprocess.py evaluate-rife \
  --input /path/to/native-high-frame-rate-source.mp4 \
  --clips 32 \
  --report runs/v8-rife-diffusion-20260826/rife-paired-32/report.json

.venv/bin/python scripts/experiment_v8_receiver_postprocess.py train-restorer \
  --steps 2000 \
  --restorer runs/v8-rife-diffusion-20260826/restorer-v2.pt

.venv/bin/python scripts/experiment_v8_receiver_postprocess.py evaluate-restorer \
  --restorer runs/v8-rife-diffusion-20260826/restorer-v2.pt \
  --sample-steps 12 \
  --report runs/v8-rife-diffusion-20260826/restorer-v2-paired-32.json
```

## Current result

RIFE is a viable optional 12 fps presentation path, but it did not beat linear
interpolation on the paired 32-segment Simpsons run. Relative to frame hold,
RIFE improved PSNR by 0.14-0.19 dB and sharply reduced the GOP-boundary error
ratio. Relative to linear interpolation it was within uncertainty on
PSNR/SSIM/LPIPS and had slightly worse boundary ratios in every channel cell.

The first diffusion candidate exposed a sampler mismatch and was rejected.
The corrected cosine-schedule, unit-residual v2 candidate was also rejected:
at 12 sampling steps its clean PSNR changed 22.81 to 18.28 dB and LPIPS 0.173
to 0.537. Fifty steps remained worse by 2.92 dB clean PSNR. The module and
evaluator are retained for redesign, but neither diffusion checkpoint is a
production candidate.

The likely next restoration experiment is deterministic SNR/confidence-aware
residual regression (or diffusion initialized around a deterministic estimate)
with an explicit identity/LPIPS anchor, before returning to stochastic pixel
sampling.
