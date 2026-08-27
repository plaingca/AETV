#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec .venv/bin/python scripts/train_gop_overlap.py train \
  --mode V8 \
  --out /pool0/AETV-runs/v8-overlap-full-w192-c8-window5 \
  --data-source stream \
  --train-gops 8 \
  --window-gops 5 \
  --emit-gops 3 \
  --synthesis-halo-frames 2 \
  --model-width 192 \
  --latent-channels 8 \
  --steps 75000 \
  --batch 2 \
  --accum 2 \
  --workers 4 \
  --lr 1e-4 \
  --min-lr 1e-6 \
  --clean-warmup 7500 \
  --channel-ramp 7500 \
  --channel-kind waveform \
  --snr-min 0 \
  --snr-max 18 \
  --p-fading 0.70 \
  --p-measured-path 0.40 \
  --checkpoint-interval 2500 \
  --keep-checkpoints 3 \
  --eval-interval 2500 \
  --eval-sequences 8 \
  --log-interval 25 \
  --gradient-checkpointing \
  --amp
