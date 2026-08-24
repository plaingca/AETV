#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

run_dir="${OUT_DIR:-runs/v8-hf3k-face-detail-v2-20260824}"
init_checkpoint="${INIT_CHECKPOINT:-runs/v8-hf3k-face-detail-lr-sweep-20260824/lr-3e-6/checkpoint_step_000400.pt}"
steps="${STEPS:-6000}"

exec .venv/bin/python scripts/train.py \
  --mode V8 \
  --stage 2 \
  --init-checkpoint "$init_checkpoint" \
  --reset-steps \
  --out "$run_dir" \
  --steps "$steps" \
  --eval-interval 1000 \
  --tb-interval 25 \
  --checkpoint-interval 500 \
  --keep-checkpoints 6 \
  --batch 8 \
  --model-width 128 \
  --latent-channels 3 \
  --threads 24 \
  --data-backend native \
  --lance-batch-size 256 \
  --lance-fetch-threads 4 \
  --encoded-queue-size 512 \
  --stream-timeout 300 \
  --lr 3e-6 \
  --compile reduce-overhead \
  --seed 20260825 \
  --clean-warmup 0 \
  --channel-ramp 1 \
  --snr-min 0 \
  --snr-max 16 \
  --p-fading 0.5 \
  --p-measured-path 0.4 \
  --mse-weight 0.25 \
  --l1-weight 0.8 \
  --dwt-weight 3.0 \
  --dwt-levels 3 \
  --grad-weight 1.5 \
  --temporal-weight 1.5 \
  --temporal-accel-weight 0.3 \
  --temporal-energy-weight 2.0 \
  --temporal-cosine-weight 0.2 \
  --lpips-weight 0.18 \
  --temporal-lpips-weight 0.10 \
  --consistency-weight 1.0 \
  --clean-anchor-weight 1.0 \
  --region-weight 1.5 \
  --detail-weight 3.0 \
  --contrast-weight 2.5 \
  --region-boost 12.0 \
  --face-model data/teachers/face_detection_yunet_2023mar.onnx \
  --adv-weight 0 \
  --fm-weight 0 \
  --lecam-weight 0
