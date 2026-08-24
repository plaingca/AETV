#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

run_dir="${OUT_DIR:-runs/v8-hf3k-detail-face-saliency-20260824}"
init_checkpoint="${INIT_CHECKPOINT:-runs/v8-hf3k-openvid1m-lance64-20260823/checkpoint_step_116500.pt}"
steps="${STEPS:-12000}"

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
  --lr 5e-6 \
  --compile reduce-overhead \
  --clean-warmup 0 \
  --channel-ramp 250 \
  --snr-min 0 \
  --snr-max 16 \
  --p-fading 0.5 \
  --p-measured-path 0.4 \
  --mse-weight 1.0 \
  --l1-weight 1.2 \
  --dwt-weight 2.0 \
  --dwt-levels 3 \
  --grad-weight 1.0 \
  --temporal-weight 2.0 \
  --temporal-accel-weight 0.5 \
  --temporal-energy-weight 3.0 \
  --temporal-cosine-weight 0.2 \
  --lpips-weight 0.12 \
  --temporal-lpips-weight 0.12 \
  --consistency-weight 1.0 \
  --clean-anchor-weight 0.75 \
  --region-weight 1.25 \
  --detail-weight 1.5 \
  --region-boost 3.0 \
  --face-model data/teachers/face_detection_yunet_2023mar.onnx \
  --adv-weight 0 \
  --fm-weight 0 \
  --lecam-weight 0
