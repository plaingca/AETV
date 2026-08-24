#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

base_checkpoint="${INIT_CHECKPOINT:-runs/v8-hf3k-detail-face-saliency-20260824/checkpoint_step_012000.pt}"
sweep_root="${SWEEP_ROOT:-runs/v8-hf3k-face-detail-lr-sweep-20260824}"
sweep_steps="${SWEEP_STEPS:-400}"
mkdir -p "$sweep_root"

for learning_rate in 1e-6 3e-6 8e-6 2e-5; do
  label="$(printf '%s' "$learning_rate" | tr -d '+')"
  out_dir="$sweep_root/lr-$label"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/checkpoint_step_$(printf '%06d' "$sweep_steps").pt" ]]; then
    echo "skipping completed $learning_rate sweep"
    continue
  fi
  echo "starting lr=$learning_rate -> $out_dir"
  set +e
  .venv/bin/python scripts/train.py \
    --mode V8 \
    --stage 2 \
    --init-checkpoint "$base_checkpoint" \
    --reset-steps \
    --out "$out_dir" \
    --steps "$sweep_steps" \
    --eval-interval "$sweep_steps" \
    --tb-interval 25 \
    --checkpoint-interval "$sweep_steps" \
    --keep-checkpoints 1 \
    --batch 8 \
    --model-width 128 \
    --latent-channels 3 \
    --threads 24 \
    --data-backend native \
    --lance-batch-size 256 \
    --lance-fetch-threads 4 \
    --encoded-queue-size 512 \
    --stream-timeout 300 \
    --lr "$learning_rate" \
    --compile reduce-overhead \
    --seed 20260824 \
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
    --lecam-weight 0 \
    >"$out_dir/train.log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 && ! -f "$out_dir/checkpoint_step_$(printf '%06d' "$sweep_steps").pt" ]]; then
    echo "lr=$learning_rate failed before producing its final checkpoint (exit $status)"
    exit "$status"
  fi
  if [[ $status -ne 0 ]]; then
    echo "lr=$learning_rate exited $status during interpreter shutdown; final checkpoint is complete"
  fi
  echo "completed lr=$learning_rate"
done

echo "all learning-rate sweeps completed"
