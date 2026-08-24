#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

base_checkpoint="${INIT_CHECKPOINT:-runs/v8-hf3k-face-detail-lr-sweep-20260824/lr-3e-6/checkpoint_step_000400.pt}"
sweep_root="${SWEEP_ROOT:-runs/v8-hf3k-face-gan-sweep-20260824}"
sweep_steps="${SWEEP_STEPS:-600}"
mkdir -p "$sweep_root"

# The global critic stays off.  These probes differ only in localized realism
# pressure, which makes their eval sheets directly comparable.
for face_adv_weight in 0.005 0.010 0.020; do
  label="$(printf '%s' "$face_adv_weight" | tr -d '+')"
  out_dir="$sweep_root/adv-$label"
  mkdir -p "$out_dir"
  final_checkpoint="$out_dir/checkpoint_step_$(printf '%06d' "$sweep_steps").pt"
  if [[ -f "$final_checkpoint" ]]; then
    echo "skipping completed face-adv=$face_adv_weight probe"
    continue
  fi
  echo "starting face-adv=$face_adv_weight -> $out_dir"
  set +e
  .venv/bin/python scripts/train.py \
    --mode V8 \
    --stage 2 \
    --init-checkpoint "$base_checkpoint" \
    --reset-steps \
    --out "$out_dir" \
    --steps "$sweep_steps" \
    --eval-interval 300 \
    --tb-interval 25 \
    --checkpoint-interval 300 \
    --keep-checkpoints 2 \
    --batch 8 \
    --model-width 128 \
    --latent-channels 3 \
    --threads 24 \
    --data-backend native \
    --lance-batch-size 256 \
    --lance-fetch-threads 4 \
    --encoded-queue-size 512 \
    --stream-timeout 300 \
    --lr 1e-6 \
    --compile reduce-overhead \
    --seed 20260826 \
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
    --face-adv-weight "$face_adv_weight" \
    --face-fm-weight 0.05 \
    --face-perceptual-weight 0.25 \
    --face-d-lr 1e-5 \
    --face-d-every 2 \
    --face-disc-warmup 75 \
    --face-crop-size 64 \
    --adv-weight 0 \
    --fm-weight 0 \
    --lecam-weight 0 \
    >"$out_dir/train.log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 && ! -f "$final_checkpoint" ]]; then
    echo "face-adv=$face_adv_weight failed before final checkpoint (exit $status)"
    exit "$status"
  fi
  if [[ $status -ne 0 ]]; then
    echo "face-adv=$face_adv_weight exited $status after its final checkpoint"
  fi
  echo "completed face-adv=$face_adv_weight"
done

echo "all face-gan probes completed"
