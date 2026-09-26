---
cursor:
  subagentId: "bc-cb0add16-30e3-5488-8e4d-95c8110feee2"
---

# Face-priority bit allocation (V8, 2.2 kHz)

**Result: no weight passed the kill check, so there was no full run and no PR.**

A transmitter-side face mask on the encoder input, plus heavy face weighting (5–20×), did not produce a clear, visible face win within 500 steps:

- **Face PSNR** rose by at most **+0.15 dB**.
- **Face-crop LPIPS** got **worse** (faces became smoother).
- **Landmark error** did not improve.
- **Whole-frame PSNR** fell by only 0.05–0.18 dB, far from the 1.5 dB budget that was allowed.

The encoder did start using the mask: it changes the transmitted latent by 14–22% of its RMS. But at this bandwidth, extra reals spent on a face that covers about 7% of the frame bought face *PSNR*, not face *structure*.

## Trade-off table

Pool figures are paired against the current face fine-tune (`models/v8-hf3k-mpp12-face-ft.pt`) on 29 pool face clips, with whole-frame on 40 pool clips. They use **fresh fade seeds (7900)**, because the face fine-tune was itself selected on seeds 7600.

A clear pass needs landmark error down, face PSNR up and face-crop LPIPS down, each by more than 2× SE.

| Sweep | Face weight k | Landmark Δ (% crop) | Face PSNR Δ (dB) | Face-crop LPIPS Δ | Whole-frame `mpp12` Δ (dB) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| A: mask input + face-weighted pixel MSE/L1 | 5 | +0.08 ± 0.27 | −0.09 ± 0.04 | −0.001 ± 0.002 | −0.15 | Fail |
| A | 10 | −0.68 ± 0.41 | −0.13 ± 0.04 | −0.003 ± 0.002 | −0.16 | Fail |
| A | 20 | −0.45 ± 0.46 | −0.04 ± 0.04 | **−0.004 ± 0.001** | −0.18 | Fail |
| B: mask input + face-crop reconstruction loss | 5 | **−0.76 ± 0.34** | −0.02 ± 0.03 | +0.009 ± 0.002 | −0.05 | Fail |
| B | 10 | −0.61 ± 0.40 | +0.02 ± 0.03 | +0.012 ± 0.002 | −0.07 | Fail |
| B | 20 | +0.08 ± 0.47 | **+0.09 ± 0.03** | +0.024 ± 0.003 | −0.05 | Fail |

Bold marks the only moves beyond 2× SE, and no row has all three. Every row stays inside the whole-frame limit (−1.5 dB).

**Why two sweeps:**

- **Sweep A** weighted only the pixel MSE and L1 by 1 + (k − 1)·mask. Those two terms are about 2% of the total loss, so even k = 20 barely changed the objective.
- **Sweep B** fixes that. It adds k × the full per-pixel reconstruction loss (MSE + L1 + gradient) on 64×64 face crops, so the face counts as a whole frame's worth of loss. At k = 20 the face is about 65% of the objective.
- A's kill check was first run on the biased seeds 7600 (it showed −0.08 to −0.19 dB face PSNR). The table uses the fresh-seed rescore. B was selected on the fresh seeds from the start.

### 64 eval clips, once, for the record

This is the k = 20 face-crop variant at step 500, not selected on eval. Face metrics are on the 38 face clips.

| Model | Landmark error | Face PSNR | Face-crop LPIPS | Whole-frame `mpp12` | Whole-frame LPIPS |
|---|---:|---:|---:|---:|---:|
| Face fine-tune (current) | 11.30 ± 1.68 | 19.43 | 0.294 | 20.74 ± 0.49 | 0.264 |
| Face-priority, k = 20 | 11.92 ± 1.75 | 19.59 | 0.315 | 20.68 ± 0.47 | 0.289 |
| **Paired Δ** | +0.62 ± 0.40 | **+0.15 ± 0.04** | **+0.021 ± 0.002** (worse) | −0.06 ± 0.05 | +0.025 |

## What was built (branch `cursor/v8-face-priority-eee2`, no PR)

- **Face mask input** (`aetv/face_priority.py`):
  - The transmitter runs YuNet on its own source GOP and feeds the encoder a fourth plane: 1 inside the face boxes expanded 1.2×, 0 elsewhere.
  - It enters through a separate zero-initialized stem convolution, with 30× the base learning rate. Step 0 is exactly the face fine-tune.
  - The wire is still 2,816 reals, and the receiver needs no mask.
  - Checkpoints carry `face_priority_input: true`, and the scorer's V8 adapter runs the detector at encode time.
- **Loss:** channel-only, as before.
  - Kept: the landmark heatmap consistency (weight 100) and face-region MSE (10).
  - Restored: face-crop VGG to its release weight of 0.25.
  - Added: the face weight k (sweep A: pixel weight map; sweep B: face-crop reconstruction loss).
  - No adversarial terms, and nothing that rewards texture absent from the source.
- **Face-crop LPIPS** in the shared scorer: AlexNet LPIPS on the same 256×256 2D-FAN face crops, source vs reconstruction.
- **Kill and selection rule** (`--face-priority`): all three face metrics must improve by more than 2× SE, with a stop at −1.5 dB whole-frame.

## Why the face win is not visible

- **Budget:** the face is about 7% of a 192×108 frame, served by 2,816 reals per second. Tilting the encoder toward it moved the latent, but in 500 steps the decoder turned that into a little more face PSNR by smoothing. That is the MSE optimum, and it raises face LPIPS.
- **Structure:** landmark placement is already carried by the landmark term. Extra pixel weight does not add structure.
- **Budget not spent:** the whole-frame drop never got near the allowed 1.5 dB (at most −0.18). The model has not really reallocated capacity from the background yet. Doing so probably needs a much longer schedule, or an explicit mechanism.
- **Suggested next attempt:** a **face ROI sub-stream**, an architectural change rather than a reweighting. Reserve a fixed share of the 2,816 reals (e.g. 25–35%) for a small face-crop codec (64×64 aligned crop) and composite it at the receiver. That guarantees face bits, and it trades background PSNR for face detail by construction. It needs a new sub-network and a multi-hour schedule, with a longer kill check (≥ 2k steps).

## Frames

`media/face-priority/`, with panels source | current face fine-tune | face-priority k = 20, all on `mpp12`:

- `fp-val{35,27,53}-f{02,05,09}.png`: face clips.
- `fp-val59-f{02,05,09}.png`: a non-face clip (flower). It still decodes sensibly and is essentially unchanged.
- `fp-val{35,27,53}-f05-face-zoom.png`: zoomed 256×256 face crops (source | face fine-tune | face-priority). Streaks at the crop edge on val35 are border padding where the face touches the frame edge.
- `face-priority-mpp12-side-by-side.mp4`: 3 face clips and 1 non-face clip × 12 frames at 6 fps, played twice.

Visually, the face-priority faces are marginally smoother than the face fine-tune's, with no added structure.

## Files

- **Code:** `aetv/face_priority.py`; face-crop LPIPS and mask-aware encoding in `aetv/shared_eval.py`; `--face-priority`, `--face-loss-weight`, `--mask-lr-scale` and `--whole-frame-stop` in `scripts/finetune_v8_channel.py`.
- **Runs on Beastmode:** `runs/fp-kill-v1/` (sweep A), `runs/fp-kill/` (sweep B), `runs/face-priority/eval64.json`. No new checkpoint in `models/`; the step-500 variants stay under `runs/`.
- **JSON copies:** `internal/face-priority/`.
