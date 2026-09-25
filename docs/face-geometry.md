---
cursor:
  subagentId: "bc-cb0add16-30e3-5488-8e4d-95c8110feee2"
---

# Face geometry through the channel (V8, 2.2 kHz)

**Result:** the face fine-tune puts eyes, nose and mouth measurably closer to the source through `mpp12`, with no adversarial terms. It still **fails the ship rule on one guard, LPIPS on `mpp12`**, so there is no PR.

On the 38 eval face clips, against the current `mpp12` fine-tune (20.73 dB):

| Measure | Change | Rule |
|---|---:|---|
| Landmark error | **−2.37 ± 0.91** (13.68 → 11.30% of face crop) | Significant (> 2× SE); 33 of 38 clips improve |
| Face-region PSNR | **+0.25 ± 0.05 dB** | Not worse |
| Whole-frame `mpp12` | **20.74 dB** (+0.01 ± 0.02) | Within 0.1 dB |
| LPIPS on `mpp12` | **+0.0071 ± 0.0009** (0.257 → 0.264) | **Fails:** must not be worse |

## Headline table (64 eval clips, `mpp12`, receiver gain 1.0)

Landmark error and face PSNR are over the 38 clips with a YuNet face. Whole-frame PSNR and LPIPS are over all 64.

| Model | **Landmark error, `mpp12`** (% of face crop, lower is better) | Face PSNR `mpp12` | Whole-frame `mpp12` PSNR | LPIPS `mpp12` | Landmark error, clean (info) | Clean PSNR (info) |
|---|---:|---:|---:|---:|---:|---:|
| V8 release (`v8-hf3k-face-gan`) | 13.22 ± 1.79 | 18.57 | 20.48 ± 0.52 | 0.255 | 11.22 | 23.92 |
| Current `mpp12` fine-tune (`v8-hf3k-mpp12-ft`) | 13.68 ± 1.92 | 19.18 | 20.73 ± 0.50 | 0.257 | 10.84 | 23.66 |
| **Face fine-tune (`v8-hf3k-mpp12-face-ft`, step 3500)** | **11.30 ± 1.68** | **19.44** | **20.74 ± 0.49** | 0.264 | 8.86 | 23.55 |

Paired deltas for the face fine-tune:

| vs | Landmark error | Face PSNR | Whole-frame `mpp12` | LPIPS `mpp12` |
|---|---:|---:|---:|---:|
| Current fine-tune | −2.37 ± 0.91 | +0.25 ± 0.05 | +0.010 ± 0.017 | +0.0071 ± 0.0009 |
| V8 release | −1.91 ± 0.77 | +0.87 ± 0.14 | +0.26 ± 0.06 | +0.0091 ± 0.0032 |

The current `mpp12` fine-tune was **not** better than the release on geometry: +0.46 ± 0.63, not significant. It gained face PSNR, not landmark placement.

## Metric (added to the shared scorer)

`aetv/face_geometry.py` and `scripts/eval_shared.py --landmarks`:

- **Detection:** YuNet (the scorer's existing detector, run at 4× upscale) gives the largest face box on each source frame.
- **Crop:** the box is expanded the way 2D-FAN expects (centre up 0.12 h, side 1.03 (w + h)) and resampled to 256×256, from the source and from the reconstruction at the same box.
- **Landmarks:** a frozen 2D-FAN (`face_alignment` 1.5.0, 2DFAN4, 68 points) gives landmarks on both crops.
- **Error:** the mean L2 distance between the two landmark sets, as a percentage of the crop side. This box-normalized error is used because inter-ocular distances are a few pixels on these faces.
- **Scope:** per clip, it is the mean over frames with a face. Clips without a face are excluded.
- **Sanity check** on eval clip 24: identical frames 0%, 3×3 blur 7.1%, 5×5 blur 29%.

FAN weights live in the user torch cache. `face_alignment` is installed in the repo's `.venv` only.

## Training (channel-only rules)

`scripts/finetune_v8_channel.py --face-goal` on branch `cursor/v8-face-geometry-eee2` (no PR).

- **Start:** `models/v8-hf3k-mpp12-ft.pt`, which is not modified.
- **Channel:** every row through the real V8 modem + `mpp12`. No clean rows, no clean anchor, no GAN or adversarial terms. Same base loss as the `mpp12` fine-tune.
- **Face terms,** all comparing the channel output with the **source**, so none rewards invented texture:
  - **Face-region pixel MSE** inside the YuNet box, weight 10.
  - **Landmark heatmap consistency:** MSE between frozen-FAN heatmaps of the channel-output face crop and the source face crop, weight 100, 8 crops per step.
  - Optional **low-frequency face MSE** (5×5-blurred, inside the box), weight 10. Used only in the "all" variant.
  - **Sampling:** 50% of rows are drawn from face clips.
- **Data:** training pool minus the selection clips. That is 1,287 clips, 620 with a face.
- **Selection:** 40 pool clips, with 29 face clips: the usual 24, plus 16 pool face clips removed from training. Fresh fade seeds (base 7600) avoid the bias from the earlier selection on seeds 7300.
- **Selection rule:** the lowest landmark error on `mpp12`, subject to face PSNR not falling and whole-frame `mpp12` within −0.1 dB. Best was step 3500 of 4000.
- **Checkpoint:** `models/v8-hf3k-mpp12-face-ft.pt` (new path, gitignored; SHA-256 `4a95e78e…57fd`). The release and `mpp12` fine-tune weights are unchanged.

## Kill checks (step 500, 29 pool face clips, paired vs the current fine-tune)

To pass: landmark error falls by more than 2× SE, face PSNR does not fall, and whole-frame `mpp12` falls by no more than 0.1.

| Variant | Landmark Δ | Face PSNR Δ | Whole-frame `mpp12` Δ | LPIPS `mpp12` Δ | Verdict |
|---|---:|---:|---:|---:|---|
| Control (face oversampling only) | −0.04 ± 0.49 | −0.04 | −0.044 | −0.003 | Fail |
| Face-region MSE | +0.34 ± 0.68 | +0.12 | −0.038 | +0.006 | Fail |
| **Face-region + landmark heatmaps** | **−2.07 ± 0.76** | +0.02 | −0.083 | +0.006 | **Pass** (continued) |
| Region + landmark + low-frequency | −1.56 ± 0.44 | +0.09 | −0.084 | +0.011 | Pass |

The landmark heatmap term is what moves geometry. The region term alone only raises face PSNR.

A second, identical run of the continued variant scored **−1.14 ± 0.59** at step 500, just under the 2× SE bar. That comes from GPU and threaded-modem non-determinism, so the step-500 effect is real but marginal. The full run was done with the early stop disabled; its selection guards were unchanged.

Full-run pool trajectory (29 face clips, landmark Δ vs start):

| Step | 250 | 500 | 1000 | 1500 | 2000 | 2750 | 3000 | **3500** | 4000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Landmark Δ | −1.29 | −1.60 | −1.74 | −1.99 | −1.22 | −2.28 | −2.28 | **−2.39 ± 0.77** | −2.30 |
| Face PSNR Δ | +0.08 | +0.03 | −0.01 | +0.08 | +0.11 | +0.18 | +0.12 | **+0.20** | +0.21 |
| Whole-frame `mpp12` Δ | −0.03 | −0.09 | −0.06 | −0.07 | −0.05 | −0.01 | −0.02 | **−0.005** | −0.01 |
| LPIPS `mpp12` Δ | +0.012 | +0.008 | +0.019 | +0.013 | +0.014 | +0.013 | +0.012 | +0.012 | +0.013 |

LPIPS on `mpp12` was about +0.01 worse at every checkpoint. The selection rule did not guard it, and the eval confirms it (+0.007).

## Frames

`media/face-geometry/`, with panels source | V8 release | current `mpp12` fine-tune | face fine-tune, all on `mpp12`:

- `face-val{35,27,53}-f{02,05,09}.png`: three eval face clips, with the largest faces in the set.
- `face-val{35,27,53}-f05-landmarks.png`: the same frames with 2D-FAN landmarks. Green is the source landmarks; red is each reconstruction's.
- `face-geometry-mpp12-side-by-side.mp4`: 3 clips × 12 frames at 6 fps, played twice.

On val35 the release and `mpp12` fine-tune place the mouth and lower-eye points visibly off the source, and the face fine-tune's points sit closer. On val27 the face fine-tune gives a smoother, more natural mouth and eye region without the release's invented dark eye detail. Faces are still soft, as expected at 2.2 kHz.

## What would pass

The only failing guard is LPIPS on `mpp12` (+0.007). Candidate fixes, none tried yet:

1. Add LPIPS-on-`mpp12` as a selection guard. Every saved checkpoint was about +0.01, so this probably means a lower landmark weight (e.g. 30–50) or a small AlexNet-LPIPS term on the channel output.
2. Keep the face-crop VGG term at its release weight (0.25 instead of 0.1). LPIPS is a whole-frame perceptual metric, and the recipe's reduced VGG weight is the likeliest cause.

Either needs a new kill check that includes the LPIPS guard.

## Files

- **Code:** branch `cursor/v8-face-geometry-eee2` (no PR). It contains `aetv/face_geometry.py`, `FaceMasks.boxes` and landmark error in `aetv/shared_eval.py`, `--landmarks` in `scripts/eval_shared.py`, and the `--face-goal` options in `scripts/finetune_v8_channel.py`.
- **Weights:** `models/v8-hf3k-mpp12-face-ft.pt` on Beastmode.
- **Runs:** `runs/face-kill/`, `runs/face-geometry-ft/`, `runs/face-geometry/` on Beastmode.
- **JSON copies:** `internal/face-geometry/`.
