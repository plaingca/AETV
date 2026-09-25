---
cursor:
  subagentId: "bc-cb0add16-30e3-5488-8e4d-95c8110feee2"
---

# VVC as a teacher for V8 at 2.2 kHz

**Result: no gain. Every variant failed its kill check, so there was no full run and no PR.**

VVC bit-allocation weighting and VVC output distillation were each tested for 500 steps from the channel-only fine-tune (`models/v8-hf3k-mpp12-ft.pt`, 20.73 dB `mpp12`). Neither moved `mpp12` PSNR relative to a no-teacher control:

| Variant vs control (fresh fade seeds, 24 pool clips) | Paired Δ `mpp12` |
|---|---:|
| Importance weighting | +0.008 ± 0.012 |
| Distillation | −0.009 ± 0.006 |
| Both | +0.010 ± 0.012 |

**Why:** at the realistic 2.2 kHz rate (≈ 4 kb/s, 1 s intra), the VVC decode is **23.3–23.5 dB**. That is no better than the fine-tune's own clean-wire decode (**23.66 dB**). The teacher carries no more picture than the student already sends. The 3 dB that V8 loses is lost in the channel (clean 23.66 → `mpp12` 20.73), and a source-coding teacher says nothing about it.

## Setup (project rules)

- **Start:** `models/v8-hf3k-mpp12-ft.pt`, which is not modified.
- **Channel-only recipe:** every training row goes through the real V8 OFDM modem + `emulate("mpp12")`, with no clean rows and no clean anchor. The loss weights are the same as the fine-tune's; lr 1e-5 with a 50-step warmup and cosine decay. Training windows are aligned to the two V8 GOPs (frames 0–5 and 6–11) so they match the VVC GOPs.
- **Selection and kill checks:** 24 training-pool clips with fade seed base 7300, `mpp12` PSNR only. The 64 eval clips were not used for any decision.
- **Kill criterion:** at step 500, pool `mpp12` ≥ +0.15 dB paired over the fine-tune.

### VVC teacher (`scripts/precompute_vvc_teacher.py`)

- **Encoding:** VVenC `slow`, `qpa=0`, 1 s intra period, 192×108 at 6 fps. Per clip, the QP is the lowest in {38, 41, 44, 47, 50, 53} whose raw stream fits 4.5 kb/s.
- **Training pool (1,327 clips):** mean **3.94 kb/s**, median QP 47, mean VVC PSNR **23.48 dB**. The mean sits below 4.5 kb/s because of the 3-step QP grid.
- **Importance map:** there is no per-block bit count from ffmpeg, so the map is a proxy for where VVC spent bits. It is the 8×8-box-blurred energy of (decode at the chosen QP − decode at QP 62), which is what the extra bits bought, normalized to mean 1 per GOP.

### Variants

| Variant | Loss change |
|---|---|
| Control | Same recipe, aligned windows, no teacher |
| 1. Importance weighting | Pixel MSE and L1 weighted by 1 + α·(map − 1), α = 1 (renormalized to mean 1) |
| 2. Output distillation | Pixel MSE target = 0.5 · source + 0.5 · VVC decode |
| 1 + 2 | Both |
| 3. Motion vectors / residual as encoder targets | **Not run.** ffmpeg does not export VVC motion vectors or per-block residuals, and it would need a new encoder feature head. Not cheap, and variants 1–2 show the teacher carries no extra signal at this rate |

## Kill-check table (step 500, 24 pool clips, paired Δ `mpp12` PSNR vs the 20.73 dB fine-tune)

| Variant | Step 250 | **Step 500 (kill check)** | LPIPS `mpp12` Δ | Verdict | Re-scored with fresh fade seeds (7600): vs fine-tune | vs control |
|---|---:|---:|---:|---|---:|---:|
| Control (no teacher) | −0.123 ± 0.039 | **−0.169 ± 0.071** | +0.007 | Fail | −0.052 ± 0.054 | — |
| 1. Importance | −0.142 ± 0.057 | **−0.164 ± 0.079** | +0.008 | Fail | −0.044 ± 0.056 | +0.008 ± 0.012 |
| 2. Distillation | −0.136 ± 0.043 | **−0.184 ± 0.067** | +0.007 | Fail | −0.060 ± 0.051 | −0.009 ± 0.006 |
| 1 + 2 | −0.131 ± 0.053 | **−0.168 ± 0.079** | +0.009 | Fail | −0.041 ± 0.056 | +0.010 ± 0.012 |

The pool baseline for the fine-tune is 20.878 dB on the selection seeds. It is **20.800 dB** on fresh seeds.

About 0.08 dB of each kill-check drop is selection bias: the fine-tune was chosen on those same clips and fade draws. The rest is small and not significant (−0.04 to −0.06 ± 0.05). The teacher terms change nothing relative to the control (within ±0.01 dB).

### Eval (64 clips, once, for the record; not used for any decision)

| | `mpp12` PSNR | Paired Δ | LPIPS `mpp12` Δ |
|---|---:|---:|---:|
| Channel-only fine-tune (current best) | 20.73 ± 0.50 | — | — |
| VVC-importance, step 500 | 20.68 ± 0.50 | −0.055 ± 0.028 | +0.006 ± 0.001 |

## Frames

`media/vvc-teacher/`:

- `vvc-teacher-val{00,24,48}-f{02,05,09}.png`: source | VVC at the teacher rate (error-free) | current fine-tune on `mpp12` | VVC-importance variant on `mpp12`.
- `vvc-teacher-mpp12-side-by-side.mp4`: 3 clips × 12 frames at 6 fps, played twice.

The teacher rates on the three eval clips:

| Clip | Rate | QP | VVC PSNR |
|---|---:|---:|---:|
| val00 | 4.3 kb/s | 47 | 22.4 dB |
| val24 | 4.3 kb/s | 47 | 20.5 dB |
| val48 | 3.8 kb/s | 41 | 29.5 dB |

At this rate VVC is blocky and blurred, no more detailed than the V8 decode.

## What this rules out, and what is left

- **Ruled out:** a teacher that only says which pixels are worth source bits. At 2.2 kHz, V8's clean wire already matches VVC at the rate the channel can realistically carry (`docs/psnr-feasibility.md`: VVC at 4.0–4.8 kb/s = 23.3–24.3 dB). The student has already learned what is worth carrying.
- **Where the dB are:** robustness to the fade. The clean → `mpp12` gap is 2.9 dB for the fine-tune. Ideas that target it are more promising than ideas that target source coding:
  - receiver/decoder conditioning on the modem confidence;
  - unequal protection of the latent coordinates, i.e. a channel-coding teacher rather than a source-coding one;
  - longer interleaving or latency.
- **Also ruled out here:** extra 500-step continuations of the fine-tune at lr 1e-5. The control shows they do not improve it (−0.05 ± 0.05 on fresh seeds).

## Files

- Code on branch `cursor/v8-vvc-teacher-eee2` (no PR): `scripts/precompute_vvc_teacher.py`, plus the `--teacher-dir`, `--importance-weight`, `--distill-weight`, `--aligned-windows` and `--stop-after-kill` options in `scripts/finetune_v8_channel.py`.
- Teacher data on Beastmode: `data/vvc_teacher/` (1.3 GB, gitignored).
- Kill-check logs, JSON and step-500 checkpoints on Beastmode: `runs/vvc-kill/`.
- JSON copies: `internal/vvc-teacher/`.
