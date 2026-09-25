---
cursor:
  subagentId: "bc-cb0add16-30e3-5488-8e4d-95c8110feee2"
---

# Fidelity plan: beat the incumbents, not AC2K

**Headline.** At 2.2 kHz the model to beat on the channel is the **V8 release checkpoint (`v8-hf3k-face-gan.pt`)**. On the shared 64-clip protocol:

| | V8 + `mpp12` PSNR | V8 + `mpp12` LPIPS |
|---|---:|---:|
| V8 | **20.48 ± 0.52 dB** | **0.255** |
| AC2K | 19.62 dB | 0.462 |

Measured pairwise over the same clips:

- **LPIPS:** V8 is clearly better, by 0.207 ± 0.023.
- **PSNR:** V8's lead over AC2K is 0.86 ± 0.50 dB, only about 1.7 SE. That is suggestive, not established.
- **Clean:** AC2K is ahead by 0.87 ± 0.06 dB, bought with 1.5× the airtime.

No attempt before this plan was compared against V8.

**Status (2026-09-25):**

- **Steps E0 and E1 are done.** The shared scorer is in [PR #28](https://github.com/plaingca/AETV/pull/28).
- **Receiver confidence scaling:** the best setting found gives V8 **+0.107 ± 0.016 dB** on `mpp12`, below the 0.2 dB ship bar. It does nothing for V7 or AC16. No receiver setting ships.
- **E2 (V8 fine-tune with real modem rows) failed its kill check.** At step 500 on the pool, `mpp12` was up **+0.100 ± 0.048 dB**, against a required +0.15. Scored once on eval for the record: **+0.066 ± 0.028 dB** `mpp12`, clean −0.039 ± 0.006, clean LPIPS +0.004 ± 0.001. No PR. See E2 results.
- **Next:** E3 (ROI fidelity) is now the gating experiment for 2.2 kHz. The E2 checkpoint's face-region `mpp12` gain (+0.20 ± 0.08) is the one signal worth following.

---

## 1. Shared baseline table (E0 scorer, measured 2026-09-25)

**Protocol.** Scored by `scripts/eval_shared.py` on branch `cursor/shared-scorer-eee2` ([PR #28](https://github.com/plaingca/AETV/pull/28)). Clips are the first 64 of the seed-2026 shuffle of `data/openvid_aetv_cache/mode_ac6_192x108_12f` (192×108, 12 frames).

- **Latent AWGN:** unit-RMS `impair_wire` noise, confidence `snr/(snr+1)`, seeded per clip.
- **Modem:** each model's own modem, `drift_track=off`. `mpp12` fade seed is `2026 + clip·G + gop`, where G is GOPs per clip.
- **Error bars:** ± is the standard error over the 64 clips.
- **Face-region PSNR:** measured on YuNet boxes from the source frames (threshold 0.72, detected at 4× upscale). 38 of the 64 clips contain a face.

Raw JSON: `internal/fidelity-step2/eval64.json`. The first-pass harness numbers (`internal/fidelity-baseline/`) reproduce to 0.01 dB.

| Band | Model | Contract | Clean | AWGN 15 | AWGN 6 | Modem clean | **mpp12** | SSIM clean / mpp12 | LPIPS clean / mpp12 | Face PSNR clean / mpp12 |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| 2.2 kHz | **V8 face-GAN (release)** | 192×108, 6 fps, 2,816/s | 23.92 ± 0.48 | 23.66 ± 0.47 | 22.40 ± 0.44 | 23.35 ± 0.46 | **20.48 ± 0.52** | 0.938 / 0.808 | **0.210 / 0.255** | 21.60 / 18.57 |
| 2.2 kHz | V8, receiver gain 1.5 (E1) | same | 23.92 | 23.66 | 22.40 | 23.35 | 20.59 ± 0.52 | 0.938 / 0.811 | 0.210 / 0.250 | 21.60 / 18.61 |
| 2.2 kHz | AC2K PSNR fine-tune | 4-frame GOP per RF second\* | **24.80 ± 0.51** | **24.37 ± 0.49** | 22.31 ± 0.40 | 23.74 ± 0.45 | 19.62 ± 0.36 | **0.946** / 0.783 | 0.282 / 0.462 | **22.49** / 17.81 |
| 2.2 kHz | AC2K + multipath-fill receiver | same as AC2K\* | 24.82 | 24.39 | 22.35 | — | 19.88 | 0.946 / 0.794 | 0.283 / — | — |
| 2.2 kHz | AC2K-v2 fidelity | same as AC2K\* | 24.26 ± 0.51 | 23.91 ± 0.49 | 22.02 ± 0.41 | 23.33 ± 0.45 | 19.43 ± 0.36 | 0.940 / 0.778 | 0.296 / 0.464 | 22.01 / 17.58 |
| 2.2 kHz | AC6 (step 24k) | 192×108, 6 fps, 2,816/s | 21.58 ± 0.45 | 21.14 ± 0.42 | 19.02 ± 0.31 | 20.06 ± 0.35 | 16.78 ± 0.32 | 0.894 / 0.683 | 0.476 / 0.717 | 19.72 / 16.26 |
| 8 kHz | **V7 rxfix (release)** | 256×144, 12 fps, 10,112/s† | 25.54 ± 0.48 | 25.41 ± 0.48 | 24.60 ± 0.47 | 25.73 ± 0.48 | **22.40 ± 0.63** (1/64 demod fail) | 0.958 / 0.865 | 0.178 / 0.262 | 22.68 / 19.85 |
| 16 kHz | **AC16 (step 70k)** | 256×144, 10 fps, 19,200/s† | 25.93 ± 0.42 | 25.80 ± 0.41 | 25.00 ± 0.40 | 25.59 ± 0.42 | 22.19 ± 0.57 (4/64 demod fail) | 0.958 / 0.853 | 0.168 / 0.302 | 24.24 / 20.43 |

The multipath-fill row is from `multipath-codec.md`, which predates the scorer. Face-PSNR standard errors are about 0.5 dB (n = 38). The earlier 32-clip V8 perceptual row (19.99 clean) is not rescored here.

**Paired deltas on the same clips (± paired SE):**

| Comparison | mpp12 | Clean | LPIPS mpp12 |
|---|---:|---:|---:|
| AC2K PSNR − V8 | −0.86 ± 0.50 | +0.87 ± 0.06 | +0.207 ± 0.023 |
| AC2K-v2 fidelity − V8 | −1.05 ± 0.49 | +0.34 ± 0.05 | +0.210 ± 0.022 |
| AC6 − V8 | −3.71 ± 0.26 | −2.34 ± 0.10 | +0.463 ± 0.013 |
| AC16 − V7 | −0.21 ± 0.17 | +0.39 ± 0.23 | +0.040 ± 0.009 |
| V8 gain 1.5 − V8 | **+0.107 ± 0.016** | 0.000 | −0.005 ± 0.002 |

The standard errors across models are large (0.4–0.5 dB) because per-clip `mpp12` scores swing with the fade draw. Only paired deltas resolve a few tenths of a dB.

\* **AC2K contract.** AC2K sends 2,816 reals per 4-frame GOP, and the V8 modem carries one GOP per second. So AC2K is 4 fps, or 1.5× V8's airtime per second of 6 fps video.

† **V7 and AC16 are cross-contract.** Their clips are bilinear-upscaled from 192×108, and V7 plays 6 fps frames as 12 fps. These rows rank within a band only.

Frames:

- `internal/fidelity-baseline/frames/`: incumbents side by side.
- `media/fidelity-step2/`: V8 at gain 1.0 vs 1.5 on `mpp12`, clips val00/24/48, frames 2/5/9. There is no visible difference at +0.1 dB.

### What exists and what is missing

| Item | Status |
|---|---|
| Shared scorer in the repo (per-clip JSON, paired SE, face-region PSNR, pool split) | **Done.** [PR #28](https://github.com/plaingca/AETV/pull/28) |
| V8 / AC2K / AC6 / V7 / AC16 on the shared protocol with error bars | **Done** |
| Receiver confidence scaling for V8 / V7 / AC16 | **Done (E1).** No ship |
| AC6 90k production checkpoint (`models/ac6_1m_production/ac6_best.pt`) | **Missing.** State dict predates the current `AC6Codec` decoder layout |
| Native 256×144 held-out cache for V7 / AC16 | **Missing** |
| Held-out set outside OpenVid-1M | **Missing.** V8 and AC2K trained on streamed OpenVid-1M |
| `ota40m`, `mpg12`, `mpp6` rows | **Missing** |
| Multipath-fill row rescored by the E0 scorer | **Missing** (it lives on `cursor/multipath-common-8125`) |
| `v8-hf3k-openvid-best`, `v7-flex8k`, `v7-severe*` weights | Not on this machine (HF hub only) |
| V8 perceptual: 19.99 clean vs 23.35 in `docs/v8-hf3k.md` | **Unexplained.** Do not warm-start from it |

---

## 2. Why the recent attempts lost

| Attempt | Init | Steps | Training channel / objective | Contract | Result | Incumbent reference |
|---|---|---:|---|---|---|---|
| RVDJSCC v1 | scratch | 8k | clean then AWGN | 2.2 kHz, 192×108 | 16.62 clean | AC2K-v2's own curve at 8k: 21.3‡ |
| RVDJSCC v2 (117.7M params) | scratch | 2.6k | clean only. AWGN stage cut 18.69 → ~16.5, so it was dropped | 2.2 kHz | 19.80 clean | AC2K curve at 2.6k: 19.6‡. **On curve, just short** |
| Narrowband joint AE | scratch | 6k / 5k | clean + AWGN | 2.2 / 8 kHz | 19.27 / 19.94 (16 clips) | AC2K curve at 5k: 20.4‡. 3.6× coordinates gave +0.67 dB |
| Codebook JSCC | scratch | 25k | fade surrogate + VQ regularizers | 2.2 kHz | 20.45 clean, 18.13 `mpp12` | AC2K curve at 25k: 22.5‡. V8 `mpp12`: 20.48 |
| DeepStream HF | scratch | 11.5k | fade surrogate + real `mpp12` rows | **32×18, 2 fps** | 17.44 `mpp12` at 32×18; unintelligible | A perfect 32×18 upscaled caps at ~21.57 dB (6× bilinear), below V8 clean |
| AC2K PSNR fine-tune | warm (AC2K-v2 fidelity) | 2k | MSE + AWGN on 40% of steps, **no fading** | AC2K | +0.54 clean; `mpp12` unchanged at 19.62 | Selected on the first 16 of the **same 64 eval clips** |
| Slack detail | frozen AC2K | 0 | dither in 8-bit bins | AC2K | +0.91 clean; **0** under any noise | Unusable on air |
| Fidelity pathway (side code) | frozen AC2K | 0 | punctured quiet slots | AC2K | 19.69 `mpp12` | Lost to 19.88 |
| Multipath-fill receiver | frozen AC2K | receiver only | confidence ×1.5–1.75 + residual head | AC2K | **+0.26 → 19.88 `mpp12`** | The only channel win, and it was warm and cheap |

‡ Training-batch PSNR from `models/ac2k-v2_1m_training_log.jsonl`, averaged ±1.5k steps. That log covers 192k steps in 10.8 h and reaches 25.2 by 190k.

**Diagnosis**

1. **Undertrained, not wrong.** Each from-scratch run stopped at 2.6k–25k steps, at or below AC2K's own curve at the same step. The incumbents had 70k–200k steps (V8: 116.5k OpenVid-1M steps + 12k detail-face + 1k face/GAN).
2. **Wrong bar.** Everything was measured against AC2K, a 4 fps, 1.5×-airtime contract that has never seen fading. V8 matches or beats every attempt on `mpp12` (20.48) and is far ahead on LPIPS (0.255 against 0.46–0.72).
3. **Wrong objective for the channel.** The PSNR fine-tune trained on AWGN only. AC2K loses 5.18 dB from clean to `mpp12`, while V8, trained with fading, loses 3.44 dB. Gains measured clean or on latent AWGN (+0.54, +0.91) never reached `mpp12`.
4. **Contract changes.** DeepStream's 32×18 / 2 fps contract caps fidelity below V8's clean picture before any training.
5. **Eval mismatches.**
   - Checkpoint selection used eval clips.
   - Latent-AWGN tables were read as channel scores.
   - 8- or 16-clip numbers were set next to 64-clip numbers.
   - No error bars. Cross-model SE is about 0.5 dB, so only paired deltas can rank close models.

---

## 3. Experiments (in order)

Shared rules for every experiment:

- Select checkpoints and settings only on **training-pool clips** (`--split pool`, fade seed base 7300). Score the 64-clip val once, at the end.
- The kill check runs first. If it fails, stop and write down the number.
- Compute estimates are for one RTX 4090:
  - V8 stage-2 training: about 2.6 steps/s at batch 8.
  - AC2K training: about 4.9 steps/s at batch 4.
  - Shared scorer: about 3 min for all six models on 64 clips.

| # | Experiment | Warm start | Kill check (cheap, before full run) | Full run | **Bar to beat (64-clip val, paired)** | Status |
|---|---|---|---|---|---|---|
| **E0** | Shared scorer: per-clip JSON, paired SE, face-region PSNR, pool split | — | Reproduce AC2K 24.80 / 19.62 and V8 23.92 / 20.48 within ±0.02 | ~3 min | n/a | **Done.** Reproduced exactly; [PR #28](https://github.com/plaingca/AETV/pull/28) |
| **E1** | Receiver confidence gain (scale weights only when they spread > 0.05 across the GOP) for V8, V7 and AC16 | release checkpoints | Gains 0.5–3.0 on 24 pool clips. Must gain ≥ 0.10 dB `mpp12` | ~2 min | +0.2 dB and > 2× paired SE | **Done, no ship.** See E1 results below |
| **E2** | **V8 channel-matched fine-tune.** Keep the face-GAN loss stack (LPIPS 0.18, DWT, face ROI, clean anchor) and add real V8 + `mpp12` rows in the forward pass on 1 of 4 steps, with lr 1e-6 → 3e-6 | `v8-hf3k-face-gan.pt` | 500-step probe on 16 pool clips: `mpp12` ≥ +0.15 dB paired, clean ≥ −0.10, `mpp12` LPIPS not worse | 3–5k steps, ~1–1.5 h | **`mpp12` ≥ 20.68 (paired ≥ +0.2 over the release, > 2 SE), clean ≥ 23.72, LPIPS clean ≤ 0.215 / `mpp12` ≤ 0.255** | **Killed at step 500** (pool +0.100 ± 0.048). See E2 results |
| **E3** | **V8 ROI fidelity.** Raise the face-region L1/MSE weight and cut `face_adv_weight` from 0.01 to about 0.003 | best of E2, else V8 release | 500 steps: face PSNR on `mpp12` ≥ +0.3 dB paired (n = 38), full-frame LPIPS within +0.01 | 2k steps, ~1 h | E2's bar **plus** face PSNR `mpp12` ≥ 18.87 and a visual A/B pass | |
| **E4** | **AC2K fading-aware fine-tune** (4 fps contract only), with the two-path fade surrogate + real `mpp12` rows | `ac2k-psnr-2.2khz-best.pt` | 1k steps on 8 pool clips: `mpp12` ≥ +0.40 dB, clean drop ≤ 0.30 | 5k steps, ~1 h | **`mpp12` ≥ 20.08** at clean ≥ 24.5. Does not replace V8 unless the product accepts 4 fps | |
| **E5** | **AC6 from AC2K** at the V8 contract, long schedule, fading from the start. Run only if E4 passes | AC2K-v2 fidelity via `load_warmstart_from_ac2k` | At 10k steps on 8 pool clips: clean ≥ 22.5 and `mpp12` ≥ 18.5. Otherwise stop | 100k steps, ~6–8 h | **V8: `mpp12` ≥ 20.68, LPIPS `mpp12` ≤ 0.255** | gated |
| **E6** | **V7 channel-matched fine-tune**, the E2 recipe | `v8-flex8k-ota-rxfix.pt` | 250 steps on 8 pool clips: `mpp12` ≥ +0.15 dB, clean ≥ −0.10 | 2k steps, ~2 h | **`mpp12` ≥ 22.60, clean ≥ 25.34, LPIPS `mpp12` ≤ 0.262** | |
| **E7** | **AC16: fix acquisition, then fine-tune for fading.** 4 of 64 `mpp12` GOPs fail to demodulate (V7: 1 of 64) | `ac16-best-inference.pt` | Demod failures ≤ 1/64 with no codec change. Then a 1k-step probe: `mpp12` ≥ +0.2 dB | 10k steps, ~2–3 h | **Must beat V7 on `mpp12`** (now −0.21 ± 0.17) | gated |

### E1 results (receiver confidence gain, no training)

| Model | Pool (24 clips, seeds 7300+): best gain → paired Δ `mpp12` | Chosen | Eval 64: paired Δ `mpp12` | LPIPS `mpp12` Δ | Clean / AWGN / modem clean | Verdict |
|---|---|---:|---:|---:|---|---|
| V8 | 1.5 → **+0.087 ± 0.026**. Also 1.25: +0.052; 1.75: +0.082; 2.0: +0.073; 3.0: +0.009; 0.5–0.9: negative | **1.5** | **+0.107 ± 0.016** (20.48 → 20.59) | −0.005 ± 0.002 | unchanged (weights flat) | Below the plan's 0.10 pool check and the 0.2 dB ship bar. **No PR** |
| V7 | 1.0. Gains > 1 all negative (1.25: −0.045 ± 0.019; 3.0: −0.139); 0.9: +0.007 | 1.0 | 0 | 0 | — | Confidence is not pessimistic. No change |
| AC16 | 1.0. Gains > 1 hurt fast (1.25: −0.20 ± 0.05; 3.0: −1.18 ± 0.15); 0.75: −0.19 | 1.0 | 0 | 0 | — | Confidence is not pessimistic; its losses are acquisition (E7). No change |

**Takeaway.** The ×1.5–1.75 receiver trick that gave AC2K +0.24 dB is worth only about +0.1 dB on V8 and nothing on V7 or AC16. The V8 decoder already uses its modem confidence close to optimally, so the remaining V8 `mpp12` gap must come from training (E2/E3), not from receiver tuning. Gain 1.5 may be stacked on an E2 candidate, but the candidate is scored against the release at gain 1.0.

### E2 results (V8 fine-tune with real modem rows)

**Recipe.** `scripts/finetune_v8_channel.py` on branch `cursor/v8-channel-ft-eee2` (no PR).

- **Warm start:** `models/v8-hf3k-face-gan.pt`, which is not modified (SHA-256 still `f218376a…`).
- **Batch:** 8 rows per step.
  - 6 rows go through the release's differentiable waveform channel (0–16 dB, p_fading 0.5, measured-path 0.4).
  - 2 rows go through the real V8 OFDM modem plus `emulate`, drawn from `mpp12` ×3, `mpp6`, `awgn12` and `clean`, with training-only fade seeds from 1,000,000. The channel error is passed straight through to the encoder.
- **Loss:** the release generator objective and weights (pixel, DWT, gradient, temporal, VGG, region/detail/contrast, face-crop VGG, consistency, clean anchor). **The face critic and PatchGAN are not trained.**
- **Schedule:** lr 3e-6 with a 50-step warmup and cosine decay. About 1.7 steps/s on the 4090.
- **Data:** training pool minus 24 selection clips (1,325 clips).
- **Selection:** 24 pool clips, fade seed base 7300. The 64 eval clips were not used for selection.
- **Checkpoint:** `runs/v8-channel-ft/best.pt` (step 500), a new path.

**Kill check (pool, 24 clips, paired against the release):**

| Step | `mpp12` Δ | Clean Δ | LPIPS clean Δ | LPIPS `mpp12` Δ | Face `mpp12` Δ | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 250 | +0.085 ± 0.039 | −0.004 ± 0.012 | +0.012 ± 0.002 | −0.001 ± 0.003 | +0.14 ± 0.11 | — |
| **500** | **+0.100 ± 0.048** | −0.035 ± 0.013 | +0.006 ± 0.002 | −0.011 ± 0.004 | +0.18 ± 0.13 | **Failed** (needs ≥ +0.15). Stopped |

**Scored once on the 64 eval clips, for the record (not for selection).** Paired against the V8 release at receiver gain 1.0.

| | Clean | Modem clean | AWGN 6 | **mpp12** | LPIPS clean | LPIPS mpp12 | Face PSNR mpp12 (n=38) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 release, gain 1.0 | 23.92 ± 0.48 | 23.35 ± 0.46 | 22.40 ± 0.44 | 20.48 ± 0.52 | 0.210 | 0.255 | 18.57 |
| E2 step 500, gain 1.0 (**fine-tune effect**) | 23.88 (−0.039 ± 0.006) | 23.31 (−0.041 ± 0.007) | 22.39 (−0.007 ± 0.006) | **20.55 (+0.066 ± 0.028)** | 0.213 (+0.004 ± 0.001) | 0.248 (−0.007 ± 0.002) | 18.77 (+0.20 ± 0.08) |
| V8 release, gain 1.5 (receiver only, E1) | 23.92 | 23.35 | 22.40 | 20.59 (+0.107 ± 0.016) | 0.210 | 0.250 | 18.61 |
| E2 step 500, gain 1.5 (fine-tune + receiver, **reported separately**) | 23.88 | 23.31 | 22.39 | 20.65 (+0.167 ± 0.035) | 0.213 | 0.251 | 18.83 |

On the eval set, the fine-tune adds +0.060 ± 0.027 dB over the release when both use gain 1.5.

**Ship rule:** fails. The `mpp12` gain is +0.066 dB, against a bar of +0.2 (20.68 dB). Clean LPIPS also regresses by +0.004 ± 0.001. Even the stacked receiver + fine-tune row (+0.167 dB) is below the bar.

**Frames:** `media/fidelity-step3/v8ft-val{00,24,48}-f{02,05,09}.png` (source | V8 release | fine-tune, `mpp12`, gain 1.0) and `media/fidelity-step3/v8ft-mpp12-side-by-side.mp4` (3 clips, 12 frames at 6 fps, played twice). Release and fine-tune are visually near-identical.

**Reading:**

1. **Real-modem rows give a small, real but slow `mpp12` gain.** +0.10 on the pool and +0.07 on eval at 500 steps. The step-250 to step-500 slope (+0.015 dB per 250 steps) does not project to +0.2 dB inside the planned 3–5k steps.
2. **Dropping the face critic softens clean texture.** That shows up as clean LPIPS +0.004 to +0.012.
3. **The face region moved most.** +0.20 ± 0.08 dB on `mpp12` faces, which is the E3 signal.

Do not rerun E2 as specified. A restart needs a changed hypothesis, for example the face critic restored, or more real rows per batch. It also needs its own kill check.

**Paused** until a kill check says otherwise: RVDJSCC / codebook / DeepStream / joint-AE ports, sub-192×108 contracts, and side codes in slack bits.

A from-scratch model may restart only if, at an equal step count, it sits **≥ 1 dB above AC2K-v2's logged curve** on pool `mpp12` for two consecutive evals. The reference points are 19.6 at 2.6k, 20.4 at 5k, 21.9 at 11.5k and 22.5 at 25k steps.

Remaining budget: E2–E4 + E6 is about 6 GPU-hours. E5 and E7 are gated.

---

## 4. Rules

1. **One table.** Every candidate is scored by `scripts/eval_shared.py` on the 64-clip protocol, in the same run as its incumbent, so the deltas are paired.
2. **No PR unless it beats the band's incumbent under `mpp12`** by ≥ 0.2 dB **and** by more than 2× the paired per-clip SE. It must also hold clean within 0.2 dB and not regress LPIPS by more than 0.005. Incumbents:
   - 2.2 kHz at 6 fps: V8 face-GAN at 20.48 (receiver gain 1.0)
   - 2.2 kHz at 4 fps: AC2K + fill at 19.88
   - 8 kHz: V7 rxfix at 22.40
   - 16 kHz: AC16 at 22.19, and it must also beat V7
3. **Same contract.** Same resolution, fps and coordinates per RF second as the incumbent.
4. **No selection on val.** Pool clips and pool fade seeds only (`--split pool`). The 64-clip val is scored once per candidate.
5. **Every result ships with side-by-side frames**: source | incumbent | candidate, clean and `mpp12`, the same clips (val00/24/48) and frame index, plus a short MP4.
6. **Report the kill-check number** even when the experiment stops there.

---

## Files

- This plan: `docs/fidelity-plan.md`. A copy is on branch `cursor/fidelity-plan-doc`.
- Scorer: `aetv/shared_eval.py`, `scripts/eval_shared.py`, `tests/test_shared_eval.py` on `cursor/shared-scorer-eee2` ([PR #28](https://github.com/plaingca/AETV/pull/28)).
- E0/E1 JSON: `internal/fidelity-step2/` (`eval64.json`, `pool-calibration.json`, `pool-calibration-low.json`).
- First-pass harness, JSON and incumbent frames: `internal/fidelity-baseline/`.
- E1 frames: `media/fidelity-step2/v8-gain-val{00,24,48}-f{02,05,09}.png`.
- E2 fine-tune script: `scripts/finetune_v8_channel.py` on `cursor/v8-channel-ft-eee2` (no PR). Checkpoint: `runs/v8-channel-ft/best.pt` (step 500, gitignored).
- E2 JSON: `internal/fidelity-step3/` (`eval64-v8ft.json`, `kill_check.json`, `history.json`, `train_log.jsonl`, `select_step0.json`).
- E2 frames: `media/fidelity-step3/v8ft-val{00,24,48}-f{02,05,09}.png` and `media/fidelity-step3/v8ft-mpp12-side-by-side.mp4`.
