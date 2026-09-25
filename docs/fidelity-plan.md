---
cursor:
  subagentId: "bc-cb0add16-30e3-5488-8e4d-95c8110feee2"
---

# Fidelity plan: beat the incumbents, not AC2K

**Headline.** At 2.2 kHz the model to beat on the channel is the **V8 release checkpoint (`v8-hf3k-face-gan.pt`)**, not AC2K. On the shared 64-clip protocol, V8 scores **20.48 dB on V8 modem + `mpp12`** with LPIPS 0.255. AC2K scores 19.62 dB with LPIPS 0.462, and the multipath-fill receiver scores 19.88 dB. No recent attempt was ever compared against V8. The cheapest wins left are warm-started, channel-matched fine-tunes of V8, V7 and AC16, plus receiver calibration. From-scratch ports are paused.

Planning only. Nothing was trained, no checkpoints or branches were touched, and there is no PR. The table below comes from about 6 minutes of evaluation on the idle RTX 4090.

---

## 1. Shared baseline table (measured 2026-09-25)

**Protocol.** First 64 clips of the seed-2026 shuffle of `data/openvid_aetv_cache/mode_ac6_192x108_12f` (192×108, 12 frames per clip).

- **Latent AWGN:** unit-RMS `impair_wire` noise, confidence `snr/(snr+1)`.
- **Modem:** each model's own modem (V8 = band W, V7 = band U, AC16 = band A). `drift_track=off`. `mpp12` uses fade seed `2026 + clip·G + gop`, where G is the number of GOPs per clip.
- **Metrics:** PSNR is `10·log10(1/MSE)`. SSIM is the project's global per-frame SSIM. LPIPS is AlexNet.

Script and JSON: `internal/fidelity-baseline/baseline_eval.py` and `baseline64.json`. The AC2K row reproduces the published numbers exactly (24.797 clean, 19.620 `mpp12`), so the harness matches the earlier reports.

| Band | Model | Contract (airtime) | Clean | AWGN 15 | AWGN 6 | Modem clean | **mpp12** | SSIM clean / mpp12 | LPIPS clean / mpp12 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2.2 kHz | **V8 face-GAN (release)** | 192×108, 6 fps, 2,816/s | 23.92 | 23.66 | 22.39 | 23.35 | **20.48** | 0.938 / **0.808** | **0.210 / 0.255** |
| 2.2 kHz | AC2K PSNR fine-tune | 192×108, 4-frame GOP per RF second\* | **24.80** | **24.37** | 22.31 | 23.74 | 19.62 | **0.946** / 0.783 | 0.282 / 0.462 |
| 2.2 kHz | AC2K + multipath-fill receiver | same as AC2K\* | 24.82 | 24.39 | 22.35 | — | 19.88 | 0.946 / 0.794 | 0.283 / — |
| 2.2 kHz | AC2K-v2 fidelity | same as AC2K\* | 24.26 | 23.90 | 22.02 | 23.33 | 19.43 | 0.940 / 0.778 | 0.296 / 0.464 |
| 2.2 kHz | AC6 (`ac6-best-inference`, step 24k) | 192×108, 6 fps, 2,816/s | 21.58 | 21.13 | 19.01 | 20.06 | 16.78 | 0.894 / 0.683 | 0.476 / 0.717 |
| 2.2 kHz | V8 perceptual (former default) | 192×108, 6 fps | 19.99 | 19.91 | 19.46 | 19.23 | 17.58 | 0.856 / 0.725 | 0.300 / 0.384 |
| 8 kHz | **V7 rxfix (release)** | 256×144, 12 fps, 10,112/s† | 25.54 | 25.41 | 24.63 | 25.73 | **22.40** (1/64 demod fail) | 0.958 / 0.865 | 0.178 / 0.262 |
| 16 kHz | **AC16 (step 70k)** | 256×144, 10 fps, 19,200/s† | 25.93 | 25.80 | 24.95 | 25.59 | 22.19 (4/64 demod fail) | 0.958 / 0.853 | 0.168 / 0.302 |

The multipath-fill row is from `multipath-codec.md`. Every other row was measured today.

\* **AC2K contract.** AC2K puts 2,816 reals on each 4-frame GOP, and the V8 modem carries one GOP per second. So AC2K runs at 4 fps, or spends 1.5× V8's airtime per second of 6 fps video (3 GOPs per 12-frame clip against V8's 2). Its +0.9 dB clean edge over V8 is bought with that extra airtime.

† **V7 and AC16 rows are cross-contract.** Their clips are bilinear-upscaled from 192×108, and V7 plays the 6 fps frames as 12 fps. That is fine for ranking within a band, but not as a native score.

Side-by-side frames (source | clean | `mpp12`, frame 5, clips val00/24/48) are in `internal/fidelity-baseline/frames/`. On val24, V8 is the sharpest but has an invented-looking face, AC2K is smooth and blurry, and AC6 falls apart under `mpp12`.

### What exists and what is missing

| Item | Status |
|---|---|
| AC2K family at 2.2 kHz, clean / AWGN / `mpp12` | Exists (reports) and reproduced here |
| **V8 release on the shared protocol** | **Missing until today.** Now measured |
| AC6 on the shared protocol | Measured here (24k checkpoint) |
| AC6 90k production checkpoint (`models/ac6_1m_production/ac6_best.pt`) | **Missing.** State dict predates the current `AC6Codec` decoder layout and does not load |
| V7 / AC16 on the shared clips through their own modem + `mpp12` | Measured here, cross-contract |
| Native 256×144 held-out cache for V7 / AC16 | **Missing** |
| Per-clip paired standard errors | **Missing.** Add in E0 |
| Held-out set outside OpenVid-1M | **Missing.** V8 and AC2K trained on streamed OpenVid-1M, so the 64 cache clips may have been seen |
| V8 / V7 / AC16 with a confidence-scaled receiver | **Missing.** See E1 |
| `ota40m`, `mpg12`, `mpp6` rows | **Missing** |
| `v8-hf3k-openvid-best`, `v7-flex8k`, `v7-severe*` weights | Not on this machine (HF hub only) |
| V8 perceptual: 19.99 clean here vs 23.35 in `docs/v8-hf3k.md` (different 32 clips) | **Unexplained.** Do not warm-start from it until resolved |

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
| Multipath-fill receiver | frozen AC2K | receiver only | confidence ×1.75 + residual head | AC2K | **+0.26 → 19.88 `mpp12`** | The only channel win, and it was warm and cheap |

‡ Training-batch PSNR from `models/ac2k-v2_1m_training_log.jsonl`, averaged ±1.5k steps. That log covers 192k steps in 10.8 h and reaches 25.2 by 190k.

**Diagnosis**

1. **Undertrained, not wrong.** Each from-scratch run stopped at 2.6k–25k steps and landed at or below AC2K's own curve at the same step. The incumbents had 70k–200k steps (V8: 116.5k OpenVid-1M steps + 12k detail-face + 1k face/GAN). A from-scratch port cannot win on one 4090 in a few thousand steps, and the curves show no sign of a better slope.
2. **Wrong bar.** Everything was measured against AC2K. AC2K is a 4 fps, 1.5×-airtime contract and has never seen fading, while V8 already clears every one of those attempts on `mpp12` (20.48) and LPIPS (0.255 against 0.46–0.72).
3. **Wrong objective for the channel.** The PSNR fine-tune trained on AWGN only. AC2K loses 5.18 dB from clean to `mpp12`, while V8, trained with p_fading 0.5 and measured-path draws, loses 3.44 dB. Gains measured on clean or latent AWGN (+0.54, +0.91) never reached `mpp12`.
4. **Contract changes.** DeepStream's 32×18 / 2 fps contract caps fidelity below V8's clean picture before any training. Its 17.44 dB is against its own downscaled ground truth.
5. **Eval mismatches.** Checkpoint selection used the eval clips (AC2K PSNR fine-tune). Latent-AWGN tables were read as channel scores. V7/AC16 were never scored on shared clips. Many reports put 8- or 16-clip numbers next to 64-clip numbers.

---

## 3. Experiments (in order)

Shared rules for every experiment:

- Select checkpoints only on **training-holdout clips** (for example, 8–16 clips with fade seed base 7300). Score the 64-clip val once, at the end.
- The kill check runs first. If it fails, stop and write down the number.
- Compute estimates are for one RTX 4090:
  - V8 stage-2 training runs at about 2.6 steps/s at batch 8 (`runs/v8-hf3k-openvid1m-full-20260823-r2/train.log`).
  - AC2K runs at about 4.9 steps/s at batch 4 (training log).
  - A real V8 modem + `mpp12` pass costs about 0.1 s per GOP on CPU.
  - A full shared table takes about 4 minutes.

| # | Experiment | Warm start | Kill check (cheap, before full run) | Full run | **Bar to beat (64-clip val)** |
|---|---|---|---|---|---|
| **E0** | Lock the protocol. Land the shared scorer in the repo with per-clip JSON, paired bootstrap SE, frame sheets, a face-ROI PSNR/LPIPS column (YuNet in `data/teachers/`), `ota40m` / `mpg12` / `mpp6` rows, and a small non-OpenVid set (`val_highres_clips`, CelebV-HQ) | — | Reproduce this table within ±0.02 dB (AC2K 24.80 / 19.62, V8 23.92 / 20.48) | ~15 min | n/a (it is the bar) |
| **E1** | Receiver confidence calibration for V8, V7 and AC16. This is the step that gave AC2K +0.24 dB: scale weights only when they vary across the GOP | release checkpoints, no training | Sweep scale {1, 1.25, 1.5, 1.75, 2} on 12 training clips. Must gain ≥ 0.10 dB `mpp12` with the AWGN rows within 0.02 | ~20 min per band | V8 **≥ 20.68**; V7 **≥ 22.60**; AC16 **≥ 22.39** |
| **E2** | **V8 channel-matched fine-tune.** Keep the face-GAN loss stack (LPIPS 0.18, DWT, face ROI, clean anchor) and add real V8 + `mpp12` rows in the forward pass on 1 of 4 steps (the DeepStream trick), with lr 1e-6 → 3e-6 | `v8-hf3k-face-gan.pt` | 500-step probe on 16 holdout clips: `mpp12` ≥ +0.15 dB, clean ≥ −0.10, `mpp12` LPIPS not worse | 3–5k steps, ~1–1.5 h | **`mpp12` ≥ 20.68 and clean ≥ 23.72 and LPIPS clean ≤ 0.215 / `mpp12` ≤ 0.255** |
| **E3** | **V8 ROI fidelity.** The frame sheets show an invented face. Raise the face-region L1/MSE weight and cut `face_adv_weight` from 0.01 to about 0.003, trading invented detail for correct detail | best of E2, else V8 release | 500 steps: face-ROI PSNR ≥ +0.3 dB with full-frame LPIPS within +0.01 | 2k steps, ~1 h | E2's bar, **plus** face-ROI PSNR ≥ incumbent + 0.3 and a visual A/B pass |
| **E4** | **AC2K fading-aware fine-tune** (4 fps contract only). Use the two-path fade surrogate + real `mpp12` rows, and keep the multipath-fill receiver | `ac2k-psnr-2.2khz-best.pt` | 1k steps on 8 holdout clips: `mpp12` ≥ +0.40 dB over the AC2K holdout score, clean drop ≤ 0.30 | 5k steps, ~1 h | **`mpp12` ≥ 20.08** (fill + 0.2) at clean ≥ 24.5. This does not replace V8 unless the product accepts 4 fps |
| **E5** | **AC6 from AC2K** at the true V8 contract (6 fps, 2,816/s), long schedule with fading on from the start. `load_warmstart_from_ac2k` already exists. Run only if E4 passes | AC2K-v2 fidelity via `load_warmstart_from_ac2k` | At 10k steps on 8 holdout clips: clean ≥ 22.5 and `mpp12` ≥ 18.5 (AC6 at 24k is 21.58 / 16.78). Otherwise stop | 100k steps, ~6–8 h | **V8: `mpp12` ≥ 20.68, LPIPS `mpp12` ≤ 0.255** |
| **E6** | **V7 channel-matched fine-tune**, the same recipe as E2. The current release has only 2 × 250-step adaptations | `v8-flex8k-ota-rxfix.pt` | 250 steps on 8 holdout clips: `mpp12` ≥ +0.15 dB, clean ≥ −0.10 | 2k steps, ~2 h (215 MB model, 256×144 × 12 frames) | **`mpp12` ≥ 22.60, clean ≥ 25.34, LPIPS `mpp12` ≤ 0.262** |
| **E7** | **AC16: fix acquisition, then fine-tune for fading.** 4 of 64 `mpp12` GOPs failed to demodulate (V7: 1 of 64), so this is a receiver problem first. Re-score excluding failures, fix, then fine-tune from step 70k with fading + real modem rows | `ac16-best-inference.pt` | Demod failures ≤ 1/64 with no codec change. Then a 1k-step probe: `mpp12` ≥ +0.2 dB | 10k steps, ~2–3 h | **Must beat V7 at `mpp12` (≥ 22.60 after E6)**; otherwise 16 kHz does not justify its bandwidth |

**Paused** until a kill check says otherwise: RVDJSCC / codebook / DeepStream / joint-AE ports, sub-192×108 contracts, and side codes in slack bits.

A from-scratch model may restart only if, at an equal step count, it sits **≥ 1 dB above AC2K-v2's logged curve** on held-out `mpp12` for two consecutive evals. The reference points are 19.6 at 2.6k, 20.4 at 5k, 21.9 at 11.5k and 22.5 at 25k steps.

Total for E0–E4 + E6: about 7 GPU-hours. E5 and E7 are gated.

---

## 4. Rules

1. **One table.** Every candidate is scored by the E0 scorer on the 64-clip protocol and its row is added next to the incumbents in this table, with clean, AWGN 15/6, modem clean, `mpp12`, SSIM, LPIPS and face-ROI.
2. **No PR unless it beats the band's incumbent under `mpp12`** by ≥ 0.2 dB and by more than 2× the paired per-clip SE. It must also stay within 0.2 dB clean and not regress LPIPS by more than 0.005. Incumbents:
   - 2.2 kHz at 6 fps: V8 face-GAN (20.48)
   - 2.2 kHz at 4 fps: AC2K + fill (19.88)
   - 8 kHz: V7 rxfix (22.40)
   - 16 kHz: AC16 (22.19), and it must also beat V7
3. **Same contract.** Same resolution, fps and coordinates per RF second as the incumbent. A contract change needs its own incumbent row at that contract before it can claim a win.
4. **No selection on val.** Holdout clips and holdout fade seeds only. The 64-clip val is scored once per candidate.
5. **Every result ships with side-by-side frames**: source | incumbent | candidate, clean and `mpp12`, the same 3 clips (val00/24/48) and the same frame index, plus a short MP4. A number without frames is not a result.
6. **Report the kill-check number** even when the experiment stops there.

---

## Files

- This plan: `docs/fidelity-plan.md`
- Scorer, raw JSON and frame sheets: `internal/fidelity-baseline/` (`baseline_eval.py`, `baseline64.json`, `frames/*.png`)
