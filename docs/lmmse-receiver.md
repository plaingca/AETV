---
cursor:
  subagentId: "bc-f935febb-c8a4-50c1-90e2-9edbbfb8d9e7"
---

# LMMSE receiver: SSTVAE PR 58, adapted (2026-09-27)

**Result: the ship rule is met, and the PR is [#35](https://github.com/plaingca/AETV/pull/35) against `main` (v0.1.23).** On `mpp12`, V8 gains **+0.274 ± 0.029 dB** PSNR and V9 **+0.228 ± 0.022 dB**, paired against the v0.1.23 receiver on the 64 eval clips. Nothing got worse in simulation, end to end, or over the air. Weak-signal beacon verification improved.

This is receive-side only, with no on-air format change. With `modem.CHANNEL_ESTIMATOR = "pilot"` and `WINDOW_PLACEMENT = False`, the receiver is bit-exact with v0.1.23. That was checked on 30 batch and streaming cases across V8, V9 and V7, so the paired baselines below are the release receiver.

## What was built

- **2-D LMMSE channel estimate** (`aetv/channel_estimate.py`), used in the `demodulate_gop_stream` and tracked-GOP payload paths.
  - **Across carriers:** pilots are projected onto the measured delay support.
  - **Across time:** a Wiener interpolator with a Gaussian (F.1487) Doppler model gives the channel at each data symbol.
  - **Last frame:** each GOP's last frame is no longer equalized with its own, up to 100 ms stale, pilot. The batch path also uses up to two pilots from each neighbouring GOP.
  - **Weights:** unchanged, still H²/(H²+N).
- **Statistics across GOPs, not per GOP.** Delay profile and Doppler correlation are normalized per GOP and accumulated with a 16-GOP forgetting horizon.
  - `demodulate_gop_stream` accumulates over the whole transmission plus the averaged preamble.
  - `StreamingDemodulator` keeps a running estimate, updated only from GOPs it accepts, and resets it on loss of lock.
- **Fallback** per GOP to the v0.1.23 estimator in three cases:
  - the profile's peak-to-median contrast is below 4 (mostly noise);
  - pilot SNR is below −6 dB;
  - the pilots are noiseless.
- **Window placement.** The FFT window moves only when a path above max(peak − 15 dB, 4 × median) lies outside the cyclic prefix, and only far enough to bring it inside with a 2-sample margin. It never centres the paths.
  - The tracked path decides from earlier GOPs. A timing jump inside a GOP stays visible to the boundary tracker, which repairs it; moving the window instead would hide the jump.
  - Moves are limited to between `ncp − 8` samples earlier and 8 samples later. The window then stays inside the one-GOP slice, so no extra look-behind is needed.
- **Simulator.** `mpp12` is unchanged and remains the bar. `mpp12-gauss` (F.1487 Gaussian taps with FFT resampling) is a separate research profile, reported separately and kept out of the operator TX list.
- **Credit.** `NOTICE` credits SSTVAE PR 58 (Artistic-2.0), next to the existing SSTVAE attribution.
- **Tests:** 20 new unit tests. The full suite passes: 527 passed, 2 skipped.
- **Not changed:** AC16 (band A, not validated) and blind late-join acquisition, whose beacon search keeps the v0.1.23 estimator.

**One departure from SSTVAE.** SSTVAE gates the delay support at 2× the median floor. With AETV's 8-pilot GOPs that admits noise peaks below about 3 dB pilot SNR. The projection then widens, and LMMSE lost to the old estimator on `mpp3`/`mpp0` (V7 `mpp3` −0.21 ± 0.16 dB latent). Gates from 3× to 8× all measured the same, turning those cells positive (V7 `mpp3` +0.75 ± 0.06 dB), so the gate is 4×. High-SNR channels are gated by the −15 dB rule and are unaffected.

## End to end: shared scorer, 64 eval clips

Each clip is encoded once, and both receivers demodulate the identical impaired waveform, using the published fade seeds. PSNR is the shared scorer's mean of per-GOP PSNR. The v0.1.23 V8 figure, 20.731 dB, matches the published fine-tune score.

| Model | Channel | v0.1.23 rx | LMMSE rx | Paired Δ PSNR | Δ LPIPS | Δ face PSNR (38 clips) |
|---|---|---:|---:|---:|---:|---:|
| **V8** `v8-hf3k-mpp12-ft` | **mpp12** | 20.731 | 21.004 | **+0.274 ± 0.029** | −0.0054 ± 0.0007 | +0.243 ± 0.040 |
| | clean modem | 23.308 | 23.317 | +0.009 ± 0.003 | −0.0002 ± 0.0001 | +0.009 ± 0.004 |
| | mpp3 | 18.918 | 19.116 | +0.198 ± 0.029 | −0.0058 ± 0.0011 | +0.204 ± 0.060 |
| **V9** `v9-wide4k` | **mpp12** | 21.268 | 21.496 | **+0.228 ± 0.022** | −0.0051 ± 0.0006 | +0.216 ± 0.038 |
| | clean modem | 23.477 | 23.504 | +0.027 ± 0.005 | −0.0003 ± 0.0002 | +0.024 ± 0.007 |
| | mpp3 | 19.121 | 19.332 | +0.211 ± 0.031 | −0.0076 ± 0.0012 | +0.146 ± 0.061 |

- **Clips:** 94% of V8 clips and 95% of V9 clips improve on `mpp12`.
- **Failures:** acquisition failures are identical in both arms (V8 0, V9 6 GOPs).

**Motion preservation.** This is the pooled ratio of reconstructed to source frame-difference energy and of Farneback flow magnitude. Neither drops; the new receiver slightly raises both.

| Model | mpp12 energy ratio | mpp12 flow ratio | mpp3 energy | mpp3 flow |
|---|---:|---:|---:|---:|
| V8 | 0.582 → 0.598 | 0.808 → 0.814 | 0.556 → 0.576 | 0.771 → 0.780 |
| V9 | 0.558 → 0.566 | 0.782 → 0.790 | 0.525 → 0.542 | 0.740 → 0.753 |

**Gaussian taps, reported separately (`mpp12-gauss`, not the bar):**

| Model | v0.1.23 rx | LMMSE rx | Paired Δ | Δ LPIPS |
|---|---:|---:|---:|---:|
| V8 | 21.426 | 21.603 | +0.177 ± 0.037 | −0.0043 ± 0.0010 |
| V9 | 21.863 | 22.044 | +0.181 ± 0.038 | −0.0029 ± 0.0011 |

The Gaussian-tap channel fades more slowly than the Butterworth `mpp12`, so the baseline scores about 0.6 dB higher. The gain there comes from window placement rather than from LMMSE.

Full-frame-rate videos of 16 clips (source | v0.1.23 receiver | LMMSE receiver, `mpp12`, 6 fps): [V8](../media/lmmse-receiver/v8-mpp12-side-by-side.mp4), [V9](../media/lmmse-receiver/v9-mpp12-side-by-side.mp4).

![V8 mpp12, eval clip 24, frame 5](../media/lmmse-receiver/v8-mpp12-eval24-f05.png)
![V8 mpp12, eval clip 0, frame 5](../media/lmmse-receiver/v8-mpp12-eval00-f05.png)
![V9 mpp12, eval clip 48, frame 5](../media/lmmse-receiver/v9-mpp12-eval48-f05.png)

More stills: [V8 clip 48](../media/lmmse-receiver/v8-mpp12-eval48-f05.png), [V9 clip 0](../media/lmmse-receiver/v9-mpp12-eval00-f05.png), [V9 clip 24](../media/lmmse-receiver/v9-mpp12-eval24-f05.png).

## Latent SNR A/B in simulation

Each cell is 48 paired seeds of one-GOP transmissions (`scripts/eval_receiver_ab.py`). The value is the effective SNR, ρ²/(1−ρ²) in dB, of latents × weights against the sent random latents. Per-seed values are floored at −20 dB, so two lost GOPs count as a tie. Without the floor, one seed whose GOP was lost in both arms swung ±20–30 dB. "Base" is the v0.1.23 receiver; the Δ columns are paired, ± 1 standard error.

| Channel | V8 base | V8 LMMSE | V8 placement | **V8 both** | V9 base | **V9 both** | V7 base | **V7 both** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AWGN 12 | 9.39 | +0.18 ± 0.01 | 0 | +0.18 ± 0.01 | 8.14 | +0.26 ± 0.01 | 5.93 | +0.34 ± 0.01 |
| AWGN 6 | 5.42 | +0.25 ± 0.01 | 0 | +0.25 ± 0.01 | 3.58 | +0.31 ± 0.01 | 0.71 | +0.44 ± 0.01 |
| AWGN 0 | −0.01 | +0.38 ± 0.01 | 0 | +0.38 ± 0.01 | −2.42 | +0.73 ± 0.03 | −6.37 | +1.33 ± 0.12 |
| AWGN −3 | −3.59 | +0.99 ± 0.07 | 0 | +0.99 ± 0.07 | −6.89 | +1.94 ± 0.16 | −12.34 | 0 (falls back) |
| **mpp12** | 3.08 | +0.54 ± 0.06 | +0.39 ± 0.08 | **+1.03 ± 0.10** | 2.60 | **+0.99 ± 0.11** | 1.05 | **+0.89 ± 0.10** |
| mpp6 | −1.86 | +0.18 ± 0.03 | +0.21 ± 0.05 | +0.42 ± 0.06 | −1.36 | +0.49 ± 0.06 | −3.55 | +0.66 ± 0.07 |
| mpp3 | −3.97 | +0.18 ± 0.04 | +0.16 ± 0.04 | +0.35 ± 0.05 | −4.24 | +0.50 ± 0.06 | −5.60 | +0.98 ± 0.08 |
| mpp0 | −6.03 | +0.19 ± 0.13 | +0.17 ± 0.05 | +0.34 ± 0.12 | −7.73 | +0.86 ± 0.13 | −9.31 | +0.86 ± 0.12 |
| ota40m 12 | 6.33 | +0.01 ± 0.04 | 0 | +0.01 ± 0.04 | 5.04 | +0.04 ± 0.04 | 2.98 | +0.29 ± 0.03 |
| ota40m 3 | 0.08 | +0.23 ± 0.03 | 0 | +0.23 ± 0.03 | −1.27 | +0.39 ± 0.03 | −4.85 | +0.80 ± 0.07 |
| mpd6 | −3.73 | +0.64 ± 0.12 | +0.65 ± 0.14 | +1.34 ± 0.20 | −4.52 | +1.29 ± 0.19 | −6.28 | +1.81 ± 0.22 |
| mppg12 (Gaussian taps, separate) | 4.34 | +0.01 ± 0.14 | +0.80 ± 0.11 | +0.85 ± 0.19 | 3.63 | +1.00 ± 0.16 | 1.95 | +0.95 ± 0.12 |

- **No cell is negative** for any mode or arm; each change alone also never loses. The −3 dB V7 cell is lost in both receivers, and the fallback correctly leaves it unchanged.
- **Placement moves only where a path is outside the CP.**
  - **Never on AWGN.** A single path sits at the 8-sample backoff.
  - **On V8 and V9, never on ota40m.** 0.6 ms is 4.8 and 7.2 samples, inside the backoff.
  - **On fading, it moves in 26–54% of transmissions.** These are the ones where acquisition locked onto the later path.
  - **On V7 ota40m, in 55% of transmissions.** 0.6 ms is 14 samples at 24 kHz, beyond the 8-sample backoff. There it adds +0.05 to +0.11 dB.
  - **When it moves, it usually gains.** On `mpp12` every placed transmission gains, by up to +2.2 dB. A few individual placements elsewhere lose: the worst is −2.1 dB on one V8 `mpd6` seed, and some `mpp0` seeds lose up to −0.2 dB. Every cell's mean is still positive.
- **Diagnosis of the prototype's low-SNR loss.** The measured Doppler spread was accurate: median 1.45 Hz on `mpp` against a true 1.41 Hz for the Butterworth taps. Fixing it to the true value did not help. The noise-widened delay support did the damage, which is what the 4× gate fixes.

## Streaming receiver and weak-signal beacon

This is the live `StreamingDemodulator` with SDR settings (boundary tracking on), fed 24-GOP continuous transmissions, 48 seeds per cell. "Beacon-verified" counts GOPs released with a beacon-derived frame counter, which is what the OTA scorer counts. GOP delivery was identical in both arms in every cell.

| Channel | V8 latent Δ | V8 beacon-verified | V9 latent Δ | V9 beacon-verified |
|---|---:|---:|---:|---:|
| AWGN −2 | +0.77 ± 0.01 | 3.6% → 12.8% | +1.55 ± 0.02 | 0% → 0% |
| AWGN −1 | +0.57 ± 0.00 | 15.6% → 31.3% | +1.10 ± 0.01 | 0% → 1.0% |
| AWGN 0 | +0.45 ± 0.00 | 38.0% → 48.4% | +0.81 ± 0.01 | 1.5% → 9.3% |
| AWGN 1 | +0.38 ± 0.00 | 55.4% → 60.2% | +0.62 ± 0.00 | 9.8% → 24.8% |
| AWGN 2 | +0.33 ± 0.00 | 67.8% → 71.4% | +0.50 ± 0.00 | 33.6% → 46.8% |
| AWGN 3 | +0.31 ± 0.00 | 72.9% → 76.0% | +0.43 ± 0.00 | 51.6% → 64.5% |
| mpp12 | +0.71 ± 0.03 | 72.0% → 71.5% (−0.5 ± 1.2) | +0.56 ± 0.02 | 68.8% → 70.4% |
| mpp6 | +0.35 ± 0.02 | 50.3% → 51.3% | +0.34 ± 0.02 | 21.4% → 27.7% |
| mpp3 | +0.38 ± 0.03 | 24.1% → 27.0% | +0.60 ± 0.04 | 0% → 1.7% |
| mpp0 | +0.65 ± 0.04 | 0.5% → 0% (−0.5 ± 0.5) | +0.99 ± 0.05 | 0% → 0% |
| ota40m 3 | +0.36 ± 0.01 | 18.1% → 27.6% | +0.49 ± 0.01 | 8.6% → 11.5% |

Weak-signal verification rises by 9–16 points for V8 at −2 to 0 dB, and by 8–15 points for V9 at 0 to 3 dB. The two small negatives are within one standard error.

## Over the air

**Setup** (the v0.1.23 release settings):
- **Radios:** PlutoSDR at 439 MHz into RTL-SDR serial 1001 at 37.2 dB, about 3 m apart.
- **Schedule:** 120 GOPs, TX −10 dB, then the weak gain at 60 s: V8 −50, V9 −50, V7 −45 dB.
- **Application:** the GUI in `--ota-validation` stress mode, with the same fixture and scorer as the release.
- **Arms:** both run from source, the v0.1.23 export against this branch, with the same pinned PyTorch checkpoints on CUDA, so only the receiver differs.
- **Order:** trials were interleaved (round 1: v0.1.23 then new; round 2 for V8/V9: new then v0.1.23).
- **Pluto:** power-down and minimum gain were verified from a fresh context after every trial.
- **Health:** every trial decoded 120/120 GOPs, with zero errors and zero underflows.

### Paired replays

Every capture was replayed through both receivers (`scripts/replay_sdr_stress.py`), so both see identical RF. The replay by the trial's own receiver reproduces its live latents (cosine 1.000). Values are pooled PSNR, the per-GOP paired Δ, and GOPs beacon-verified out of 48, weak window unless marked.

| Capture | Mode | Weak PSNR, v0.1.23 → LMMSE rx | Per-GOP Δ | Weak beacon-verified | Strong per-GOP Δ |
|---|---|---:|---:|---:|---:|
| release rc-V8 | V8 | 18.69 → 18.84 | +0.16 ± 0.02 | 21 → 26 (of 47) | +0.04 ± 0.01 |
| live old-V8 | V8 | 18.54 → 18.74 | +0.22 ± 0.03 | 32 → 30 | +0.03 ± 0.01 |
| live new-V8 | V8 | 18.51 → 18.70 | +0.19 ± 0.03 | 14 → 19 | +0.01 ± 0.00 |
| live old-V8-r2 | V8 | 18.60 → 18.78 | +0.18 ± 0.02 | 27 → 38 | +0.03 ± 0.01 |
| live new-V8-r2 | V8 | 18.63 → 18.77 | +0.13 ± 0.02 | 17 → 37 | +0.03 ± 0.01 |
| release rc-V9 (−50 dB) | V9 | 18.66 → 18.96 (23 valid) | +0.34 ± 0.05 | 0 → 12 | +0.02 ± 0.00 |
| release rc-V9 (−47.75 dB) | V9 | 19.52 → 19.67 | +0.16 ± 0.02 | 31 → 48 | +0.02 ± 0.00 |
| live old-V9 | V9 | 18.78 → 19.06 (16 valid) | +0.32 ± 0.04 | 0 → 2 | +0.02 ± 0.00 |
| live new-V9 | V9 | 18.37 → 18.71 | +0.39 ± 0.04 | 0 → 5 | +0.02 ± 0.00 |
| live old-V9-r2 | V9 | 17.20 → 17.50 (14 valid) | +0.27 ± 0.04 | 0 → 0 | +0.02 ± 0.00 |
| live new-V9-r2 | V9 | 18.26 → 18.75 | +0.45 ± 0.05 | 0 → 0 | +0.03 ± 0.00 |
| release rc-V7 | V7 | 20.48 → 20.66 | +0.20 ± 0.02 | 48 → 48 | +0.07 ± 0.01 |
| live old-V7 | V7 | 20.38 → 20.54 | +0.21 ± 0.02 | 48 → 48 | +0.07 ± 0.01 |
| live new-V7 | V7 | 20.36 → 20.56 | +0.22 ± 0.02 | 48 → 48 | +0.07 ± 0.01 |

- **PSNR:** every capture and window improves.
- **V8 weak beacon verification:** 111 → 150 of 239 GOPs over the five captures. Four captures rise, and one drops by 2.
- **V9:** 0 → 19 at −50 dB, and 31 → 48 at −47.75 dB.
- **Latent SNR** against the transmitted latents rises by +0.40 to +0.48 dB (V8 weak), +0.49 to +0.95 dB (V9 weak) and +0.49 to +0.50 dB (V7 weak).
- **"Valid" counts:** the release scorer keeps a GOP only if it is beacon-aligned or has latent cosine ≥ 0.5, which leaves the v0.1.23 V9 weak windows with 14–23 GOPs. The all-GOP column below includes every GOP.

### Live trials (separate link realisations)

| Trial pair | Strong PSNR | Weak PSNR (release scorer) | Weak PSNR, all GOPs | Weak beacon-verified / ambiguous | Weak median SNR |
|---|---:|---:|---:|---:|---:|
| V8 round 1: v0.1.23 / new | 21.20 / 21.22 | 18.54 / 18.70 | 18.54 / 18.70 | 32/0 / 19/0 | −1.1 / −1.2 dB |
| V8 round 2: v0.1.23 / new | 21.19 / 21.22 | 18.60 / 18.77 | 18.60 / 18.77 | 27/0 / 37/0 | −0.8 / −0.9 dB |
| V9 round 1: v0.1.23 / new | 21.39 / 21.40 | 18.78 (16 valid) / 18.71 | 18.41 / 18.71 | 0/32 / 5/0 | −1.2 / −1.4 dB |
| V9 round 2: v0.1.23 / new | 21.39 / 21.40 | 17.20 (14 valid) / 18.75 | 18.40 / 18.75 | 0/34 / 0/0 | −0.9 / −1.3 dB |
| V7: v0.1.23 / new | 22.54 / 22.59 | 20.37 / 20.56 | 20.37 / 20.56 | 48/0 / 48/0 | 4.2 / 4.1 dB |

**Weak beacon counts in live trials vary with the link.**
- **V8 round 1:** 32 → 19, even though the paired replay of that new-receiver capture shows v0.1.23 verifying only 14 there.
- **Release precedent:** 38 against 21 with essentially identical beacon code.

That is why the paired replays above are the beacon evidence.

**V9 ambiguity.** With the new receiver, no weak V9 GOP was ambiguous (latent cosine < 0.5 and no beacon); with v0.1.23, 32–34 of 48 were. That matters for the release's known limit that V9 at V8's TX power drops out of beacon verification.

![OTA frames, source | v0.1.23 receiver | LMMSE receiver: V8, V9, V7, strong then weak (separate live trials, same source GOP)](../media/lmmse-receiver/ota-frames-strong-weak.png)

## Ship rule

| Condition | Result |
|---|---|
| `mpp12` PSNR ≥ +0.2 dB and > 2× paired SE on V8 or V9 | Met on both. V8 +0.274 ± 0.029 (9.4 SE), V9 +0.228 ± 0.022 (10 SE) |
| No regression, including low-SNR profiles and OTA | Met. No latent-SNR cell is negative, every PSNR row improves, LPIPS improves, motion ratios hold, and every OTA capture improves |
| Weak-signal beacon verification not worse | Met. Simulated streaming rises 9–16 points (V8) and 8–15 points (V9) at weak AWGN; paired OTA replays go 111 → 150 (V8) and 31 → 48 (V9 at −47.75) |

## Cost and limits

- **CPU:** +0.2 ms (V8), +0.7 ms (V9) and +3.1 ms (V7) per one-second tracked GOP, under 1% of real time.
- **Not covered:**
  - AC16 keeps the v0.1.23 estimator; band A was not validated.
  - Blind late-join acquisition keeps the old estimator for its beacon search.
- **Doppler model:** the Gaussian Doppler assumption underfits the Butterworth `mpp12` skirts. LMMSE still gains there, because the spread is measured.
- **Mixed versions:** the decoders were fine-tuned on v0.1.23's receiver output and were not retrained; the gains carry over untrained, as SSTVAE found. A station on this receiver decodes v0.1.23 transmissions unchanged, since only the receiver differs.

## Evidence

On Beastmode, under `/pool0/AETV-runs/lmmse-receiver-20260927/`:

| Evidence | Files |
|---|---|
| Latent-SNR sweeps | `ab-v2.json` and `stream-v2.json` (per seed) |
| End-to-end PSNR | `psnr-eval64.json`, with per-clip values and motion |
| Replays | `replay/`, `replay-live/`, `replay-scores.json`, `replay-live-scores.json` |
| OTA trials | `ota/<trial>/` (raw `rtl.cu8`, `decoded.rgb`, `metrics.json`), `ota/trials.log`, `ota/all-gop-psnr.json` |
| v0.1.23 baseline source | `v0.1.23-src/` (`git archive v0.1.23`) |
| Bit-exactness check | `bitexact.py` |
