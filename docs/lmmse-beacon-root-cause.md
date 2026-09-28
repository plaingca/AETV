---
cursor:
  subagentId: "bc-f935febb-c8a4-50c1-90e2-9edbbfb8d9e7"
---

# LMMSE receiver and weak-signal beacons: root cause (2026-09-27)

**Result: the LMMSE channel estimate was not the cause, and no estimator change was needed.** The fix is in beacon verification. It is [PR #37](https://github.com/plaingca/AETV/pull/37) against `main`.

- **Every capture:** on identical samples, weak-signal beacon verification is now at or above v0.1.23 on all 22 OTA captures, including the eight from the 0.1.24 release check.
- **The two regressing captures:** V8 goes from 42 (v0.1.23) and 36 (0.1.24) to **48**, and V9 at −47.75 dB from 48 and 42 to **48**.
- **Picture quality:** it is unchanged from 0.1.24. Latents and weights are bit-identical to `main` on every capture.

**Root cause.** At weak signal, the beacon is verified through binary gates on individual superframes:
- a 13-chip sync correlation above 0.5 on both frames of a fallback pair;
- a CRC-valid soft-list candidate on both frames;
- no use of the positions and counters a verified stream can predict.

A single marginal superframe that flips at one of these gates removes about six GOPs of verification. The LMMSE estimate gave better chips on every measure, but on the two failing captures one such superframe each fell on the wrong side of a gate. Near saturation, only those losses could show.

## 1. Reconciliation: #35's gains and the release check's losses

| | #35 replay A/B | 0.1.24 release check |
|---|---|---|
| Captures | 5 V8 captures: the v0.1.23 release `rc-V8` plus 4 of #35's live trials; V9 at −50 and −47.75 dB from the v0.1.23 release | 8 new captures from 2026-09-27 (0.1.23 and 0.1.24 packages; V8, V9 at −50 and −47.75, V7) |
| Receiver code | #35 head `97fd7fe` | `main` at `c52f0a2`. **`aetv/` is identical to `97fd7fe`.** |
| Receiver config | `scripts/replay_sdr_stress.py` (boundary tracking, `verify_gap_phase=True`, automatic tuning) | Same script and settings |
| Beacon threshold | `find_beacon_superframe` default 0.5; pair fallback limited to the top 8 pairs | Same |
| Counting | GOPs with valid alignment in the live `metrics.json`, weak window 70–118 s, verified when the replay row has a beacon frame counter | **Same script** (`score_pairs.py` is #35's `score_replay.py`) |
| **v0.1.23 baseline on these captures** | **14–32 of 48** (V8), **0–31** (V9) | **42–48 of 48** on 6 of 8 |

Nothing in the method differed; both results are correct on their own captures. What differs is where the baseline sits:
- **#35's captures** started well below saturation, so LMMSE's better chips showed as gains.
- **The release captures** mostly started at 42–48 of 48. There a better receiver cannot gain, and a single unlucky superframe shows as a six-GOP loss.

Recounting all 22 captures with production replays gives v0.1.23 599, 0.1.24 678 and this fix 823 weak-window verified GOPs. On the eight release captures I count 313 → 317 for 0.1.24; the release report has 315. The difference is in how two borderline GOPs are matched, and it does not change any conclusion.

## 2. Superframe by superframe on the failing captures

Every analysis below uses production replays of the release captures, with `CHANNEL_ESTIMATOR` set to `pilot` or `lmmse` (placement on). They recorded, for every accepted GOP:
- the received symbols;
- both channel estimates;
- the equalized beacon chips.

The transmitted beacon chips are known exactly from the frame counter; the transmitted symbols come from `sent-latents.npy`.

**Verified GOPs** (GOP index along each string):

| Capture | Estimator | Verified pattern |
|---|---|---|
| V8, 0.1.24 package | legacy | `…VVVV` then **84–89 unverified**, then `VVVV…` |
| | LMMSE | `…VVVV` then **84–95 unverified**, then `VVVV…` |
| V9 −47.75, 0.1.24 package | legacy | all verified |
| | LMMSE | **84–89 unverified** |

Emulating `StreamingDemodulator._accumulate_beacon` offline, with its rolling three-superframe history, reproduces these patterns GOP for GOP.

**V9 −47.75 at GOP 84.** The history holds superframes 13 and 14 (frames 588 and 633); neither single-decodes in either receiver.

| | Legacy | LMMSE |
|---|---|---|
| Sync correlation at the true starts (frames 588 / 633) | 0.66 / **0.51** | 0.67 / **0.41** |
| Correct counter in each frame's soft list | yes / yes | yes / yes |
| Pair (588, 633) tried? | yes: 1 match, verified | **no**: the partner is below 0.5, and a 0.63 noise peak sits two chips away |

**V8 at GOPs 90–95.** The pair is formed in both receivers.

| | Legacy | LMMSE |
|---|---|---|
| Chip errors / chip SNR, earlier frame (633) | 37 / −1.0 dB | **34 / −0.8 dB** |
| Soft-list candidates after identity combining, earlier / later frame | 1 / 1 → verified | **0** / 1 (the correct 678) → not verified |

LMMSE had fewer chip errors on that superframe. They fell in the counter/CRC Golay words in a pattern the 256-candidate list decoder didn't recover, and a CRC-valid candidate carrying the right counter on the later frame alone isn't accepted.

**Weak-window superframes, per capture:**

| Capture | Legacy passes | LMMSE passes | Chip SNR Δ |
|---|---:|---:|---:|
| V8 | SF12, SF19 of 11 | 0 of 11 | +0.25 dB |
| V9 −47.75 | 4 of 11 | 7 of 11 | +0.45 dB |

On the V9 capture LMMSE decodes more superframes and still lost verification.

## 3. Hypotheses tested

Pooled over 827 weak-window GOPs and 153 superframes from the 16 V8/V9 captures:

| Hypothesis | Test | Result |
|---|---|---|
| Beacon carrier at the band edge, where 2-D LMMSE extrapolates badly | Data-aided residual \|r − ĥx\|² per carrier, LMMSE against legacy | **Refuted.** LMMSE is lower on every carrier: −0.60 to −0.67 dB at the beacon carrier, −1.1 dB at carrier 0 |
| Delay/Doppler statistics stale or wrong at the beacon position | Residual by frame position within the GOP (extrapolated last frame included) | **Refuted.** LMMSE is lower at all 32 positions: −0.4 to −1.1 dB, largest at the extrapolated last frame |
| Shrinkage bias scales the channel toward zero, breaking a threshold | Regress r on ĥx | **Refuted.** The gain is 0.765 for LMMSE against 0.693 for legacy (1 = unbiased): LMMSE is *less* biased. Phase bias is under 1.3°. |
| Hard decisions or a fixed amplitude reference | Code review of `beacon.py` | **Refuted.** Sync correlation is normalized and Golay soft scores are scale-invariant per word; there is no absolute threshold on chip amplitude |
| Noise-variance or covariance mismatch in the soft chips | Decode with production soft chips `Re(r ĥ*)/(\|ĥ\|²+N)`, plain MRC `Re(r ĥ*)`, and LLR `Re(r ĥ*)/N` | **Refuted as a cause.** Pooled single-superframe passes, legacy → LMMSE: production 17 → 30, MRC 13 → 27, LLR 13 → 29. The production scaling is no worse than either alternative, and LMMSE leads under each |
| Timing/window interaction | Release check's flag bisection | **Refuted.** Placement alone reproduces v0.1.23 exactly |
| **Verification gates amplify single superframes** | Per-step pass rates below, and the traces in section 2 | **Confirmed.** See below |

**Beacon step pass rates** (153 weak superframes):

| Step | Legacy | LMMSE |
|---|---:|---:|
| Beacon hard-chip error rate | 20.2% | 18.0% |
| Sync correlation at true start, mean (paired Δ +0.038 ± 0.007) | 0.618 | 0.656 |
| Sync correlation > 0.5 at true start | 80% | 86% |
| Superframes above 0.5 in only this receiver | 3 | 12 |
| Correct payload in the soft list | 35% | 52% |
| Single-superframe CRC decode | 9.2% | 15.0% |

**The minimal reproductions are unit tests in #37:**
- **Pairing:** a pair whose second sync sits below 0.5 while both payloads are list-decodable. `main` rejects it; the fix verifies it.
- **Anchor:** a superframe that is list-decodable only, with no sync, at a predicted position and counter. `main` rejects it; the fix verifies it.

## 4. Fix

In `aetv/beacon.py` `find_beacon_superframe`, plus the anchor in `StreamingDemodulator._accumulate_beacon`:

1. **Pair partners by position.** A sync peak is paired with the frames exactly one or two superframes away, ranked by the pair's combined sync evidence (still the top 8 pairs). The partner's own 13-chip sync no longer has to clear 0.5. Everything else is still required: a CRC-valid candidate in both frames, identical callsign/mode words, and a counter step equal to the spacing.
2. **Anchor on the verified phase.** After a verified superframe, the receiver keeps its absolute chip position, counter and callsign. Later superframes sit at known positions with known counters. A soft-list CRC candidate is accepted only at exactly the predicted position, with exactly the predicted counter and the anchor's callsign. That is the same corroboration a second frame gives today, except that the second observation is the earlier verified superframe.
   - A random candidate would need to pass a 16-bit CRC, match a 10-bit counter and match 36 callsign bits.
   - If the stream has slipped, the predicted position holds no superframe, so nothing verifies.
   - The anchor is cleared whenever the beacon history is: loss of lock, GOP-phase correction, or a realign larger than the CP.

This is receive side only, with no wire-format change, and the legacy equalizer isn't used for the beacon. The search is also cheaper: 4.2 ms mean per GOP on the weak V8 capture, against 6.9 ms before, because the anchor usually resolves first.

## 5. Results

### The 22 OTA captures: paired production replays, release counting, weak window

| Capture | v0.1.23 | 0.1.24 (`c52f0a2`) | Fix |
|---|---:|---:|---:|
| **0.1.24 release, V8, 0.1.24 package** | 42 | **36** | **48** |
| 0.1.24 release, V8, 0.1.23 package | 42 | 48 | 48 |
| 0.1.24 release, V9 −50, 0.1.24 package | 0 | 10 | 33 |
| 0.1.24 release, V9 −50, 0.1.23 package | 37 | 37 | 47 |
| **0.1.24 release, V9 −47.75, 0.1.24 package** | 48 | **42** | **48** |
| 0.1.24 release, V9 −47.75, 0.1.23 package | 48 | 48 | 48 |
| 0.1.24 release, V7, 0.1.24 package | 48 | 48 | 48 |
| 0.1.24 release, V7, 0.1.23 package | 48 | 48 | 48 |
| 0.1.23 release, V8 | 21 (of 47) | 26 | 47 |
| 0.1.23 release, V9 −50 | 0 (of 23) | 12 | 15 |
| 0.1.23 release, V9 −47.75 | 31 | 48 | 48 |
| 0.1.23 release, V7 | 48 | 48 | 48 |
| #35 live, V8 (v0.1.23 rx), round 1 | 32 | 30 | 48 |
| #35 live, V8 (LMMSE rx), round 1 | 14 | 19 | 42 |
| #35 live, V8 (v0.1.23 rx), round 2 | 27 | 38 | 42 |
| #35 live, V8 (LMMSE rx), round 2 | 17 | 37 | 37 |
| #35 live, V9 (v0.1.23 rx), round 1 | 0 (of 16) | 2 | 2 |
| #35 live, V9 (LMMSE rx), round 1 | 0 | 5 | 9 |
| #35 live, V9 (v0.1.23 rx), round 2 | 0 (of 14) | 0 | 7 |
| #35 live, V9 (LMMSE rx), round 2 | 0 | 0 | 14 |
| #35 live, V7 (both receivers) | 48 / 48 | 48 / 48 | 48 / 48 |
| **Total** | **599** | **678** | **823** |

- **Minimums:** the fix is at or above v0.1.23 and at or above 0.1.24 on every capture.
- **Strong windows:** 48/48 in all three receivers on every capture.
- **Denominators:** counts are out of 48 unless marked. Smaller denominators are the GOPs the release scorer aligns in V9's weakest windows.

**Picture quality is kept exactly.** The beacon change doesn't touch latents or weights, and they are bit-identical to `main` on all 22 captures. The release check's paired PSNR gains therefore stand as measured: weak windows +0.09 to +0.27 dB per GOP on all eight captures.

The shared-scorer eval, rerun on this branch (64 clips), reproduces #35 exactly:

| Model | `mpp12` PSNR, v0.1.23 rx → fix | Paired Δ | `mpp3` Δ | Clean-modem Δ |
|---|---:|---:|---:|---:|
| V8 | 20.731 → 21.004 | +0.274 ± 0.029 | +0.198 | +0.009 |
| V9 | 21.268 → 21.496 | +0.228 ± 0.022 | +0.211 | +0.027 |

### Simulated beacon A/B: streaming receiver, 24-GOP continuous transmissions, 48 paired seeds

The v0.1.23 and 0.1.24 arms run their own source trees. "Legacy + fix" is the pilot estimator with the new beacon search.

| Cell | v0.1.23 | 0.1.24 | Fix | Fix − v0.1.23 | Fix − 0.1.24 | Legacy + fix | Delivered, v0.1.23 / 0.1.24 / fix |
|---|---:|---:|---:|---:|---:|---:|---|
| V8 AWGN −2 | 3.6% | 12.8% | 18.7% | +15.0 ± 2.5 | +5.9 ± 1.6 | 4.8% | 1.000 / 1.000 / 1.000 |
| V8 AWGN −1 | 15.6% | 31.3% | 39.6% | +24.0 ± 2.7 | +8.2 ± 2.1 | 22.2% | 1.000 / 1.000 / 1.000 |
| V8 AWGN 0 | 38.0% | 48.4% | 55.9% | +17.9 ± 3.0 | +7.5 ± 1.9 | 48.5% | 1.000 / 1.000 / 1.000 |
| V8 AWGN 1 | 55.4% | 60.2% | 62.2% | +6.8 ± 2.5 | +1.9 ± 0.8 | 59.5% | 1.000 / 1.000 / 1.000 |
| V8 AWGN 2 | 67.8% | 71.4% | 71.9% | +4.1 ± 1.5 | +0.4 ± 0.4 | 69.3% | 1.000 / 1.000 / 1.000 |
| V8 AWGN 3 | 72.9% | 76.0% | 76.0% | +3.1 ± 1.2 | +0.0 ± 0.0 | 73.4% | 1.000 / 1.000 / 1.000 |
| V8 mpp12 | 72.0% | 71.5% | 71.5% | −0.5 ± 1.2 | +0.0 ± 0.0 | 72.0% | 0.981 / 0.981 / 0.982 |
| V8 mpp6 | 50.3% | 51.3% | 55.7% | +5.5 ± 2.4 | +4.4 ± 1.5 | 54.9% | 0.970 / 0.970 / 0.980 |
| V8 mpp3 | 24.1% | 27.0% | 33.4% | +9.3 ± 2.9 | +6.4 ± 1.8 | 30.6% | 0.958 / 0.958 / 0.983 |
| V8 mpp0 | 0.5% | 0.0% | 0.9% | +0.3 ± 0.8 | +0.9 ± 0.6 | 1.0% | 0.771 / 0.771 / 0.771 |
| V8 ota40m 3 | 18.1% | 27.6% | 36.8% | +18.7 ± 2.6 | +9.2 ± 2.1 | 26.5% | 0.978 / 0.978 / 0.978 |
| V9 AWGN −2 | 0.0% | 0.0% | 0.0% | +0.0 | +0.0 | 0.0% | 1.000 / 1.000 / 1.000 |
| V9 AWGN −1 | 0.0% | 1.0% | 1.9% | +1.9 ± 1.2 | +1.0 ± 1.0 | 0.0% | 1.000 / 1.000 / 1.000 |
| V9 AWGN 0 | 1.5% | 9.3% | 14.2% | +12.8 ± 2.2 | +4.9 ± 1.6 | 1.5% | 1.000 / 1.000 / 1.000 |
| V9 AWGN 1 | 9.8% | 24.8% | 32.9% | +23.1 ± 3.0 | +8.1 ± 2.0 | 15.3% | 1.000 / 1.000 / 1.000 |
| V9 AWGN 2 | 33.6% | 46.8% | 53.5% | +19.9 ± 2.8 | +6.7 ± 1.7 | 41.7% | 1.000 / 1.000 / 1.000 |
| V9 AWGN 3 | 51.6% | 64.5% | 66.1% | +14.6 ± 2.9 | +1.6 ± 0.8 | 56.7% | 1.000 / 1.000 / 1.000 |
| V9 mpp12 | 68.8% | 70.4% | 70.9% | +2.1 ± 1.3 | +0.5 ± 0.5 | 69.4% | 0.988 / 0.988 / 0.988 |
| V9 mpp6 | 21.4% | 27.7% | 38.8% | +17.4 ± 3.8 | +11.1 ± 2.2 | 36.4% | 0.968 / 0.968 / 0.973 |
| V9 mpp3 | 0.0% | 1.7% | 5.5% | +5.5 ± 1.8 | +3.7 ± 1.4 | 2.8% | 0.880 / 0.880 / 0.887 |
| V9 mpp0 | 0.0% | 0.0% | 0.0% | +0.0 | +0.0 | 0.0% | 0.767 / 0.768 / 0.768 |
| V9 ota40m 3 | 8.6% | 11.5% | 15.4% | +6.8 ± 2.1 | +3.9 ± 1.4 | 10.9% | 0.934 / 0.934 / 0.934 |

- **Against 0.1.24:** the fix is at or above it in every cell.
- **Against v0.1.23:** it is at or above in every cell except V8 `mpp12` (−0.5 ± 1.2 points), which is 0.1.24's own figure, within noise.
- **Legacy estimator + fix:** it gains less than LMMSE + fix. Once the gates stop discarding superframes, the better LMMSE chips turn into more verifications.
- **Delivered GOPs:** these rise on fading cells (V8 `mpp3` 0.958 → 0.983) because blind late-join acquisition uses the same pair search.
  - In the traced cases (V8 `mpp3` seeds 15 and 37), `main` never acquired in 20 s.
  - The fix acquired after about 11 s and delivered 14 GOPs, each matching the correct transmitted GOP: cosine 0.43–0.80, next best about 0.05.

### Safety

- **Noise:** 24 live receivers (V8 and V9) fed 90 s of noise each produced no acquisitions, no delivered GOPs and no beacons.
- **Unit tests:** they reject noise and each wrong anchor: a counter off by one, another callsign, or a position off by 4 chips.
- **Test suite:** 534 passed, 2 skipped.

## Evidence

On Beastmode, under `/pool0/AETV-runs/lmmse-receiver-20260927/`:

| Evidence | Files |
|---|---|
| Debug replays of all 22 captures, both estimators | `rc/all/`, with per-GOP received symbols, both channel estimates and beacon chips |
| Analysis scripts | `rc/superframes.py`, `rc/carriers.py`, `rc/position.py`, `rc/pool.py`, `rc/history.py`, `rc/search.py`, `rc/sync_stats.py`, `rc/fixes.py` |
| Three-way production replays | `fix/replay/`, scored by `fix/count.py` into `fix/beacon-counts.json` |
| Simulation | `fix/sim-{v0123,main,fix}.json`, `fix/sim-table.txt` |
| Eval | `fix/psnr-eval64.json` |
