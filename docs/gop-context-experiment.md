# GOP-to-GOP continuity experiment

Date: 2026-08-25

## Outcome

A resettable stateful correction adapter is supported by the V8 experiment.
It reduces every measured boundary error in every held-out sequence while the
released encoder, decoder, latent budget, and waveform remain unchanged. It is
an experimental artifact and is **not enabled in the GUI** by this change.

The simpler independent-GOP decoder fine-tune was rejected. Although it reduced
pixel-domain boundary error, it significantly worsened both overall LPIPS and
boundary-delta LPIPS.

## Motion-aligned follow-up

The small RGB adapter above was visibly too weak, so a second controlled
experiment tested three stronger approaches:

1. A 1.55M-parameter, six-block refiner with the previous full GOP as context.
   Capacity alone remained limited because the two GOPs were spatially
   unaligned.
2. RAFT-Small alignment followed by gated pixel fusion. This gave the largest
   boundary improvement, but strong settings softened moving details and
   regressed whole-clip LPIPS.
3. RAFT-aligned propagation of the decoder's 32-channel final feature map,
   followed by a learned 1.55M-parameter feature refiner and an optional weak
   output fusion. This retained more detail and is the supported follow-up.

Attempting to concatenate two or three GOP latent grids and decode them as one
larger temporal volume was also rejected. The released decoder was trained for
a fixed three-slice latent volume; untrained six/nine-slice decoding reduced
clean PSNR from 23.30 dB to 20.78/12.28 dB.

The retained balanced operating point uses feature context plus 10% gated
output fusion. On the same 32-sequence paired matrix:

| Cell | Boundary delta | Low-pass boundary step | Boundary delta LPIPS | Overall LPIPS | PSNR | Within-GOP delta |
|---|---:|---:|---:|---:|---:|---:|
| Clean | -11.71% | -23.70% | -10.68% | -0.09% (flat) | +0.044 dB | +0.51% |
| AWGN 6 dB | -13.15% | -24.53% | -10.56% | -0.17% | +0.044 dB | +0.25% |
| MPP 12 dB | -13.09% | -22.05% | -9.19% | -0.32% | +0.056 dB | +0.38% |

All boundary, PSNR, and SSIM changes are paired and statistically resolved.
Clean LPIPS has an improving mean but is statistically flat; AWGN and MPP
LPIPS improve. Runtime is 12.23 ms per boundary on the RTX 4090. The adapter
checkpoint is 5.94 MiB (1,552,416 parameters) and keeps the released latent
wire contract unchanged.

For comparison, strong 75% pixel-only flow fusion reduced boundary delta by
14.7-16.1% and boundary-delta LPIPS by 8.6-15.1%, but regressed overall LPIPS by
0.5-1.1%; it is retained only as a visual/aggressive preset, not promoted.

## Controlled setup

- Base: `models/v8-hf3k-face-gan.pt`
  (`f218376af9f9916050c9e345353da0c0970c392f58755efaa81d01e7ded8fc40`)
- Mode: V8, 192x108 at 6 fps, 6 frames/GOP, 2,816 analog values/GOP.
- Training: 128 three-GOP OpenVid sequences from native dataset shard 0/8.
- Evaluation: 32 three-GOP sequences from disjoint shard 1/8.
- Channel cells: clean, 6 dB AWGN, and 12 dB MPP fading.
- Pairing: identical source sequences and deterministic channel realizations.
- GUI boundary blending: disabled; metrics score raw decoder/adapter output.
- Adapter training: 500 steps, batch 2, frozen base encoder and decoder, 20%
  random state resets, source/perceptual anchoring, and source-referenced
  boundary losses.

The boundary loss matches the source transition rather than asking adjacent
frames to be similar, so genuine motion and cuts are not treated as errors:

```text
|(recon[g,0] - recon[g-1,-1]) - (source[g,0] - source[g-1,-1])|
```

## Results

Percent changes are paired candidate-versus-baseline means. Negative is better
for all columns except PSNR/SSIM.

| Cell | Boundary delta | Low-pass boundary step | Boundary acceleration | Boundary delta LPIPS | Overall LPIPS | Within-GOP delta |
|---|---:|---:|---:|---:|---:|---:|
| Clean | -6.76% | -20.01% | -6.83% | -4.45% | -0.19% | +0.62% |
| AWGN 6 dB | -8.42% | -19.75% | -9.97% | -4.36% | -0.14% | +0.43% |
| MPP 12 dB | -8.05% | -16.53% | -10.26% | -3.80% | -0.17% | +0.50% |

PSNR increased by 0.018 dB clean, 0.019 dB at 6 dB AWGN, and 0.023 dB in
12 dB MPP. SSIM was flat-to-slightly better. Every one of the 32 held-out
sequences improved in boundary delta, low-pass boundary step, and boundary
delta LPIPS under all three channel cells. Overall LPIPS improved on 31/32
clean sequences, 28/32 AWGN sequences, and 32/32 MPP sequences.

The resolved cost is a 0.4-0.6% increase in within-GOP delta error. Increasing
the within-GOP loss weight from 0.2 to 2.0 did not remove it; the balanced run
was retained because it had slightly better boundary LPIPS and spatial metrics.

The adapter contains 101,547 parameters and its checkpoint is 406 KiB. On the
RTX 4090 it measured 0.36 ms/GOP, compared with 10.9 ms/GOP for the V8 decoder.

## Safety properties

- No previous GOP means an exact tensor bypass.
- Zero context confidence means an exact tensor bypass.
- Context confidence is the minimum of the previous and current GOP confidence.
- The correction tapers to zero at the final frame, preventing an appearance
  offset from accumulating through an unbounded predictive chain.
- A missing GOP, new acquisition, mode change, or stream discontinuity can reset
  the adapter and immediately recover the released independent-GOP behavior.
- The adapter checkpoint is bound to the SHA-256 of its base checkpoint.

These properties are covered by focused tests. Loss/drop recovery still needs a
stream-level integration evaluation after the adapter is connected to the
receiver; the present experiment exercises explicit resets and channel
confidence but does not modify receiver state management.

## Artifacts

- Boundary-only rejected candidate and report:
  `runs/gop-boundary-v8/`
- Retained stateful adapter:
  `runs/gop-context-v8-balanced/adapter.pt`
  (`9a99891239f7f878fa08d44b7848e0a6ac4f9a759f175ddb57e4d9758869bb3b`)
- Full paired report:
  `runs/gop-context-v8-balanced/comparison.json`
- Raw per-sequence baseline/candidate metrics:
  `runs/gop-context-v8-balanced/baseline.json` and `candidate.json`
- Labeled source/baseline/context renders:
  `runs/gop-context-v8-balanced/renders/`
- Retained motion-aligned feature checkpoint:
  `runs/gop-feature-context-v8/refiner.pt`
  (`b2559948fb0540744b0cc9719801c2e4db96d6f65a9bdb289e62e18fcc2f1ae9`)
- Retained balanced feature/flow report and renders:
  `runs/gop-feature-flow-balanced-v8/`
- Feature-only report (strict LPIPS-improving point):
  `runs/gop-feature-context-v8/comparison.json`
- Aggressive pixel-flow report and renders (not promoted):
  `runs/gop-flow-v8-strong/`
- Fixed sequence manifests and tensors:
  `runs/gop-boundary-data/`

## Reproduction

```bash
.venv/bin/python scripts/experiment_gop_boundaries.py prepare \
  --train-sequences 128 --eval-sequences 32

.venv/bin/python scripts/experiment_gop_context.py all \
  --steps 500 --batch 2 --within-weight 2.0 \
  --out runs/gop-context-v8-balanced \
  --adapter runs/gop-context-v8-balanced/adapter.pt

.venv/bin/python scripts/experiment_gop_feature_context.py train \
  --steps 300 --batch 1 \
  --out runs/gop-feature-context-v8 \
  --refiner runs/gop-feature-context-v8/refiner.pt

.venv/bin/python scripts/experiment_gop_feature_context.py compare \
  --eval-sequences 32 --output-flow-strength 0.10 \
  --out runs/gop-feature-flow-balanced-v8 \
  --refiner runs/gop-feature-context-v8/refiner.pt

.venv/bin/python scripts/experiment_gop_feature_context.py render \
  --render-count 3 --output-flow-strength 0.10 \
  --out runs/gop-feature-flow-balanced-v8 \
  --refiner runs/gop-feature-context-v8/refiner.pt
```

The full `all` command also writes three labeled MP4 comparisons. Use
`compare` to repeat only the paired evaluation and `render` to regenerate only
the visual diagnostics.

## Production follow-up

1. Move the feature adapter module into `aetv/` and package its weights beside the V8
   checkpoint.
2. Maintain the previous decoded feature state/frame and confidence in the receive pipeline.
   Reset on acquisition, gaps, dropped GOPs, mode/callsign changes, and explicit
   scene-cut decisions.
3. Disable the current four-frame RGB boundary blend when the learned adapter is
   active; retain it only as a legacy-checkpoint fallback.
4. Repeat the same paired experiment for V7. The code is mode-generic, but the
   V7 checkpoint is not installed in this checkout, so no V7 result is claimed.
5. Before enabling by default, extend the paired matrix to measured-path replay,
   deliberate missing GOPs, and midstream tune-in/reacquisition.
