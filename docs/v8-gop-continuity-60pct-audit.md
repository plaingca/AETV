# V8 GOP continuity: 60% target audit

Date: 2026-08-27

The requested target was at least 60% reduction in the source-referenced
boundary RGB error at the independent `frame 5 -> frame 6` join. The target
was not reached by receiver-only context methods while preserving V8's
2,816-value GOP wire contract.

## Fixed evaluation

- Released checkpoint: `models/v8-hf3k-face-gan.pt`
- V8: 192x108, six frames/GOP, 2,816 analog values/GOP
- 32 held-out cached sequences
- Clean, 6 dB AWGN, 12 dB MPP, and measured-HF cells
- GUI boundary blending disabled
- Paired source/channel evaluation with boundary-specific metrics

## Results

| Candidate | Clean | AWGN 6 | MPP 12 | Measured HF | Quality result |
|---|---:|---:|---:|---:|---|
| Validated balanced stateful corrector | 6.8% | 8.4% | 8.1% | not in report | LPIPS improved |
| Full context, width 48, 800 steps | 15.5% | 15.4% | 14.8% | 15.5% | LPIPS flat/improved |
| Full context, width 64, 1000 steps | 22.9% | 24.3% | 24.1% | 24.9% | LPIPS flat/improved; detail/motion regressions |
| Full context, max residual 0.8, 1200 steps | 23.8% | 24.6% | 24.3% | **25.1%** | LPIPS flat; detail/motion regressions |
| Motion-aligned feature fusion, output strength 1.0 | 19.9% | 24.7% | 24.9% | 24.7% | LPIPS and spatial detail regressed |
| Learned bottleneck/latent-state adapter, 800 steps | not retained | not retained | not retained | 5.5% | LPIPS regressed 2.1% |
| Native decoder fine-tune, 8x boundary loss, 800 steps at 1e-5 | 17.7% | 21.5% | 25.0% | 22.1% | LPIPS +8.9/+10.4/+10.3/+11.8% |
| Boundary-only 2D predictor, 1000 steps | 17.5% | 21.9% | 24.5% | 23.7% | LPIPS flat; boundary-frame-only |
| Direct previous-latent first-slice injection | harmful | harmful | harmful | harmful | 5% already worsened; 50% gave -28% HF |
| Joint bottleneck-state adapter + decoder fine-tune, 1000 steps | negligible | negligible | negligible | 0.9% | no meaningful LPIPS change |
| Joint encoder + decoder + bottleneck state, 1000 steps | 0.1% | 0.1% | 0.1% | 0.1% | no meaningful LPIPS change |
| Full-resolution boundary-only predictor, 2000 steps | 23.6% | 31.5% | 34.9% | 33.6% | LPIPS flat/improved |

The strongest raw result is therefore 25.1% on measured HF, not 60%.
Derived `boundary_excess` reductions can exceed 60% because the candidate
also increases within-GOP temporal error; that is not counted as success for
the raw boundary-error target.

## Interpretation

Static previous-frame carryover peaks at 45% only in the MPP cell and is
worse on clean content. Motion extrapolation is weaker than static carryover.
Increasing adapter capacity, residual range, and boundary weighting produces
diminishing returns and eventually costs detail/LPIPS. The remaining 60%
target likely requires information not present in the receiver's current GOP:
native two-GOP training with a learned decoder state, a transmitted transition
reference/overlap, or a jointly trained temporal decoder. Any such change must
be evaluated under a new paired wire-rate gate.

The boundary-only predictor saw all six decoded frames from both adjacent GOPs
and modified only the first frame of the new GOP. A post-training amplitude
sweep peaked at 26.4% MPP and 25.5% measured HF raw reduction. It did not turn
the 58.8% boundary-excess result at nominal amplitude into a 60% raw result.

As an upper-bound protocol test, direct latent overlap/fusion was evaluated
on the same 32 clean sequences. One-frame overlap (stride 5) reduced incoming
boundary error by 28.6% at 20% extra values/frame; two-frame overlap (stride 4)
reduced it by 33.2% at 50% extra; three-frame overlap (stride 3) reached 30.9%
at 100% extra; four-frame overlap (stride 2) reached 33.9% at 200% extra; and
stride-one full overlap reached 35.3% at 500% extra. More overlap therefore
does not approach 60%, even when the wire-rate cost is allowed to grow.

Artifacts:

- `runs/gop-context-v8-full48-bw8/comparison.json`
- `runs/gop-context-v8-full64-aggressive/comparison.json`
- `runs/gop-context-v8-max08-bw32/comparison.json`
- `runs/gop-feature-flow-full-v8/comparison.json`
- `runs/v8-latent-context/adapter.pt`
- `runs/v8-latent-context/comparison.json`

The direct bottleneck-state experiment used the previous received GOP's
decoder bottleneck, cross-attention, confidence gating, and the same
contiguous three-GOP cache. It reached only 5.5% measured-HF raw reduction
and was rejected because overall LPIPS rose 2.1%.

The higher-learning-rate native decoder sweep fine-tuned the released decoder
on exact runtime-received two-GOP windows. Its 8x boundary-loss arm was the
strongest raw arm, but reached only 22.1% measured-HF reduction and regressed
LPIPS by 11.8% with a 16.3% spatial-detail loss. Full paired renders are in
`runs/v8-two-gop-boundary-sweep-lr1e5/renders/` and the report is
`runs/v8-two-gop-boundary-sweep-lr1e5/comparison.json`.

Naively replacing/blending the current latent grid's first temporal slice with
the previous GOP's final slice was also tested at 5%, 10%, 20%, 30%, 50%, 75%,
and 100%. It worsened raw boundary error at every nonzero strength in clean,
AWGN, MPP, and measured-HF cells; the latent coordinates are not directly
time-aligned across independent encoder calls.

Finally, the learned bottleneck adapter was trained jointly with the decoder's
temporal synthesis path for 1,000 steps. After correcting the evaluator to
load the fine-tuned decoder weights, the candidate reached only 0.9%
measured-HF raw boundary reduction. This rules out the hypothesis that the
small bottleneck adapter failed only because the released decoder was frozen.

The final zero-extra-bit variant also unfroze the encoder, allowing TX and RX
to learn a state-compatible 2,816-value representation together. After 1,000
steps and full paired evaluation, raw boundary reduction was 0.1% in clean,
AWGN, MPP, and measured-HF cells. This closes the tested learned-state options
under the released wire budget.

Receiver-only symmetric smoothing of both sides of the join was also tested
on the fixed 32-sequence runtime cache. A fixed alpha of 0.5 reduced raw
boundary error by 32.9% clean, 52.5% AWGN, 55.6% MPP, and 62.5% measured HF,
but increased within-GOP temporal magnitude by 11.4%, 19.4%, 23.2%, and 30.0%
respectively, with LPIPS worsening by 0.8--1.6%. It is therefore not a
promotion-safe 60% solution. A decoded step-magnitude gate selected smoothing
for every sequence and gave the same result. Interpolating frame 6 from the
decoded frames 5 and 7 peaked at only 37.1% measured-HF reduction.

A mixed-cell learned symmetric transition corrector was trained for 2,000
steps from the fixed 128-sequence runtime RX cache. It corrected both frame 5
and frame 6 from the decoded two-GOP context. On the held-out 32-sequence
cache it reduced raw boundary error by 26.9% clean, 46.0% AWGN, 47.2% MPP,
and 54.8% measured HF. LPIPS worsened 8.1%, 6.2%, 4.9%, and 3.4%, while
within-GOP temporal error increased about 32% in every cell. Residual-amplitude
sweeps did not improve those reductions. The result is rejected.

A measured-HF-only version with stronger within-GOP and anchor penalties was
trained for 3,000 steps. The two-sided form reached 56.0% raw measured-HF
reduction; the one-sided form, which changes only frame 6, reached 55.7% and
improved within-GOP temporal error by 16.1% with essentially unchanged PSNR.
Residual-amplitude sweeps peaked at 56.1%. Thus cleaner causal correction does
not cross 60% either.

A higher-capacity one-sided measured-HF corrector (width 256, 12 residual
blocks, 3,000 steps) was also trained. It reached 55.0% held-out measured-HF
raw reduction, with within-GOP temporal error improving 15.1% and PSNR nearly
unchanged. Capacity therefore did not close the remaining gap.

An aggressive one-sided measured-HF arm (maximum residual 0.8, boundary weight
64, 3,000 steps) reached 55.3% raw reduction; within-GOP error improved 16.6%
and PSNR improved 0.2%. Combining its learned frame-6 correction with fixed
symmetric alpha-0.5 damping reproduced the 62.5% measured-HF upper bound but
did not produce a distinct general solution.

The native centered three-GOP overlap model was then trained from scratch for
1,000 effective steps on genuinely contiguous five-GOP clips, with latent
channel corruption at 6--12 dB. On 8 held-out 30-frame clips it reduced the
mean raw boundary delta error by 13.8%, while increasing overall temporal
delta error by 11.4%. It is rejected and does not approach the 60% target.
The initial five-GOP cache contained only 12 frames; corrected clips are in
`runs/gop-boundary-data/v8_192x108_5gop_real_train` and
`runs/gop-boundary-data/v8_192x108_5gop_real_eval`.

Practical-RIFE v4.25-lite midpoint synthesis was evaluated specifically for
frame 6 from decoded frames 5 and 7 on the fixed runtime cache. It reduced raw
boundary error by 21.0% clean, 30.7% AWGN, 31.9% MPP, and 36.4% measured HF,
with negligible PSNR changes. It is rejected; motion-aware interpolation does
not close the 60% gap.

The literal decoder-bottleneck carryover path was retrained directly on the
fixed received-latent cache for 3,000 steps. The current GOP bottleneck was
cross-attended to the previous GOP bottleneck before the released synthesis
tail. Held-out raw boundary reduction was 11.2% clean, 29.1% AWGN, 31.4% MPP,
and 35.9% measured HF; within-GOP error stayed within about 1% and PSNR was
mostly flat. It is quality-safe but does not approach 60%.

Confidence-conditioned one-sided correction was trained for 3,000 steps across
all four fixed runtime cells, adding the two received-GOP confidence means as
receiver-state inputs. On the held-out 32-sequence cache it reached 26.5%
clean, 43.4% AWGN, 44.1% MPP, and 52.0% measured-HF raw reduction, below the
image-only specialized arm. Confidence conditioning is rejected.

An unconstrained measured-HF ceiling run (two-sided, maximum residual 1.0,
boundary weight 96, 3,000 steps) also reached 62.5% raw reduction. It exactly
matched the fixed alpha-0.5 symmetric smoother, while causing severe collapse:
within-GOP temporal error increased by about 326%, PSNR fell about 42%, and
LPIPS worsened about 41%. This is a degenerate smoothing crossing, not a
learned continuity improvement, and is rejected.

Frequency-selective symmetric smoothing was tested with average-pool kernels
3 through 25, retaining the original high-frequency residuals while smoothing
only the frame-5/frame-6 low-pass components. Its best raw reductions were
21.5% clean, 43.2% AWGN, 47.1% MPP, and 54.9% measured HF. The full-frame
62.5% smoother therefore cannot be made quality-safe by simply preserving
high-frequency detail.

Carried latent-statistics alignment was tested by affine-matching each current
received latent vector's mean and standard deviation to the previous received
GOP, with a strength sweep from 0 to 1. It was neutral on clean content and
inconsistent under channel distortion; the best measured-HF reduction was only
15.4%. Independent unit-RMS normalization may contribute to the visual shift,
but blind receiver rescaling cannot recover the discarded per-GOP amplitude.

The whole-GOP scene corrector was rendered alongside the released decoder,
full-resolution boundary predictor, symmetric smoother, and one-sided
corrector in `runs/v8-boundary-technique-renders/`. Visual inspection shows
less single-frame popping, but a persistent per-GOP scene/color shift remains;
the 56.8% measured-HF metric does not represent robust visual consistency.

As a direct diagnostic, the decoded current GOP was affine-matched in output
space to the previous GOP's global mean and standard deviation. Sweeping the
mixing strength from 0.25 to 1.0 reduced the raw boundary delta by only 0.1%
to 4.4% across the four fixed cells, while leaving the scene shift intact.
This rules out a simple output color/contrast mismatch as the primary cause.
The remaining artifact is content/latent-state drift caused by independently
decoded GOPs; post-hoc boundary filtering is therefore not a robust solution.

The decoder-context adapter was then changed to carry the corrected previous
bottleneck forward recursively, instead of recomputing context from each
GOP's uncorrected base feature. Fixed continuous 5-GOP RX caches were built
for 32 training and 8 held-out clips across all four channel cells. At the
best early held-out checkpoint (400 steps, lower learning rate), the
recurrent state reduced mean raw boundary error by 37.2% clean, 35.7% AWGN,
34.3% MPP, and 34.4% measured HF. Within-GOP temporal error increased by
3.6%, 4.2%, 4.3%, and 4.3%, respectively. It is a genuine stateful
improvement, but it remains well below the 60% target and is not a robust
solution yet.

A separate full-GOP causal pixel-state corrector was trained for 1,000 steps
on the same fixed 5-GOP runtime cache. It carried the corrected previous
decoded GOP while predicting a bounded residual for all six frames of the
next GOP. On the held-out eight clips it reached 49.8% clean, 52.7% AWGN,
50.9% MPP, and 52.4% measured-HF raw boundary reduction, but increased
within-GOP temporal error by 11.7%, 8.5%, 8.5%, and 8.4%. It is rejected:
extra pixel-state capacity moves toward the degenerate smoothing solution,
not robust scene preservation.

To test whether the frozen synthesis tail was limiting the recurrent approach,
the best recurrent bottleneck adapter was continued with only the decoder
output and temporal-skip layers unfrozen. A higher boundary-loss weight was
also tested. The best held-out result was 45.7% measured-HF reduction (44.5%
clean, 44.8% AWGN, 44.1% MPP), with within-GOP temporal error improving by
about 3--6%. Joint tail adaptation is quality-safer but does not close the
60% gap; the result is rejected for the stated target.

Finally, a high-ceiling one-sided measured-HF predictor was trained with
width 256, 16 residual blocks, maximum residual 1.0, boundary weight 128,
and 1,000 steps. On the fixed 32-sequence held-out cache it reached only
48.5% clean, 48.9% AWGN, 47.9% MPP, and 50.1% measured-HF reduction, while
increasing within-GOP temporal error by about 17.5% in every cell. Raising
the correction ceiling therefore does not provide the missing 60% robust
solution; it only moves toward distortion.
