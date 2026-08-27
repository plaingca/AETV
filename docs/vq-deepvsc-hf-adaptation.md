# VQ-DeepVSC ideas adapted to V8 narrowband HF

The source paper is [VQ-DeepVSC (arXiv:2409.03393)](https://arxiv.org/abs/2409.03393).
Its transferable recipe has two stages: reduce temporal redundancy by selecting
hard-to-interpolate key frames according to content and channel quality, then
reduce spatial redundancy with a discrete vector-quantized image codec.

V8 already has a learned spatiotemporal codec and exactly 2,816 real analog
latent values per one-second GOP. Replacing it with the paper's VQ indices,
LDPC, QPSK/16-QAM, and digital OFDM would no longer be AETV's analog graceful-
degradation design. This experiment therefore adopts only the temporal recipe:

1. Decode 11 source frames at 12 fps for each one-second six-slot V8 GOP.
2. Keep both endpoints and greedily add the frame with the largest linear
   interpolation residual (RGB L1 plus an edge residual).
3. At reduced key-frame counts, fill all six codec slots by repeating the most
   important selected frames with diminishing returns.
4. Carry the resulting GOP through the released V8 encoder, production OFDM
   waveform, clean/AWGN/MPP/measured-HF channel, demodulator, and decoder.
5. Average repeated decoded slots and reconstruct the original timestamp grid
   with scene-cut-aware variable-gap interpolation.

The `uniform_k6` arm is the current uniform 6-to-12 fps pattern. `adaptive_k6`
isolates content-aware selection at the same temporal rate. The `adaptive_k5`,
`adaptive_k4`, and `adaptive_k3` arms test the paper's poor-channel trade:
fewer distinct frames, more protection per selected frame, and more receiver
interpolation. All arms retain exactly six codec slots and the same
2,816-value-per-second RF budget.

This is not wire compatible yet. The receiver currently receives an 11-bit
key-position mask per GOP out of band. A production version needs a robust,
versioned mini-header and must use the previous GOP's pilot SNR/coherence and
latent-confidence statistics so transmitter and receiver agree on the policy
without oracle knowledge of the current fade.

Run the fixed 32-clip experiment with:

```bash
.venv/bin/python scripts/experiment_adaptive_keyframes.py \
  --input "/path/to/native-12fps-or-higher-source.mp4" \
  --clips 32 \
  --report runs/vq-deepvsc-hf-adaptation/paired-32.json
```

Promotion requires the same paired clean/AWGN/fading/measured-path
PSNR/SSIM/LPIPS matrix used for other V8 candidates. A post-hoc best key count
per channel is diagnostic only; a deployable SNR/coherence-to-key-count policy
must be fitted on training clips and evaluated unchanged on the held-out 32.

## Paired 32-clip result

The fixed Simpsons evaluation rejected every adaptive arm. The table reports
arm-minus-`uniform_k6` paired mean deltas; positive PSNR/SSIM and negative
LPIPS would be improvements.

| Cell | Arm | PSNR dB | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Clean | adaptive k6 | +0.001 | -0.0064 | +0.0062 |
| AWGN 6 dB | adaptive k6 | -0.012 | -0.0054 | +0.0065 |
| MPP 12 dB | adaptive k6 | -0.018 | -0.0056 | +0.0043 |
| Measured HF | adaptive k6 | +0.037 | -0.0052 | +0.0067 |
| Clean | adaptive k5 | -0.200 | -0.0137 | +0.0155 |
| AWGN 6 dB | adaptive k5 | -0.128 | -0.0098 | +0.0157 |
| MPP 12 dB | adaptive k5 | -0.177 | -0.0111 | +0.0144 |
| Measured HF | adaptive k5 | -0.064 | -0.0071 | +0.0148 |
| Clean | adaptive k4 | -0.579 | -0.0210 | +0.0312 |
| Measured HF | adaptive k4 | -0.284 | -0.0101 | +0.0260 |
| Clean | adaptive k3 | -1.193 | -0.0348 | +0.0482 |
| Measured HF | adaptive k3 | -0.741 | -0.0195 | +0.0444 |

Adaptive k6 left PSNR effectively unchanged but regressed SSIM and LPIPS in
all four channel cells. Reducing the distinct-frame count made the tradeoff
progressively worse; repeated input frames are not independent protection in
V8's joint spatiotemporal latent. No arm passes the perceptual promotion gate,
so none should be wired into the transmitter or made a default.

The useful next experiment keeps all six uniformly timed source frames and
adapts protection inside the transmitted latent instead: train an explicit
importance-ranked latent grouping or unequal-power mask, with the existing
clean anchor and fixed paired channel gate. That preserves the part of the
paper that allocates channel resources by semantic importance without asking
an encoder trained at uniform 6 fps to ingest an irregular or repeated clock.

Detailed means, per-sequence metrics, schedules, paired standard errors, and
the exact channel cells are in
`runs/vq-deepvsc-hf-adaptation/paired-32.json`.
