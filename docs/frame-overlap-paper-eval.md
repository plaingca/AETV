# Training-free frame-overlap evaluation on released V8

This experiment evaluates Section 3.3 of *Perceptual Neural Video Compression
with Video Variational Autoencoder at Low Bitrates* (under review at ICLR
2026). The paper advances a `T`-frame input window by `T-1` frames, reconstructs
the shared transition frame twice, and averages the two reconstructions with
`theta = 0.5`.

## Rate mapping

The paper uses `T = 9`. Released AETV V8 uses `T = 6`, 2,816 transmitted values
per one-second GOP, and six source frames per second. Faithful overlap therefore
changes the steady-state rate from `2816 / 6` to `2816 / 5` values per source
frame: exactly 20% more symbols per second.

The rate-neutral diagnostic retains an average of `2816 * 5 / 6 = 2346.67`
coordinates per overlapping GOP. Uniform selection spreads omissions across
the full V8 decoder grid; prefix selection tests the truncation pattern seen by
the existing channel curriculum. Either form would require packet repacking;
the current modem's one-second, 2,816-value GOP framing cannot carry it as-is.

## Method

- Released checkpoint: `models/v8-hf3k-face-gan.pt`.
- No model training or GUI boundary blending.
- `theta` selected on eight disjoint training-cache clips from
  `{0, 0.25, 0.5, 0.75, 1}`; 0.5 won for all three rate variants.
- Final evaluation: 32 held-out contiguous OpenVid sequences.
- Transition metrics score both edges adjacent to the shared frame. This is
  necessary because interpolation distributes a decoder switch across the
  incoming and outgoing temporal deltas.

## Held-out clean results

| Metric | Released V8 | Faithful overlap (+20%) | Fixed-rate uniform |
|---|---:|---:|---:|
| PSNR | 22.9639 | 23.2126 | 21.5542 |
| SSIM | 0.76038 | 0.76716 | 0.71293 |
| LPIPS | 0.20654 | 0.20784 | 0.21504 |
| Two-sided seam delta | 0.03564 | 0.03044 | 0.03586 |
| Low-pass seam delta | 0.01487 | 0.01069 | 0.01582 |
| Seam acceleration | 0.06017 | 0.04351 | 0.04658 |
| Seam-delta LPIPS | 0.25669 | 0.22228 | 0.25391 |
| Within-region temporal delta | 0.02692 | 0.02661 | 0.02879 |

Faithful overlap reduces two-sided seam error by 14.59%, low-frequency seam
error by 28.11%, seam acceleration by 27.69%, and seam-delta LPIPS by 13.41%.
It also gains 0.249 dB PSNR and 0.0068 SSIM, although overall LPIPS regresses by
0.63%.

The fixed-rate form fails the clean gate: it loses 1.41 dB PSNR, 0.0475 SSIM,
and 4.12% LPIPS while two-sided seam error is 0.62% worse. Prefix truncation is
also unacceptable, losing 1.64 dB and increasing within-region temporal error
by 11.47%.

## Conclusion

The paper's averaging mechanism transfers to V8 and is substantially stronger
at boundaries than the earlier post-decode blend, but its redundant encoding
is the source of the gain. With a strict fixed-symbols-per-second contract, the
released V8 latent cannot surrender the required one sixth of its coordinates
without a large reconstruction penalty. Additional receive latency does not
fix this steady-state throughput mismatch.

The idea becomes more plausible in a jointly trained rate-neutral model: train
six-frame windows at five-frame stride while constraining each window to about
2,347 transmitted values, or increase the temporal window so the redundant
fraction is smaller. V7's 12-frame GOP would have a 9.09% faithful overhead,
but its checkpoint is not installed in this checkout, so no V7 quality result
is claimed here.

Artifacts are under `/pool0/AETV-runs/v8-paper-frame-overlap/`; the executable
experiment is `scripts/experiment_frame_overlap.py`.
