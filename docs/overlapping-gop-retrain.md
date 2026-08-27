# V8 overlapping-GOP full retrain

## Wire and latency contract

- One six-frame GOP still produces exactly 2,816 transmitted analog values.
- The encoder remains GOP-local, so transmitter latency is unchanged.
- The receiver buffers five latent vectors and jointly emits the center three
  GOPs, then advances by three. Adjacent decode calls share two latent GOPs of
  context. This adds one GOP (one second) of receive lookahead.
- Full-window attention runs at low resolution; expensive synthesis retains the
  18 emitted frames plus a two-frame halo instead of synthesizing all 30.
- An eight-GOP training sequence invokes two overlapping decoder windows and
  emits six consecutive GOPs. Losses therefore score both joins within a decode
  call and the join between separate decode calls used by the receiver.
- There is no recurrent state. Start/end acquisition can replicate the nearest
  latent with zero context weight; loss does not poison later windows.

The model uses width 192 and eight internal latent channels. Learned full-grid
packing maps the encoder's complete `8x3x14x24` grid to the fixed 2,816-value
vector. Learned unpacking maps every received value to the decoder's
`8x3x13x24` grid. Increasing internal dimensions therefore does not add radio
symbols or silently truncate the wider grid.

## Training objective

Training scores the three emitted center GOPs jointly with pixel MSE/L1,
spatial gradients, 3D Haar detail, perceptual features, signed temporal delta,
temporal acceleration, and source-referenced boundary delta/low-pass/
acceleration losses. Genuine cuts are targets rather than smoothing errors.

The first 7,500 steps are clean. A differentiable waveform channel ramps in
over the next 7,500 steps, then remains active with 0--18 dB SNR, broad
Watterson fading, and the measured-path mixture. Gradient checkpointing keeps
the 165.1M-parameter configuration within the RTX 4090 budget with batch two.
The 75,000-step schedule sees 1.2 million source GOPs and supervises 900,000
emitted GOPs, so the shorter step count is not a corresponding data reduction
from a one-GOP trainer.

## Run

```bash
scripts/train_v8_overlap_full.sh
```

Checkpoints and TensorBoard logs are written to
`/pool0/AETV-runs/v8-overlap-full-w192-c8-window5`. Only the newest three periodic
checkpoints are retained; `latest.pt` is a symlink rather than a duplicate.

Before promotion, evaluate fixed paired 32-sequence clean/AWGN/fading/measured
cells plus the 60-second Simpsons MPP12 render. Required gates are no clean or
channel LPIPS regression, at least 25% continuous-shot boundary reduction,
exact cut handling, and no meaningful within-GOP acceleration regression.
