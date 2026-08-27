# LTX-Video VAE channel-robustness experiment

## Outcome

The trained LTX-V8 candidate does **not** beat the released V8 checkpoint and
must not be promoted.  LTX's unmodified VAE has a much stronger clean
reconstruction prior, but compressing its six-frame latent from 6,144 values to
the fixed 2,816-value V8 RF budget makes the representation substantially more
sensitive to the analog channel than the native AETV latent.

## Fixed format and data

- Video format: 192x108 color, 6 frames/s, six frames per one-second RF GOP.
- RF budget: 2,816 real values per second over the existing W-band modem.
- LTX input: six source frames plus three repeated tail frames.  The official
  VAE emits `128x2x4x6 = 6,144` latent values and decodes nine frames; output is
  cropped back to the original six 192x108 frames.
- Training data: all 10,000 clips in the local V8 OpenVid cache.
- Held-out data: the same separate 32-clip
  `runs/openvid-cache-5fps-eval-v68` set used for prior V8 promotion tests.
- Reference: `models/v8-hf3k-face-gan.pt`.

The official `Lightricks/LTX-Video` VAE was loaded from its Hugging Face
Diffusers checkpoint.  Its unmodified clean reconstruction on the 32 held-out
clips was 32.82 dB PSNR / 0.0421 LPIPS before imposing the RF budget.

## Architecture

`aetv/ltx_channel.py` supplies a reliability-aware global rank-2,816 adapter:

```
LTX 128x2x4x6 latent
  -> per-channel normalization
  -> global 6,144-to-2,816 projection
  -> unit-RMS V8 symbol vector
  -> existing OFDM/channel/modem
  -> confidence-conditioned 2,816-to-6,144 projection
  -> LTX decoder
```

Local 64-channel and temporal-plane bottlenecks were rejected in probes.  They
reached low training latent MSE but decoded poorly because they discarded
decoder-sensitive cross-channel directions.  The global projection generalized
only after using the full 10,000-clip cache.

## Training

The experiment ran 125,000 optimizer updates across two controlled
continuations from the shared 20,000-step clean bottleneck anchor:

1. 20,000 clean latent-adapter steps on all cached OpenVid clips.
2. An 80,000-step latent-channel continuation, ending with 15,000 exact
   waveform/Watterson steps.  This branch was rejected because it degraded the
   clean anchor without beating V8 under channel impairment.
3. A separate 25,000-step source-aware continuation from the clean anchor.
   This used video reconstruction, spatial-gradient, temporal-delta and LPIPS
   losses through the LTX decoder, a low-rate update of the decoder input
   convolution, 25% explicit clean batches, a channel ramp, and 15,000 exact
   waveform/Watterson steps.

Observed throughput on the RTX 4090 was about 200 latent steps/s, 117 exact
waveform latent steps/s, and 12.9 source-aware video steps/s with LPIPS.

## Final paired 32-clip evaluation

All deltas below are LTX minus released V8.  Every listed PSNR/LPIPS regression
is larger than twice its paired standard error.

| Cell | V8 PSNR | LTX PSNR | Delta | V8 LPIPS | LTX LPIPS | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 29.2672 | 26.4633 | -2.8039 | 0.1157 | 0.1304 | +0.0147 |
| 18 dB | 28.3039 | 24.6476 | -3.6563 | 0.1157 | 0.2076 | +0.0919 |
| 12 dB | 27.7146 | 23.7233 | -3.9913 | 0.1219 | 0.2266 | +0.1047 |
| 6 dB | 26.1996 | 21.4731 | -4.7265 | 0.1475 | 0.2971 | +0.1496 |
| 0 dB | 22.6058 | 17.9563 | -4.6495 | 0.2432 | 0.4771 | +0.2339 |
| MPP 12 OTA | 25.4457 | 20.3519 | -5.0938 | 0.1596 | 0.3613 | +0.2017 |
| MPP 6 | 23.6650 | 18.6595 | -5.0055 | 0.2018 | 0.4198 | +0.2180 |
| MPP 0 | 19.8573 | 14.8860 | -4.9713 | 0.3205 | 0.5517 | +0.2312 |

The full report also contains SSIM, frame-delta L1, acceleration L1 and
delta-LPIPS for all 12 channel cells.

## Artifacts

- Final experimental checkpoint:
  `runs/ltx-v8-openvid-full-20260826/checkpoint.pt`
  (`fe3b8ad7bc6305a4b54678ad307728ccb082e710a982d041de4d27afcb9e173f`)
- Preserved clean bottleneck anchor:
  `runs/ltx-v8-openvid-full-20260826/checkpoint-clean-020000.pt`
  (`37547025a4b39eaef7c666e3990ea1103136654c24636190a925f1925ff47fe3`)
- Full paired result:
  `runs/ltx-v8-openvid-full-20260826/paired-32-final.json`
- Human-readable evaluation log:
  `runs/ltx-v8-openvid-full-20260826/paired-32-final.log`
- TensorBoard events:
  `runs/ltx-v8-openvid-full-20260826/tensorboard/`

## Decision

Keep `models/v8-hf3k-face-gan.pt` as the released default.  The LTX VAE remains
useful evidence that a strong pretrained reconstruction prior exists, but its
native latent is not robust enough after a 6,144-to-2,816 reduction.  A future
attempt should change the transport unit rather than repeat this objective—for
example, a stateful multi-GOP LTX stream that amortizes temporal latent slices,
or explicit decoder-sensitivity-weighted unequal error protection.
