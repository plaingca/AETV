# V7 severe-channel fine-tune

The V7 severe model is a 500-step stage-2 warm start from the published
`models/v7-flex8k.pt` checkpoint. It trains against the corrected physical SNR
calibration and calibrated Watterson fading in the production OFDM waveform
channel.

## Reproduce

Install the training dependencies and place the published baseline at
`models/v7-flex8k.pt`, then run:

```powershell
uv sync --extra train
.\scripts\finetune_v7_severe.ps1 `
  -CacheDir D:\path\to\openvid_aetv_cache `
  -InstallModels
```

The wrapper deliberately passes `--reset-steps`: the channel objective changed,
so the old optimizer moments and annealed learning-rate position must not be
resumed. Its complete recipe is:

- V7 stage-2 differentiable OFDM channel
- physical SNR uniformly sampled from -2 through +6 dB
- Watterson fading on 40% of training examples
- 500 steps, batch 1, four-step gradient accumulation
- learning rate 1e-5 and a 25-step channel ramp
- no adversarial discriminator
- checkpoints and held-out evaluation at steps 250 and 500

Step 250 is installed as `v7-flex8k-severe-balanced.pt`; step 500 is installed
as `v7-flex8k-severe.pt`. V8 OTA-perceptual later superseded it as the default.

## Reference results

Three cached held-out V7 clips were evaluated with five deterministic seeds per
impaired condition through the production continuous modem and neural decoder.

| Checkpoint | Clean | AWGN 6 | AWGN 0 | AWGN -2 | MPP 0 |
|---|---:|---:|---:|---:|---:|
| Published baseline | 23.13 | 19.18 | 14.22 | 12.57 | 12.33 |
| Step 250 balanced | 22.53 | 19.25 | 16.08 | 14.60 | 13.97 |
| Step 500 severe | 21.60 | 19.21 | 16.54 | 15.20 | 14.53 |

Values are mean reconstructed-video PSNR in dB. All impaired rows decoded
15/15 transmissions. The locally produced checkpoint hashes were:

| File | SHA-256 |
|---|---|
| `v7-flex8k-severe-balanced.pt` | `18d610a35797f3bffb86f55bd8d9a79182d24b961640b19ffa27304763dcfa03` |
| `v7-flex8k-severe.pt` | `e900bd7da2f080d23926a64f76d4b2624c413e08ddcdb91fde81c8acdd9a53b4` |

Exact hashes require the same baseline, cached clips, dependency versions,
hardware kernels, and random-number stream. Metric-level reproduction is the
portable expectation across GPU systems.
