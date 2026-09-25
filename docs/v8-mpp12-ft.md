# V8 channel-only fine-tune (mpp12)

`models/v8-hf3k-mpp12-ft.pt` is the V8 release checkpoint (`v8-hf3k-face-gan.pt`) fine-tuned for PSNR through the channel: V8 modem + Watterson `mpp12`. The waveform contract is unchanged: 192×108, 6 fps, 2,816 reals per GOP, band W. Both ends must use the same checkpoint.

| Field | Value |
|---|---|
| Init | `models/v8-hf3k-face-gan.pt` (not modified) |
| Selected step | 3000 of 4000 |
| Bytes | 215,739,153 |
| SHA-256 | `a8a60ade9d7f6e178c7595b2c10b96c26f5bd003d44c9c03d5bedbfdfe67f2f7` |

## Score

These scores come from the shared protocol (`scripts/eval_shared.py`):

- **Clips:** the first 64 of the seed-2026 shuffle of `data/openvid_aetv_cache/mode_ac6_192x108_12f`.
- **Channel:** each GOP goes through the V8 modem and then `mpp12`, with fade seed `2026 + clip·2 + gop`.
- **Receiver:** confidence gain 1.0.
- **Error bars:** ± is the paired standard error over clips.

| | mpp12 PSNR | mpp12 LPIPS | mpp12 face-region PSNR (38 clips) |
|---|---:|---:|---:|
| V8 release | 20.48 ± 0.52 | 0.255 | 18.57 |
| This checkpoint | **20.73 ± 0.50** | 0.257 | 19.18 |
| Paired delta | **+0.250 ± 0.051** | +0.002 ± 0.003 | +0.61 ± 0.15 |

53 of the 64 clips improve. Clean-latent PSNR falls by 0.27 ± 0.05 dB. Clean is reported for information only and does not gate this model.

A receiver confidence gain of 1.5 is a separate, receiver-only effect. It adds about +0.07 to +0.11 dB to either model and is not part of the numbers above.

## Recipe

`scripts/finetune_v8_channel.py --channel-only`:

- **Channel:** every row of every batch goes through the real V8 OFDM modem and `emulate(..., "mpp12")`, using training-only fade seeds. The channel error is passed straight through to the encoder.
- **No clean terms:** there is no clean render, clean anchor, consistency term, or regularizer toward the release model's clean output.
- **Loss:** MSE 1.0, L1 0.8, DWT 1.0, gradient 0.5, temporal 1.0 / acceleration 0.3 / cosine 0.2, VGG 0.06, YuNet region 1.5, face-crop VGG 0.1. All terms are computed on the channel output.
- **Schedule:** lr 1e-5 with a 50-step warmup and cosine decay to 1e-6. Batch 8. 4,000 steps, about 35 minutes on an RTX 4090.
- **Data:** training pool only. That is the seed-2026 shuffle after the 64 eval clips, minus 24 selection clips.
- **Selection:** the checkpoint with the best `mpp12` PSNR on the 24 selection clips, fade seed base 7300. The eval clips were not used.
- **Kill check at step 500:** pool `mpp12` +0.235 ± 0.120 dB, against a required +0.15.

```bash
python scripts/finetune_v8_channel.py --out runs/v8-mpp-only-ft --steps 4000 --kill-step 500 \
  --eval-interval 250 --channel-only --mse-weight 1.0 --l1-weight 0.8 --dwt-weight 1.0 \
  --grad-weight 0.5 --temporal-weight 1.0 --temporal-accel-weight 0.3 --temporal-energy-weight 0 \
  --temporal-cosine-weight 0.2 --lpips-weight 0.06 --temporal-lpips-weight 0 --region-weight 1.5 \
  --detail-weight 0 --contrast-weight 0 --face-perceptual-weight 0.1 --lr 1e-5 --lr-min 1e-6
python scripts/eval_shared.py --models v8-face-gan v8-mpp12-ft=models/v8-hf3k-mpp12-ft.pt \
  --gains 1.0 --out runs/shared-eval/eval64-v8mpp.json
```
