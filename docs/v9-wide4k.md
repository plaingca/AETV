# V9: 4 kHz wide mode (band M)

V9 is a video mode for transceivers with 6 kHz transmit filters. It carries the V8 picture (192×108, 16:9, 6 fps) on a 3.8 kHz OFDM band, with 1.7× V8's latent budget.

| | V8 (band W) | **V9 (band M)** | V7 (band U) |
|---|---:|---:|---:|
| Carriers (latent + beacon) | 44 + 1 | **75 + 1** | 158 + 2 |
| Carrier span | 450–2650 Hz | **1100–4850 Hz** | 1000–8950 Hz |
| TX bandpass | 350–2750 Hz | **900–5050 Hz** | 500–9500 Hz |
| Audio sample rate | 8 kHz | **12 kHz** | 24 kHz |
| Reals per 1 s GOP | 2,816 | **4,800** | 10,112 |
| Video | 192×108 @ 6 fps | **192×108 @ 6 fps** | 256×144 @ 12 fps |
| Beacon mode index | 8 | **9** | 7 |

Numerology is unchanged: 50 Hz carrier spacing, 25 ms symbols, 8 frames per GOP. The budget is the band's full capacity: 75 carriers × 4 data symbols × 8 frames × 2 (I/Q). The occupied 3.8 kHz leaves about 2 kHz of a 6 kHz filter for program audio. A V9 A/V composite is not part of this change.

## Checkpoint

`models/v9-wide4k.pt`: V8 widened from 3 to 6 latent channels, then trained through the V9 modem.

| Field | Value |
|---|---|
| Init | `models/v8-hf3k-face-gan.pt`, via `aetv.models.widen_latent_channels` |
| Selected step | 6000 of 6000 |
| Bytes | 215,987,409 |
| SHA-256 | `15ada7b19f24a6e475dd7d0cab3c130903a45e41c4287e2817674b9e3bb4a51b` |

Widening copies the V8 encoder's three output channels into the new ones and zeroes the decoder's inputs for them. Before training, V9 decodes exactly as V8 from its first 2,808 values.

## Score

These scores come from the shared protocol (`scripts/eval_shared.py`):

- **Clips:** the 64 eval clips.
- **Channel:** each model goes through its own modem and then `mpp12`, with fade seed `2026 + clip·2 + gop`.
- **Receiver:** confidence gain 1.0.
- **Error bars:** ± is the paired standard error over clips.

| | mpp12 PSNR | mpp12 LPIPS | mpp12 SSIM | mpp12 face-region PSNR (38 clips) |
|---|---:|---:|---:|---:|
| V8 release | 20.48 ± 0.52 | 0.255 | 0.808 | 18.57 |
| **V9** | **21.27 ± 0.50** | 0.253 | 0.834 | 19.49 |
| Paired delta | **+0.79 ± 0.25** | −0.001 ± 0.014 | +0.025 ± 0.018 | +0.92 ± 0.43 |

60 of the 64 clips improve. For information only: clean-latent PSNR is 23.79 against V8's 23.92, and clean-modem PSNR is 23.48 against 23.35.

At the same transmit power, each V9 carrier gets about 2.3 dB less SNR than a V8 carrier. The gain comes from the extra latent dimensions.

V9 lost 6 of its 128 GOPs to a weak mode header (confidence 0.04–0.07 against the 0.075 floor). V8 lost none, because its misread headers still name other band-W modes. The live receiver passes the operator's mode as a prior, which lowers the floor to 0.045, so it would recover most of those GOPs. The score above keeps the shared scorer's protocol unchanged.

## Recipe

`scripts/finetune_wide4k.py`:

- **Channel:** every row goes through the real V9 modem and `emulate(..., "mpp12")`, using training-only fade seeds. The channel error is passed straight through to the encoder.
- **Loss:** the V8 channel-only fine-tune loss.
- **Schedule:** lr 2e-5, with 5× on the four widened tensors. 100-step warmup, then cosine decay to 1e-6. Batch 8. 6,000 steps, about 50 minutes on an RTX 4090.
- **Data and selection:** training pool only. Checkpoints are selected on 24 pool clips through V9 + `mpp12`, fade seed base 7300.
- **Kill check at step 1000:** V9 must at least match V8 on the same pool clips. It was +0.57 dB.

```bash
python scripts/finetune_wide4k.py --out runs/v9-wide4k --steps 6000 --eval-interval 500 \
  --kill-step 1000 --kill-margin 0.0 --real-rows 8 --workers 8
python scripts/eval_shared.py --models v8-face-gan v9-wide4k --gains 1.0 --out runs/shared-eval/eval64-v9.json
python scripts/export_onnx_runtime.py models/v9-wide4k.pt --output models --runtime-name v9-wide4k
```
