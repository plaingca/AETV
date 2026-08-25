---
license: artistic-2.0
tags:
  - amateur-radio
  - video
  - ofdm
  - joint-source-channel-coding
  - pytorch
  - onnx
---

# AETV release checkpoints

Training checkpoints and ONNX runtime graphs for [AETV](https://github.com/plaingca/AETV), a learned
video codec and OFDM modem for challenging amateur-radio HF/VHF channels.
The application downloads the selected mode's default runtime bundle from this
repository automatically and verifies its byte count and SHA-256 before use.

## Defaults

| Mode | Training checkpoint | Runtime bundle | Video/waveform |
|---|---|---|---|
| V7 | `v8-flex8k-ota-rxfix.pt` | `v8-flex8k-ota-rxfix.{encoder,decoder}.onnx` | 256×144 at 12 fps; receiver-corrected Flex-8k OTA model |
| V8 | `v8-hf3k-face-gan.pt` | `v8-hf3k-face-gan.{encoder,decoder}.onnx` | 192×108 at 6 fps; face-perceptual/GAN model |

The other files are reproducibility snapshots and operating-point variants.
`v7-flex8k-severe*.pt` are inference-only exports; optimizer and discriminator
state from the original training checkpoints has been removed.

## Use

Install and launch AETV normally; no Hugging Face account is required for the
public release files. To download a specific alternate manually:

```bash
hf download AETV/AETV v8-hf3k-robust.pt --local-dir models
```

The complete size/checksum index is in `manifest.json`. These checkpoints are
purpose-built for AETV's matching modem and mode geometry; they are not general
video-generation models. Generated detail can be plausible rather than
source-exact, particularly on damaged channels and small faces.
