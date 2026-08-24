---
license: artistic-2.0
tags:
  - amateur-radio
  - video
  - ofdm
  - joint-source-channel-coding
  - pytorch
---

# AETV release checkpoints

Inference checkpoints for [AETV](https://github.com/plaingca/AETV), a learned
video codec and OFDM modem for challenging amateur-radio HF/VHF channels.
The application downloads the selected mode's default checkpoint from this
repository automatically and verifies its byte count and SHA-256 before use.

## Defaults

| Mode | Checkpoint | Video/waveform |
|---|---|---|
| V7 | `v8-flex8k-ota-rxfix.pt` | 256×144 at 12 fps; receiver-corrected Flex-8k OTA model |
| V8 | `v8-hf3k-face-gan.pt` | 192×108 at 6 fps; OpenVid-1M stage-2 model with face-crop perceptual/GAN fine-tuning |

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
