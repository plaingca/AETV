---
cursor:
  subagentId: "bc-ba5cf180-3997-5ea2-9b93-5cd716aacd5b"
---

# 4 kHz wide mode, V9 (issue #12)

**Result: V9 scores 21.27 dB on `mpp12`, +0.79 ± 0.25 dB over V8. That clears the ship rule, so it has a draft PR: [#32](https://github.com/plaingca/AETV/pull/32), stacked on the shared scorer ([#28](https://github.com/plaingca/AETV/pull/28)).**

V9 lands between V8 (20.48) and V7 (22.40), as planned.

## Design

Issue #12 asks for "2k-ish for audio, 4k-ish for video" inside a 6 kHz transceiver filter.

| | V8 (band W) | **V9 (band M)** | V7 (band U) |
|---|---:|---:|---:|
| Carriers (latent + beacon) | 44 + 1 | **75 + 1** | 158 + 2 |
| Carrier span | 450–2650 Hz | **1100–4850 Hz** | 1000–8950 Hz |
| Audio sample rate | 8 kHz | **12 kHz** | 24 kHz |
| Reals per 1 s GOP | 2,816 | **4,800** | 10,112 |
| Video | 192×108 @ 6 fps | **192×108 @ 6 fps** | 256×144 @ 12 fps |

- **Band:** 3.8 kHz occupied, TX bandpass 900–5050 Hz. That leaves about 2 kHz of a 6 kHz filter for program audio.
- **Budget:** the band's full capacity, 75 carriers × 4 data symbols × 8 frames × 2 (I/Q).
- **Video:** the V8 picture is kept, so V8 can warm-start V9 and the eval cache matches.
- **Modem:** numerology is unchanged. The pilot sequence has 7.7 dB preamble PAPR, the same as band W.
- **Cost:** at equal transmit power, each carrier gets about 2.3 dB less SNR than on V8. The extra dimensions more than pay for it.

## Codec: warm start from V8

- **Widening:** V8 goes from 3 to 6 latent channels. The new encoder channels repeat V8's, and the decoder's inputs for them start at zero. Before training, V9 decodes exactly as V8 from the first 2,808 values. Its first pool score was 20.56, equal to V8.
- **Training:** 6,000 steps through the real V9 modem plus `mpp12`, using the V8 channel-only loss. It took about 50 minutes on the 4090. Selection used 24 training-pool clips.
- **Pool curve (vs V8 on the same clips):** +0.54 at step 500, +0.57 at the step-1000 kill check, +0.88 at 3,500 and +0.99 at 6,000. It was still rising slowly at the end, so a longer run might add a little.

## Score (shared scorer, 64 eval clips, each model through its own modem + `mpp12`)

| | mpp12 PSNR | LPIPS | SSIM | Face PSNR (38 clips) |
|---|---:|---:|---:|---:|
| V8 release | 20.48 ± 0.52 | 0.255 | 0.808 | 18.57 |
| **V9** | **21.27 ± 0.50** | 0.253 | 0.834 | 19.49 |
| Paired Δ | **+0.79 ± 0.25** | −0.001 ± 0.014 | +0.025 ± 0.018 | +0.92 ± 0.43 |

- 60 of the 64 clips improve.
- Clean latent and clean modem scores are for information only: 23.79 / 23.48 dB against V8's 23.92 / 23.35 dB.
- **Lost GOPs:** V9 lost 6 of 128 GOPs to a weak mode header (confidence below the 0.075 floor that applies without a mode prior). V8 lost none, because its misread headers still name other band-W modes. The live receiver uses the operator's mode as a prior (floor 0.045), so it would recover most of these. The score keeps the scorer's protocol unchanged.

Frames are in `media/wide-4k-mode/`: `v9-val{00,24,48}-f{02,05,09}.png`, with panels source | V8 | V9, all on `mpp12`, labeled with per-frame PSNR. These three standard clips gain less than average: +0.03 to +0.2 dB per frame.

## Open before merge

- **Upload weights:** `v9-wide4k.pt` and its ONNX bundle to `AETV/AETV`, then pin `HF_MODE_REVISIONS["V9"]`. For now the weights are on Beastmode only.
- **A/V composite:** V9 audio plus video in 6 kHz is not built yet.
- **Over-the-air test:** not done yet.
