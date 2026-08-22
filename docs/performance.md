# Flex-8k V7 measurements

Published checkpoint: `models/v7-flex8k.pt`  
Source run: `aetv-v7-flex8k-144p-stage2-nogan`, step 10000  
SHA-256: `afe476e5c5681210817a8e0598ec38ef40bdbd609485ad10d7a13ae9e6cd460b`

## Held-out OpenVid grid

30 clips through the real NumPy modem (`runs/aetv-v7-eval-run-latest` in the research tree). Means:

| Path | Mean PSNR |
|---|---:|
| Clean loopback | 23.93 dB |
| HF 24 dB | 23.86 dB |
| HF 12 dB | 23.17 dB |
| HF 6 dB | 20.69 dB |
| Multipath fading | 17.30 dB |

Clip PSNR spans 18.62–31.97 dB clean. Soft animation and high-detail faces sit at the low end; static or low-motion clips sit at the high end.

## 40 m OTA (2026-08-21)

30 GOPs of animation, Flex 6600 DAX, mode V7, callsign VA7EET.

| Quantity | Value |
|---|---|
| Codec ceiling (no modem) | 20.69 dB |
| Clean TX loopback | 20.98 dB |
| Frames recovered | 240 / 240 |
| Pilot SNR on the loopback | 48.7 dB |

The OTA path used this same checkpoint. Digital VVC-over-modem comparisons were run on the same GOP budget; AETV stayed watchable after the digital copies cliffed. Treat that as an analog-vs-digital operating-point result, not a claim that VVC is a worse compressor in the abstract.

## How to reproduce

```powershell
uv run aetv eval -- --checkpoint models/v7-flex8k.pt --out runs/eval --clips 8
uv run aetv simulate --source clip.mp4 --gops 4 --snr 12 --out sim.mp4
```
