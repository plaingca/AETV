# V8 OTA-perceptual checkpoint

This is the original V8 OTA-perceptual checkpoint. The receiver-adapted
`models/v8-flex8k-ota-rxfix.pt` now supersedes it as the application default;
see `docs/v8-ota-rxfix.md`.

| Field | Value |
|---|---|
| Mode | V7 waveform contract (256x144, 12 fps, 8 kHz occupied audio) |
| Training source | V7 published baseline, then two short stage-2 refinements |
| Selected state | `v8-ota-perceptual-c2`, step 250 |
| Inference size | 215,761,913 bytes |
| SHA-256 | `425f112924693170c61cebb6ab5865bd526714a4afae9aec88a37709441b5d47` |
| GPU | NVIDIA GeForce RTX 4080, bfloat16 AMP |

## OTA calibration

Candidate selection used the recorded 40 m examples in
`D:\SSTVAE\runs\aetv-ota-40m`, not a nominal SNR label alone.

| Capture analysis | Pilot SNR | Mean OTA PSNR | Notes |
|---|---:|---:|---|
| `decode-final` | 11.63 dB | 20.31 dB | clean 30-GOP result |
| `decode-run3` | 10.71 dB | 14.42 dB | 252 km, two repaired capture dropouts |
| `decode-repaired` | 9.25 dB | 17.71 dB | seven repaired capture dropouts |
| `decode` | 5.62 dB | 12.03 dB | marginal/unrepaired example |

The primary realistic selection band is therefore 9-12 dB. Training sampled
70% from 5.5-12 dB in the first refinement, then 75% from 3.5-10 dB in the
temporal-perceptual refinement. The remainder covered the complete -2 to
12 dB range; 35% of examples used differentiable Watterson fading.

## Released-model comparison

Thirty-two fixed held-out clips were paired across identical modem/channel
realizations. LPIPS and delta-LPIPS are lower-is-better; delta-LPIPS applies
LPIPS to signed inter-frame differences. The complete per-clip data and paired
standard errors are in `runs/v8-ota-perceptual-c2/final32.json`.

| Channel | PSNR base -> new | LPIPS base -> new | Delta-LPIPS base -> new |
|---|---:|---:|---:|
| Clean | 27.57 -> 28.40 | 0.1659 -> 0.1532 | 0.2454 -> 0.2305 |
| 18 dB | 27.79 -> 27.96 | 0.1472 -> 0.1387 | 0.2305 -> 0.2118 |
| 12 dB | 26.69 -> 26.83 | 0.1748 -> 0.1656 | 0.2472 -> 0.2282 |
| 10 dB OTA | 25.78 -> 25.94 | 0.2029 -> 0.1945 | 0.2617 -> 0.2437 |
| 9 dB OTA | 25.18 -> 25.32 | 0.2225 -> 0.2161 | 0.2716 -> 0.2550 |
| 6 dB | 22.73 -> 22.96 | 0.3043 -> 0.3043 | 0.3187 -> 0.3082 |
| 0 dB | 14.03 -> 15.77 | 0.5826 -> 0.5519 | 0.3944 -> 0.3942 |
| -2 dB | 12.29 -> 13.61 | 0.6340 -> 0.6050 | 0.4102 -> 0.4087 |
| MPG 12 | 25.16 -> 25.33 | 0.2162 -> 0.2096 | 0.2662 -> 0.2487 |
| MPP 12 OTA | 23.82 -> 24.08 | 0.2513 -> 0.2455 | 0.2815 -> 0.2651 |
| MPP 6 | 19.36 -> 20.47 | 0.4030 -> 0.3886 | 0.3430 -> 0.3384 |
| MPP 0 | 12.66 -> 14.26 | 0.6798 -> 0.6277 | 0.4004 -> 0.3953 |

PSNR improved in every cell. Frame LPIPS improved in every cell and tied to
four decimal places at 6 dB. Delta-LPIPS improved throughout the complete
grid. In the observed 9-12 dB OTA band, the new checkpoint also beats both
released severe-channel checkpoints by a wide margin; those models remain
stronger only at the 0/-2 dB PSNR cliff.

## Matched-symbol VVC reference

The reference uses official Fraunhofer VVenC 1.14.0 and VVdeC 3.2.0 with a
one-second IDR period. V7 carries 5,056 complex payload symbols per second, so
QPSK on the same slots has a 10,112 raw-bit/s ceiling. A 5,000 bit/s VVenC
target produced 6,193.75 bit/s over the 32 clips, leaving 38.75% of that raw
transport for FEC and framing. Metrics below assume the digital bitstream is
received without error; below its FEC threshold VVC instead has a cliff.

| Path | PSNR | SSIM | LPIPS | Delta-LPIPS |
|---|---:|---:|---:|---:|
| VVC, 6.194 kb/s, error-free | 27.36 | 0.7801 | 0.2608 | 0.2643 |
| AETV, 12 dB OTA | 26.83 | 0.8160 | 0.1656 | 0.2282 |
| AETV, 10 dB OTA | 25.94 | 0.7940 | 0.1945 | 0.2437 |
| AETV, 9 dB OTA | 25.32 | 0.7775 | 0.2161 | 0.2550 |

AETV wins both perceptual metrics at all three observed successful OTA
conditions and wins SSIM at 10 and 12 dB. Error-free VVC retains the PSNR
advantage, and narrowly wins SSIM at 9 dB. This is a channel-operating-point
comparison, not a claim that AETV is a better general-purpose compressor.

Full VVC results are in
`runs/v8-ota-perceptual-c2/vvc32/vvc_results.json`.

## OTA trial

1. Verify the checkpoint hash:

   ```powershell
   (Get-FileHash models\v8-flex8k-ota-perceptual.pt -Algorithm SHA256).Hash.ToLower()
   ```

2. Select `models/v8-flex8k-ota-perceptual.pt` on both transmitter and receiver,
   keep mode V7, and use the normal 24 kHz / 800-9200 Hz DIGU path.
3. Enable **Save TX waveform, Kiwi IQ, and modem debug logs**. Begin with an
   audio-only loopback, then a low-power 30-GOP OTA run using the same animation
   segment as the 2026-08-21 reference if available.
4. Record pilot SNR, accepted GOPs, LPIPS/delta-LPIPS, and PSNR. The key acceptance
   region is 9-12 dB pilot SNR; retain the 5-6 dB capture as the marginal test.

Headless example:

```powershell
uv run aetv send --checkpoint models\v8-flex8k-ota-perceptual.pt `
  --source clip.mp4 --callsign YOURCALL --gops 30 --out v8-ota.wav
```
