---
cursor:
  subagentId: "bc-cb0add16-30e3-5488-8e4d-95c8110feee2"
---

# Is 28–29 dB PSNR through the channel reachable?

**Short answer:**

- **2.2 kHz (band W): no.** Not over `mpp12` at the V8 contract, and not with any fps or resolution change tested. The channel's ideal capacity (8.1 kb/s) is below what the best error-free codec needs for 28 dB (10.6 kb/s). The plausible ceiling is **~23–24 dB** (today's AETV best is 20.73), or **~26 dB** if latency doubles to a 2 s intra period.
- **8 kHz (band U):** 28 dB is borderline. It needs a modem at ≥ 62% of capacity; 29 dB needs ≥ 77%.
- **16 kHz (band A):** 28 dB is reachable with a realistic modem, and 29 dB at ≥ 56% efficiency.

Nothing was trained. The measurements use the shared 64 eval clips and the shared scorer's PSNR statistic on Beastmode's CPUs. No PR, and no checkpoint changes.

## Headline table

Contract: 192×108, 6 fps, 1 s intra period (today's V8 latency). The separated codec is VVC (VVenC `slow`, QP sweep, `qpa=0`), delivered error-free. PSNR is the mean over the 64 clips at a constant bit rate.

| Band | Ideal `mpp12` capacity (ergodic) | Realistic 50–60% | **VVC PSNR at realistic rate** | VVC PSNR at ideal rate | Rate needed for **28 / 29 dB** | 28 dB? |
|---|---:|---:|---:|---:|---:|---|
| W 2.2 kHz (V8) | 8.1 kb/s | 4.0–4.8 kb/s | **23.3–24.3 dB** | 26.7 dB | **10.6 / 13.2 kb/s** | **No.** Above even ideal capacity |
| U 8 kHz (V7) | 17.2 kb/s | 8.6–10.3 kb/s | **27.0–27.9 dB** | 30.3 dB | 10.6 / 13.2 kb/s | Borderline: needs ≥ 62% efficiency (29 dB: ≥ 77%) |
| A 16 kHz (AC16) | 23.5 kb/s | 11.8–14.1 kb/s | **28.5–29.3 dB** | 31.7 dB | 10.6 / 13.2 kb/s | **Yes.** 29 dB needs ≥ 56% |

The same numbers with a **2 s intra period** (double latency):

| Band | VVC PSNR at realistic rate | At ideal rate |
|---|---:|---:|
| W | 25.4–26.3 dB | 28.8 dB |
| U | 29.0–29.9 dB | 32.2 dB |
| A | 30.5–31.3 dB | 33.5 dB |

At a 2 s intra period the bar drops to **6.9 / 8.6 kb/s**.

For scale, the best AETV model on W today (the E2b fine-tune) is **20.73 dB** through `mpp12`. That is what error-free VVC gives at about **2.7 kb/s**, a third of W's ideal capacity.

---

## 1. Channel capacity at `mpp12`

The `mpp12` noise model (`aetv/hfchannel.py`) is 12 dB SNR in a fixed 2,500 Hz reference, referenced to transmit power, whatever the band. Signal power is spread over the occupied band, so a wider band sees a lower SNR per Hz. All bands use 50 Hz carrier spacing.

Fading is the `mpp` preset: two equal-power Rayleigh paths, 1 Hz Doppler, 2 ms delay. The taps come from the repo's own generator, with perfect receiver CSI. Capacity is the Monte Carlo of `B · E[log2(1 + snr·|H(f,t)|²)]` over 4,000 one-second blocks. The analytic Rayleigh ergodic value agrees (W: 8.06 vs 8.08 kb/s).

| Band | Carriers | Occupied B | SNR in B | AWGN capacity (no fade) | **Ergodic** (long interleave) | 1 s blocks: 10th / 5th / 1st percentile |
|---|---:|---:|---:|---:|---:|---|
| W | 45 | 2,250 Hz | 12.5 dB | 9.5 kb/s | **8.1 kb/s** | 6.5 / 6.0 / 5.0 kb/s |
| U | 160 | 8,000 Hz | 6.9 dB | 20.6 kb/s | **17.2 kb/s** | 12.7 / 11.4 / 9.3 kb/s |
| A | 302 | 15,100 Hz | 4.2 dB | 28.0 kb/s | **23.5 kb/s** | 16.5 / 14.6 / 11.5 kb/s |

**Realistic** means 50–60% of ergodic. That covers cyclic prefix, preamble, pilots, beacon, and a practical code a few dB from Shannon. For reference, the AETV modem's own slot efficiency is 0.63 complex slots per Hz·s (W: 1,408 complex latent symbols/s in 2,250 Hz), before any coding loss. A separated digital link also has a cliff. The percentile columns are the rate a 1 s codeword can carry in 90%, 95% and 99% of seconds.

## 2. Best separated codec, error-free

**Setup:**

- **Clips:** the same 64 eval clips (192×108, 12 frames = 2.0 s at 6 fps).
- **Codecs:** ffmpeg 8.1 with `libvvenc` (VVC), `libaom-av1`, `libx265` (HEVC). Encodes are 4:2:0 from RGB and back.
- **PSNR:** the shared scorer's statistic, the mean of the two 6-frame GOP PSNRs on RGB in [0, 1].
- **Bit rate:** raw elementary stream bits / 2.0 s, with no container.
- **Constant rate:** each clip's PSNR is interpolated at a fixed rate on its own R–D curve, then averaged over clips. This models a constant-rate channel.
- **Floor:** the lossless 4:2:0 round trip is 40.0 dB at 516 kb/s, so colour conversion is not a limit.

**Mean PSNR (dB) at a constant bit rate:**

| Config | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 | 48 kb/s | kb/s for 24 / 26 / **28 / 29** dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **VVC 6 fps, 1 s intra** | — | 21.6 | 23.3 | 24.4 | 25.3 | 26.7 | 27.7 | 28.6 | 29.9 | 31.0 | 31.8 | 33.2 | 35.0 | 4.6 / 7.0 / **10.6 / 13.2** |
| **VVC 6 fps, 2 s intra** | 21.1 | 23.8 | 25.4 | 26.5 | 27.4 | 28.7 | 29.7 | 30.5 | 31.8 | 32.8 | 33.6 | 34.7 | 36.2 | 3.1 / 4.6 / **6.9 / 8.6** |
| AV1 6 fps, 1 s | | | | | | | | | | | 31.6 | 32.8 | 34.5 | (floor ~22 kb/s) |
| AV1 6 fps, 2 s | | | | | | | | | | 32.1 | 32.8 | 33.9 | 35.4 | (floor ~17 kb/s) |
| HEVC 6 fps, 1 s | | | | | | | | | | | | 28.6 | 32.0 | 30.5 / 33.5 |
| VVC 3 fps, 1 s, linear to 6 fps | 20.8 | 22.5 | 23.5 | 24.2 | 24.7 | 25.4 | 25.9 | 26.3 | 26.9 | 27.3 | 27.6 | 28.0 | 28.5 | 4.7 / 10.3 / 31.6 / never |
| VVC 3 fps, 2 s, linear | 22.4 | 23.8 | 24.7 | 25.3 | 25.7 | 26.4 | 26.8 | 27.2 | 27.6 | 27.9 | 28.1 | 28.4 | 28.7 | 3.2 / 6.8 / 21.3 / never |
| VVC 3 fps, 1 s, hold | 20.6 | 22.2 | 23.0 | 23.6 | 24.0 | 24.6 | 25.0 | 25.3 | 25.8 | 26.1 | 26.3 | 26.6 | 26.9 | 6.0 / 19.3 / never |
| VVC 2 fps, 1 s, linear | | | 23.0 | 23.5 | 23.9 | 24.4 | 24.7 | 25.0 | 25.3 | 25.6 | 25.8 | 26.0 | 26.3 | 6.5 / 32 / never |
| VVC 1 fps, 1 s, linear | | | 21.6 | 21.9 | 22.1 | 22.3 | 22.5 | 22.6 | 22.8 | 22.9 | 23.0 | 23.1 | 23.2 | never |
| VVC 128×72 → bicubic, 6 fps, 1 s | 18.4 | 21.7 | 23.2 | 24.2 | 24.9 | 25.8 | 26.5 | 27.0 | 27.6 | 28.0 | 28.3 | 28.6 | 29.0 | 4.8 / 8.5 / 20.2 / 52 |
| VVC 96×54 → bicubic, 6 fps, 1 s | 18.3 | 21.8 | 23.2 | 24.1 | 24.7 | 25.6 | 26.2 | 26.6 | 27.2 | 27.5 | 27.7 | 27.9 | 28.1 | 4.9 / 9.2 / 35 / never |
| VVC 96×54, 3 fps, linear | | 22.5 | 23.3 | 23.8 | 24.2 | 24.6 | 24.9 | 25.1 | 25.4 | 25.5 | 25.6 | 25.6 | 25.7 | 5.6 / never |

A blank cell means some clip could not reach that rate within the quality sweep.

**Findings:**

- **VVC is the best codec here.** AV1 (libaom `cpu-used 3`, CRF up to 58) never went below 17–22 kb/s on these tiny frames. Where the rates overlap, VVC beats AV1 by about 0.3 dB and HEVC by 4–5 dB.
- **Lower fps does not help at 28 dB.** It is cheaper below about 24 dB, but motion interpolation caps it: 3 fps linear tops out near 28.9, 2 fps near 26.4, 1 fps near 23.3.
- **Lower resolution does not help either.** It saturates at its upsampling ceiling (96×54 near 28.2; 128×72 near 29.1) and costs more bits than native 192×108 above 25 dB.
- **Longer latency is the one contract change that helps a lot.** Going from 1 s to a 2 s intra period cuts the 28 dB rate from 10.6 to 6.9 kb/s. Longer periods would help more, but the 12-frame clips cannot measure them.

## 3. Oracle ceilings (no bits, no channel)

These come from the same 64 clips and statistic. They reproduce the store's earlier bilinear numbers (2× = 27.98 dB).

| Oracle | PSNR |
|---|---:|
| Area-down 1.5× → bilinear / bicubic up | 28.87 / 30.14 |
| Area-down 2× → bilinear / bicubic up | **27.98** / 29.36 |
| Area-down 3× | 25.41 / 26.18 |
| Area-down 4× | 23.71 / 24.45 |
| Area-down 6× (32×18, the DeepStream contract) | 21.85 / 22.45 |
| 3 fps, hold / linear interpolation | 28.15 / 30.53 |
| 2 fps, hold / linear | 25.36 / 27.44 |
| 1 fps, hold / linear | 22.21 / 23.65 |
| Lossless 4:2:0 round trip | 40.05 |

28 dB needs either full 192×108 detail at 6 fps, or at most about 1.5–2× downsampling with bicubic up, or 3 fps with good interpolation. Those ceilings are with perfect pixels. Any code that also spends bits sits well below them at W's rates.

## 4. Conclusion per band

The "best plausible over `mpp12`" column is VVC at realistic capacity. A better source code, such as a learned codec, might add about 1 dB. JSCC can avoid the digital cliff but not beat the rate-distortion limit.

| Band | Best plausible over `mpp12` (192×108, 6 fps, 1 s latency) | AETV today | What would make 28 dB possible |
|---|---:|---:|---|
| **W 2.2 kHz** | **~23–24 dB** (≤ 26.7 at ideal capacity) | 20.73 (E2b) | **Not with fps or resolution changes.** It needs ≥ 10.6 kb/s delivered. That means a much better channel (about 16 dB `mpp` SNR with an ideal modem, about 28 dB with a 55% modem), or ≥ 2 s latency *and* a near-ideal (≥ 85%) modem, or more bandwidth |
| **U 8 kHz** | **~27–28 dB** (≤ 30.3 ideal) | V7 22.40 (cross-contract) | 28 dB at 1 s latency needs ≥ 62% modem efficiency. With a **2 s intra period**, 28 is comfortable (29.0–29.9 realistic) and 29 needs ≥ 50% |
| **A 16 kHz** | **~28.5–29.3 dB** (≤ 31.7 ideal) | AC16 22.19 (cross-contract) | **Reachable at today's latency** with a realistic modem. 29 dB needs ≥ 56% efficiency. With a 2 s intra period, 30.5–31.3 |

**Bottom line for Patrick's 28–29 dB target:**

- **Band W:** out of reach at `mpp12`. It is a capacity limit, not a model limit; the gap to 28 dB is more than a whole band's worth of bits.
- **What would get there:** the 16 kHz band, or the 8 kHz band with 2 s latency, at the same 192×108 / 6 fps picture.
- **Reality check:** the current AETV models sit 6–7 dB below those ceilings (V7 22.4, AC16 22.2). Closing that gap is its own project, whichever band is chosen.

**Caveats:**

- The 8 and 16 kHz rows are for delivering the same 192×108 / 6 fps clips over those bands, not the V7/AC16 native 256×144 contracts.
- Capacity assumes perfect channel knowledge. The realistic factor is a stated assumption, not a measured modem.
- Latency beyond 2 s was not measurable with 12-frame clips.

## Files

- Data and scripts: `internal/psnr-feasibility/` (`capacity.py`, `capacity.json`, `rd_sweep.py`, `rd_analyze.py`, `rd64.json` with per-clip R–D points, `summary.json`).
- On Beastmode: `runs/psnr-feasibility/` in the repo checkout.
