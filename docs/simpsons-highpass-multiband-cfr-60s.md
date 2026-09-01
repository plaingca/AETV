# High-pass aggressive multiband evaluation

Date: 2026-08-25

Artifacts are in `runs/simpsons-highpass-multiband-cfr-60s`.

This variant adds a raised-cosine 150 Hz stop / 220 Hz pass high-pass before
the three-band aggressive compressor. The compressor bands are approximately
200--500, 500--1050, and 1050--2200 Hz. Voice RMS is restored after filtering,
so the voice branch retains its RF power allocation instead of surrendering it
to the AETV branch.

## Result

- source power removed by the pre-compressor high-pass: 55.19%
- received voice power below 200 Hz: 41.12% before, 2.89% after
- complete waveform PAPR: 5.329 dB
- sustained PAPR: 5.262 dB
- average power at 100 W PEP: 29.32 W
- clean video PSNR: 20.922 dB versus 20.925 dB without high-pass
- clean latent NMSE: 0.10009 versus 0.09945 without high-pass
- clean audio STOI: 0.769 versus 0.828 without high-pass
- AWGN 6 dB audio STOI: 0.532 versus 0.509 without high-pass and 0.470 control
- MPP 6 audio STOI: 0.413 versus 0.395 without high-pass and 0.375 control

The high-pass barely changes PAPR or equal-PEP RMS because both aggressive
variants already hit the 5 dB composite CFR target. It does improve low-SNR
speech intelligibility by reallocating bass power to the intelligibility band,
but the clean soundtrack is noticeably thinner. It is useful as an HF speech
profile, not as a transparent program-audio profile.

## Comparisons

- `simpsons_60s_ab_clean_equal_pep.mp4`: non-high-pass multiband on the left
  audio channel and high-pass multiband on the right at equal RF PEP
- `simpsons_60s_ab_clean.mp4`, `simpsons_60s_ab_awgn6.mp4`, and
  `simpsons_60s_ab_mpp6.mp4`: RMS-matched direct comparisons
- `waterfalls/simpsons_60s_*_rx_waterfall.mp4`: all seven synchronized channel
  visualizations
- `metrics.json` and `sweep.json`: complete numerical output
