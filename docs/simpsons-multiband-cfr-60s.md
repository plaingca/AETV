# Aggressive multiband Simpsons evaluation

Date: 2026-08-25

Artifacts are in `runs/simpsons-multiband-cfr-60s`.

## Processing

- bands: 0--450, 450--1050, and 1050--2200 Hz
- threshold: -20 dB relative to the input voice peak
- ratio: 6:1
- attack/release: 3/100 ms
- maximum gain reduction: 12 dB
- real-audio peak limiter: 7 dB crest target
- 3 dB CESSB after multiband processing
- 32-way AETV common-phase selection per GOP
- selected whole-composite CFR target: 5 dB

The input real-audio crest factor was 17.88 dB and the multiband/limiter output
was 8.93 dB before CESSB. Average gain reduction was 4.44 dB in the low band,
0.83 dB in the middle band, and 0.61 dB in the high band. The low band reached
11.83 dB maximum gain reduction.

## Result

- complete waveform PAPR: 5.398 dB
- sustained PAPR: 5.225 dB
- average power at 100 W PEP: 28.85 W
- clean video PSNR: 20.925 dB versus 20.938 dB control
- clean latent NMSE: 0.09945 versus 0.09710 control
- clean audio STOI: 0.828 versus 0.949 control
- clean audio SI-SDR: 7.06 dB
- initially composite-limited samples: 2.812%

The video cost is negligible, but the speech processing is plainly aggressive:
it gains about 1.86 dB recovered voice RMS at equal PEP relative to the earlier
7.2 dB CESSB/CFR candidate, at the cost of audible pumping and spectral-balance
changes. This should be treated as the loudness endpoint, not the default voice
profile.

## Comparisons

- `simpsons_60s_ab_clean_equal_pep.mp4`: earlier 7.2 dB candidate on the left
  audio channel and 5.4 dB multiband candidate on the right, both normalized to
  the same RF peak-envelope ceiling
- `simpsons_60s_ab_clean.mp4`, `simpsons_60s_ab_awgn6.mp4`, and
  `simpsons_60s_ab_mpp6.mp4`: RMS-matched audio comparisons
- `waterfalls/simpsons_60s_*_rx_waterfall.mp4`: source, decoded video, and RX
  spectrum for all seven channel profiles
- `sweep.json` and `metrics.json`: complete numeric results
