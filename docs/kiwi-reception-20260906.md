# Kiwi reception investigation — September 6, 2026

The 15:08 reception contains a persistent timing jump that explains much of
its sustained quality loss. The latest 15:11 capture instead has weak and
varying signal quality, without evidence of an accumulating timing error.

## Captures and method

Both sessions used `ve7fsr.dyndns-home.com:8073`, dial 7.221 MHz, V8.
Original captures are in `C:\Users\patri\AETV\received\debug`:

- `20260906-150809-198_rx_kiwi`: one 30-GOP webcam transmission.
- `20260906-151110-575_rx_kiwi`: a 15-GOP transmission and a 30-GOP webcam
  transmission, separated by idle reception.

Replayed the raw IQ in 512-sample blocks through `IqToPassband`, using the
exact advertised sample rates (11998.909686 and 11998.909378 Hz), then fed
the modem in 800-sample blocks. Neither capture records packet gaps or
reconnections. Compared recovered payloads against clean decodes of the
corresponding saved TX WAVs. Correlation measures the confidence-weighted
latent payload, not image PSNR or subjective video quality.

## Confirmed timing defect and correction

In the 15:08 session, waveform correlation moves approximately 32–34 samples
earlier at GOP 15 and stays there. Cyclic-prefix timing independently finds
a -33-sample correction (4.125 ms). The original receiver continues decoding
at the old boundary because its pilots still pass presence detection. Kiwi
reception previously disabled all boundary tracking on the assumption that
the exact-rate resampler was sufficient.

The correction retains one cyclic prefix of history and checks nearby symbol
timing. It requires CP coherence of at least 0.50, a shift between four samples
and one CP, candidate pilot coherence of at least 0.20, and an improvement
of at least 0.05 over the current boundary. It does not enable the soundcard
pilot-slope clock estimator. Replaying that estimator on the latest capture
reduced recovery from 45 to 42 GOPs and introduced spurious adjustments.

| 15:08 capture, final five GOPs | Original | Corrected |
| --- | ---: | ---: |
| Mean pilot-derived SNR | 5.13 dB | 10.10 dB |
| Mean recovered-payload correlation | 0.724 | 0.935 |
| Total recovered GOPs | 30 | 30 |

There is exactly one boundary correction in the full 15:08 replay. The deep
fade at GOPs 11–12 remains; timing correction cannot restore information
lost there. The origin of the timing jump upstream of the modem has not been
isolated. Absence of Kiwi sequence gaps does not rule out a TX-side timing jump.

The latest session recovers all 45 GOPs, with no boundary adjustments and
identical payload results. Its final transmission improves toward the end:
mean pilot SNR rises from -1.59 dB in the first five GOPs to 4.09 dB in the
last five, with a severe dip around GOPs 8–12. Correlation against TX remains
aligned within one sample throughout both transmissions.

## Beacon and false-sync findings

V8 sends 32 beacon chips per second. A complete 181-chip beacon takes
5.65625 seconds and requires seven Golay words plus a valid CRC. The strong
15:08 reception identifies VA7EET by GOP 6. In the latest capture, the short
transmission never identifies; the second transmission first identifies at
GOP 23. Raw beacon decision error rates against the known transmitted chips
are 21.9% and 20.6%, respectively. Simple MMSE weighting, matched-filter
weighting, and clipping experiments did not improve first identification.
The beacon weakness in this capture remains unresolved by the timing fix.

The latest log's 83 rejected candidates are not accepted false syncs. It
emits exactly the expected 15 + 30 GOPs, and loses tracking only after each
transmission ends. A separate 15:07 test does emit an extra weak tail GOP
after its ten payload GOPs (-9.5 dB estimated SNR); end-of-transmission
acceptance deserves a separate regression and fix. This change does not
claim to fix that case.

## Validation and local artifacts

Regression coverage includes timing jumps immediately after acquisition and
later in a stream, preserving healthy timing, beacon continuity, blind
acquisition, noise-tail rejection, and source-specific tracking policy.
All 115 tests in `test_core.py`, `test_station.py`, and `test_kiwi.py` pass.
The correction is in the checkout; no live RF validation or packaged build
has been performed.

Local replay scripts and results are in `D:\AETV\runs\kiwi-20260906` and
`D:\AETV\runs`. In particular, `payload-analysis.json` contains per-GOP
measurements and beacon experiments; `fixed-replay.json` contains complete
corrected-session events and payload measurements; `timing-correction.png`
plots the measured improvement.
