# Receiver accuracy review — September 6, 2026

Reviewed capture buffering, Kiwi IQ conversion, acquisition, carrier-frequency
correction, cyclic-prefix timing, sample-clock estimation, pilot channel/noise
estimation, payload equalization, beacon decoding, and the model input contract.
The largest measured opportunity is the channel estimate used by the equalizer.
Several acquisition and stream-state defects also needed correction.

A subsequent [beacon reception improvement](beacon-reception-20260906.md)
recovers two previously unidentified short transmissions. The measurements
below describe the receiver before that follow-up.

The guarded boundary correction is now also enabled for soundcard reception.
Soundcard retains its supported sample-clock tracking alongside that check;
Kiwi continues to use boundary correction without the soundcard clock loop.
Regression coverage includes positive/negative small jumps immediately after
acquisition and later, a 375-sample endpoint insertion, simultaneous +/-500 ppm
clock drift and a 4 ms deletion across all three modem bands, beacon continuity,
and idle-hum rejection.

Offline soundcard replay compared the clock tracker alone with both controls
enabled. The September 1 recordings `141048-547` and `140916-368` (12 kHz analog
A/V capture, replayed through the station's separator and resampler) retain
162 and 36 GOPs respectively, with identical mean pilot SNR, callsigns, and
tracking event counts. The native 8 kHz `140303-005` recording recovers no GOPs
under either configuration, so it provides no positive payload validation.
Detailed events are in `runs/soundcard-boundary-20260906/`. This remains offline
validation; a live soundcard hardware test has not been performed.

## Changes in the checkout

### Pilot estimation and equalization

Previously, noisy least-squares pilot estimates went directly into the
equalizer. The new estimator removes the bulk frequency phase slope, tries
three short smoothing kernels, and selects the amount of smoothing using the
raw noise estimate and smoothing residual. It backs off when noise is low or
channel curvature makes smoothing expensive. This reduces pilot noise while
retaining frequency-selective multipath.

Data-symbol channel interpolation now separates the common carrier rotation
before interpolating complex channel values. Independent magnitude/phase
interpolation could bridge a selective fade with an unrealistically strong
channel estimate. Complex interpolation allows that cancellation while
preserving gain under a common phase rotation.

Detection confidence, pilot coherence, occupancy, and SNR still use raw pilots.
This separation matters: shrinking noisy pilot estimates changes the apparent
confidence distribution and otherwise rejects legitimate weak acquisitions.

The model receives unshrunk equalized latents and separate reliability weights.
Its decoder applies `z * weights` and also consumes the weights themselves;
the receiver must not apply the weights to the latents a second time. Beacon
soft decisions now use reliability weights because that decoder consumes the
decisions directly.

Pilot averaging and its noise/detail tradeoff are standard channel-estimation
tools; the acceptance evidence for this implementation is the recorded and
synthetic comparison below. See [MathWorks' channel-estimation description](https://www.mathworks.com/help/lte/ug/channel-estimation.html).

### Acquisition and stream identity

- Blind late-entry acquisition previously estimated CFO from real-audio CP
  correlation and coherently averaged pilots over twelve seconds. The real
  signal's conjugate image makes that CFO estimate ineffective, and even a
  small residual offset cancels the temporal pilot average. It now estimates
  fractional CFO from analytic audio, searches whole-carrier offsets using
  frequency coherence of the known pilots, then refines residual CFO. A
  matching mode and valid beacon CRC remain mandatory. Header-aided acquisition
  uses the corrected fractional-CFO estimator too.
- The selected model previously restricted header scoring to that mode alone.
  A strong other-mode header could therefore pass through its nonzero codeword
  correlation. All in-band modes are now compared. The selected mode remains a
  prior for weak headers, but a strong contradictory header is rejected.
- Loss of tracking now clears accumulated beacon chips and cached identity.
  A subsequent transmission cannot inherit the previous station's callsign.
  Accumulated beacon results must match the active mode.

### Timing and capture integrity

- The earlier guarded CP/pilot boundary correction remains enabled for Kiwi
  reception. It corrects the observed -33-sample jump without enabling the
  soundcard sample-clock loop. Details are in
  [the original reception investigation](kiwi-reception-20260906.md).
- Sample-clock phase-slope fits now require coherent supporting evidence and
  must fall within the tracking range. Unsupported fits neither remove apparent
  noise nor drive clock corrections. Diagnostics report an unknown rate when
  most pilot pairs do not support a fit.
- Ring-buffer writers previously modified samples outside the lock protecting
  reader copies and cursor publication. Wrapping could expose samples from
  different buffer generations. Sample writes, reads, and cursor publication
  now share the lock. A deterministic concurrency regression covers the race.
  This is a real defect; it is not proof that this race caused the recorded
  waveform jump.

## Recorded comparison

Replayed four raw Kiwi IQ recordings through the exact advertised sample rate,
512-sample input chunks, and 800-sample modem chunks. Acquisition uses only RX
data. Corresponding saved TX waveforms supply the clean scoring references.
The comparison disables only pilot denoising and the new interpolation in its
baseline; both sides include the timing/acquisition correctness fixes.

Symbol error is mean per-GOP normalized squared error of `latents * weights`
against the corresponding clean TX decode. It measures what the model receives
in its weighted latent input, not source-video fidelity. Model weights are
evaluated separately through actual ONNX decoder inference.

| Capture | GOPs, both variants | Previous NMSE | Updated NMSE | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| Latest | 45 / 45 | 0.6202 | 0.4886 | 21.2% |
| Timing jump | 30 / 30 | 0.2642 | 0.2029 | 23.2% |
| Additional 15 GOPs | 15 / 15 | 0.6848 | 0.5465 | 20.2% |
| Additional 20 m | 30 / 30 | 0.9033 | 0.7673 | 15.0% |

All 120 expected GOPs are retained. The timing-jump capture has exactly one
boundary correction; the other three have none. First beacon identification
is unchanged: GOP 38 in the latest capture (GOP 23 of its second transmission),
GOP 6 in the timing-jump capture, and no identification in the other two.

The installed `v8-hf3k-face-gan.decoder.onnx` was also run on 16 selected GOPs
(four per recording: the second, one-third, midpoint, and final GOP). Compared
with decoding the clean TX reference, output PSNR improves in 15 of 16 cases,
by approximately 0.28–3.68 dB. The final 20 m GOP regresses from 15.86 to
14.99 dB. These scores compare decoder outputs, not the original source video;
severe fades still produce visibly distorted output. The local contact sheet
is `runs/rx-review-20260906/model-comparison.png`, with all sampled scores in
`model-output.json` alongside it.

Capture filenames, under `C:\Users\patri\AETV\received\debug`:

| Capture | RX stem | Corresponding TX stems |
| --- | --- | --- |
| Latest, 45 GOPs | `20260906-151110-575_rx_kiwi` | `20260906-151107-371_tx_V8_VA7EET`, `20260906-151232-018_tx_V8_VA7EET` |
| Timing jump, 30 GOPs | `20260906-150809-198_rx_kiwi` | `20260906-150811-354_tx_V8_VA7EET` |
| Additional 15 GOPs | `20260906-151023-158_rx_kiwi` | `20260906-151025-339_tx_V8_VA7EET` |
| Additional 20 m, 30 GOPs | `20260906-150329-520_rx_kiwi` | `20260906-150333-685_tx_V8_VA7EET` |

## Regression coverage and remaining limits

Controlled late-entry probes decode at offsets 0, 0.4, 2, 20, -75, and 120 Hz
at unity and 0.0001 amplitude. Previously only the zero-offset probes decoded.
The new probes recover CFO within 0.00003 Hz on these clean synthetic signals;
this does not imply that precision on noisy RF. Regression tests also cover
all three bands, multipath-preserving denoising, common rotation and selective
fades, weak selected-mode acquisition, contradictory headers, beacon identity
reset, unsupported timing fits, buffer races, and resampler chunk continuity.
The final complete test suite passes: **254 tests**, in 119.41 seconds.
`git diff --check` also passes.

Known limitations:

- Weak beacon identification remains unresolved in the recordings. Reliability
  weighting does not advance first identification in these four captures.
- The separate ten-GOP 15:07 capture can still emit one extra weak tail GOP.
  A stricter CP cutoff also discarded genuine weak 20 m payloads, so it was
  not adopted. End-of-transmission discrimination needs independent evidence.
- The last four data symbols in each GOP still hold the final pilot estimate.
  Next-GOP pilot lookahead showed a smaller potential gain, but needs a
  separately validated streaming boundary and latency change.
- Applying the guarded soundcard timing loop to the latest Kiwi recording
  still emits 46 GOPs for 45 transmitted. The fit guard prevents unsupported
  phase slopes in the regression tests, but does not establish end-to-end
  robustness of that loop. It remains disabled for Kiwi. Soundcard RF replay
  and stronger recovery validation remain necessary.
- A cubic fractional resampler substantially improved isolated band-edge tone
  error but slightly worsened weighted-symbol error on all four recordings.
  The existing linear resampler is retained. Better standalone waveform error
  did not establish a better receiver/model input in these captures.
- These are offline V8 RF replays and synthetic regressions. They do not cover
  every channel, prove uniformly better subjective video, or replace a live
  RF test and packaged-build check. No model training or weight changes were made.

Local analysis scripts and outputs are in the ignored `runs` directory:
`validate_rx_review.py`, `rx_review_experiments.py`, `blind_rx_probe.py`,
`check_rx_model_output.py`, and `rx-review-20260906/`. Raw captures and the local
ONNX model are required to reproduce the recorded comparison; they are not
repository test fixtures.
