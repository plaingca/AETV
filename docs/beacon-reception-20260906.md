# Weak beacon reception — September 6, 2026

The receiver now identifies both 15-second transmissions that previously
never produced a callsign. Both identify VA7EET at GOP 12. This is a receive-only
change: the beacon waveform, payload layout, Golay code, and CRC are unchanged.

## What was losing sensitivity

The 78-bit beacon payload occupies seven 12-bit Golay message words. The last
word includes six known zero-padding bits, but the decoder previously searched
all 4096 messages and discarded the padding afterward. Decoding within the
valid padded codebook repairs the additional 15-second recording: its second
beacon had six correctly decoded words and one incorrect final word.

The latest short transmission has a different failure. Each of its first two
beacons contains damaged words, so independently choosing the best Golay word
never produces a valid CRC. Four whole Golay words contain only callsign/mode
bits and repeat unchanged. Combining those observations repairs much of the
identity, while a bounded soft-candidate search handles uncertainty in the
remaining words.

## Acceptance checks

The fast path still uses a single best decode and a valid CRC, with the selected
mode checked afterward. Forcing that mode during a single-frame decode raised
false identification in a noise stress test and was not retained. The fallback
requires two complete frames with Barker sync peaks separated by exactly one
or two beacon periods. Only the invariant whole words are combined; each
frame retains its own received counter and CRC words.

Soft-candidate search is bounded to 256 hypotheses per frame, with at most
eight frame pairs examined. A list candidate cannot identify a station alone.
Two candidates must independently pass their CRCs, agree on callsign and mode,
and have counters consistent with the measured chip spacing, including
counter wraparound. Ambiguous matches are rejected. The selected RX mode is
available to the decoder, and scanning continues past wrong-mode beacons.

The caller's existing loss-of-tracking reset still clears beacon history, so
the short transmission's recovered identity is not inherited by the next
transmission.

## Recorded results

Compared against the receiver after the earlier pilot/equalizer review:

| Transmission | Before | Updated |
| --- | --- | --- |
| Latest capture, first 15 GOPs | No identification | GOP 12 |
| Latest capture, following 30 GOPs | GOP 23 | GOP 23 |
| Earlier 30-GOP timing-jump recording | GOP 6 | GOP 6 |
| Additional 15-GOP recording | No identification | GOP 12 |
| Additional weak 20 m recording | No identification | No identification |

Four complete raw-IQ replays retain all 120 expected GOPs. Every recovered
video latent and reliability weight is bit-for-bit identical to the previous
receiver. No wrong callsigns are emitted in these recordings. Acquisition and
tracking event counts are unchanged.

Capture filenames and the earlier symbol measurements are in
[the RX pipeline review](rx-pipeline-review-20260906.md). These are offline
results; no packaged build or live RF test has been performed.

## Validation artifacts

Regression tests cover padding evidence, Golay score batching, repeated weak
frames, gain and polarity, counter wraparound, station/counter disagreement,
wrong-mode scanning, CRC lookup equivalence, nonfinite input, and random
payloads with deliberately perfect repeated sync.

The complete suite passed 270 tests. After the final single-frame acceptance
tightening, all 149 tests in the beacon/core/station/Kiwi receiver suites passed
again. `git diff --check` passes.

An independent synthetic benchmark uses varying callsigns, modes, and counters,
two complete beacon frames, random leading chip offsets, additive chip noise,
and a 24-chip erasure on half the trials. Noise sigma is relative to unit BPSK
chips; this is a logical-beacon test, not an RF SNR calibration.

| Chip-noise sigma | Previous correct / 200 | Updated correct / 200 |
| --- | ---: | ---: |
| 0.65 | 199 | 200 |
| 0.85 | 130 | 196 |
| 1.00 | 40 | 151 |
| 1.15 | 1 | 75 |

Neither decoder emitted a wrong identity in those 800 trials. A separate
5,000-window noise test, half with deliberately perfect repeated Barker sync,
produced zero identifications with either the previous or final decoder.
This is a finite stress test, not a guarantee of zero false detections.

The local `runs/beacon-review-20260906/` directory contains extracted soft
chips, full replay events, and benchmark results. The corresponding scripts
are `runs/cache_beacon_review.py`, `runs/validate_beacon_review.py`, and
`runs/benchmark_beacon_review.py`. These local RF artifacts are ignored by Git;
the synthetic regression tests are in the repository.
