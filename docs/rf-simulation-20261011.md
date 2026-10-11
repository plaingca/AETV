# Cloud RF simulation validation — 2026-10-11

## Scope and reproduction

This is an actual execution of the production AETV real-audio OFDM modem, with
synthetic unit-RMS Gaussian latent payloads. It does not load learned video
weights, evaluate visual quality, key a radio, or prove over-the-air operation.

Current release modes come from `aetv.config.RELEASE_MODES`: V8 (8 kHz sample
rate), V9 (12 kHz), V7 (24 kHz), and experimental AC16 (48 kHz). The AC16 label's
16 kHz refers to channel width, not the audio sample rate. These tests use each
mode's actual geometry; the historical single 16 kHz/10 dB assumption is not used.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/validate_rf_simulation.py --seeds 3 --gops 4 \
  --output docs/validation/rf-simulation-20261011.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_rf_simulation.py tests/test_analog_validation.py
```

NumPy/SciPy version information and every individual GOP measurement are in the
JSON report. Limiting BLAS threads avoids severe oversubscription in this cloud
container. `--modes`, `--scenarios`, `--seeds`, and `--gops` support smaller
reproductions. Seeds/GOP counts are constrained to 1–100 to reject accidental
unbounded inputs.

## Method

Each trial transmits four independently generated GOPs using
`modulate_continuous_chunks`, including the real preamble, mode header, pilots,
beacon, interleaver, production TX clipping and filtering, and continuous TX
conditioner state. The waveform is impaired as a whole, then received by
`StreamingDemodulator` with sample-clock and boundary tracking enabled and
alternating awkward callback sizes of floor(fs/23), floor(fs/11), floor(fs/7).
No source timing, latent samples, or channel taps are supplied to the receiver.

Recovered GOPs are associated with the transmitted source only **after** decoding
by maximum correlation over the four distinct source vectors, requiring >0.25.
The report counts unique matched sources, unmatched outputs, normalized latent
MSE, correlation, confidence, estimated SNR, and receiver wall time. MSE is not
corrected by fitting gain or bias. A recovered GOP does not imply acceptable
video quality: analog errors can be severe despite synchronization succeeding.

The 15 scenarios, each run with seeds 0, 1, and 2:

- Clean and AWGN at 12, 6, and 0 dB.
- Historical Butterworth two-path Rayleigh MPP (2 ms delay / 1 Hz nominal
  Doppler) at 12 and 6 dB; MPD (4 ms / 2 Hz) at 12 dB.
- Separate Gaussian-Doppler MPP at 12 dB (2 ms / 1 Hz two-sigma spread).
- +25 and −25 Hz carrier offsets at 12 dB AWGN.
- +100 and −100 ppm sample-clock skew at 12 dB AWGN, using rational polyphase
  resampling (positive means more received samples).
- Additional hard clipping at ±1 waveform RMS, with 12 dB AWGN.
- A sinusoidal interferer 137 Hz above the mode center, signal/interferer ratio
  6 dB, plus 12 dB AWGN. Exact tone frequencies are V8=1737 Hz,
  V9=3137 Hz, V7=5137 Hz, AC16=8137 Hz, with zero initial phase.
  Tone amplitude is sqrt(2) × RMS(clean conditioned whole TX waveform)
  × 10^(-6/20); thus the 6 dB SIR uses pre-channel whole-waveform average
  signal power, including acquisition and silence, not post-fade power.
- Combined historical MPP, 12 dB AWGN, +25 Hz carrier offset, +100 ppm skew.

AWGN is referenced to pre-fade transmit power in **2500 Hz**, consistently with
`hfchannel.emulate`; equal numeric SNR is not equal in-channel noise across
mode bandwidths. Channel order is extra clipping, fading/noise, CFO, interferer,
then clock resampling. There is no post-channel AGC. Whole-waveform historical
fading is intentionally distinct from the GUI's sum-of-sinusoids streaming
emulator; both retain the repository's established definitions.

## Sweep results

180 trials transmitted **720 GOPs**, with four distinct payloads within each
trial (reused across scenarios for paired comparisons). Each cell is matched
unique sources / 12 sent, followed by mean NMSE over matched outputs only.
The correlation >0.25 association gate is not a video-quality threshold.
Rows with low recovery can have selection-biased NMSE; consult the raw JSON.

| Scenario | V8 | V9 | V7 | AC16 |
|---|---:|---:|---:|---:|
| clean | 12/12; 0.091 | 12/12; 0.081 | 12/12; 0.073 | 12/12; 0.093 |
| awgn12 | 12/12; 0.123 | 12/12; 0.141 | 12/12; 0.206 | 12/12; 0.385 |
| awgn6 | 12/12; 0.223 | 12/12; 0.323 | 12/12; 0.615 | 12/12; 1.972 |
| awgn0 | 12/12; 0.661 | 12/12; 1.109 | 5/12; 4.034 | 0/12; — |
| mpp12 | 12/12; 0.371 | 12/12; 1.143 | 8/12; 1.498 | 11/12; 3.583 |
| mpp6 | 12/12; 2.083 | 11/12; 1.962 | 6/12; 4.560 | 2/12; 7.827 |
| mpd12 | 12/12; 0.861 | 12/12; 1.184 | 11/12; 2.501 | 7/12; 3.518 |
| gaussian_mpp12 | 12/12; 0.541 | 12/12; 0.624 | 12/12; 1.163 | 11/12; 2.412 |
| offset_plus25 | 12/12; 0.123 | 12/12; 0.141 | 12/12; 0.206 | 12/12; 0.385 |
| offset_minus25 | 12/12; 0.123 | 12/12; 0.141 | 12/12; 0.206 | 12/12; 0.385 |
| clock_plus100 | 12/12; 0.124 | 12/12; 0.142 | 12/12; 0.207 | 12/12; 0.392 |
| clock_minus100 | 12/12; 0.124 | 12/12; 0.141 | 12/12; 0.207 | 12/12; 0.403 |
| clip_1rms | 12/12; 0.182 | 12/12; 0.208 | 12/12; 0.267 | 12/12; 0.443 |
| tone_sir6 | 12/12; 0.264 | 1/12; 0.266 | 0/12; — | 0/12; — |
| combined | 12/12; 0.393 | 12/12; 0.534 | 8/12; 2.013 | 11/12; 3.132 |

| Mode | Receiver outputs | Matched unique source GOPs | Unmatched outputs |
|---|---:|---:|---:|
| V8 | 180 | 180/180 | 0 |
| V9 | 169 | 168/180 | 1 |
| V7 | 155 | 146/180 | 9 |
| AC16 | 144 | 138/180 | 6 |

All clean trials recovered all source GOPs; their nonzero NMSE reflects the
intentional production transmit conditioner. Carrier offset ±25 Hz and clock
skew ±100 ppm with 12 dB AWGN also recovered all source GOPs across modes.
The in-band tone prevented acquisition in most wide-mode trials. Low SNR and
multipath caused substantial latent corruption and some lost/unmatched GOPs.
These are measured limits under these exact scenarios, not proven regressions
against a prior revision and not a basis for changing release defaults.

## Executable regressions

`tests/test_rf_simulation.py` plus `tests/test_analog_validation.py`: **18 passed**
in the final run (16.52 seconds with single-thread BLAS). No modem acquisition
thresholds or release defaults were changed.

## Regression fix

`hfchannel.fading` previously constructed a delayed signal longer than the
input when path delay exceeded buffer length. For an 8-sample input at 8 kHz
with MPP's 16-sample delay, array multiplication failed. The delayed path now
uses a same-sized zero-filled array, and empty input returns an empty float
array. Tests cover empty, one-sample, shorter/equal/longer-than-delay buffers,
zero path delay, deterministic impairments, and clean two-GOP reception in all
four current release modes.

## Limits

Only three seeds and four GOPs per trial are sampled. These are bounded
regression/stress observations, not statistical reliability guarantees. There
is no real RF front end, soundcard, rig, RF amplifier, band-limiting radio
filter, sustained clock-drift run, model reconstruction, lip synchronization,
or RF regulatory evaluation. The synthesized interferer and clipping are
controlled stressors rather than calibrated hardware models. Historical fading
normalizes tap energy over the short trial and is not a claim of strict
standards conformance.

## In-band interferer diagnosis

V9, `tone_sir6`, seed 1 produced no receiver output. A separate diagnostic run
with the receiver's `on_debug` callback shows the true preamble candidate at
sample 1400: metric 0.844, header score 0.616, pilot coherence 0.744, but pilot
occupancy 0.360. It is rejected by the existing 0.40 pilot-bandwidth occupancy
floor. The subsequent searches fail to establish lock. Full debug evidence is
in `validation/rf-v9-tone-diagnostic-20261011.json`.

This identifies an interference-tolerance limit in acquisition, not a crash.
The presence gate also rejects non-OFDM signals, so its threshold was not
lowered to make this one fixture pass. Improving robust occupancy estimation
would require both interferer sweeps and no-signal/tone-only false-positive
validation; the clean regression fix does not claim to solve this limitation.
