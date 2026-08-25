# Analog OFDM modem validation — 2026-08-25

## Result

The linear real-audio OFDM transport preserves the analog latent stream to
floating-point precision on an ideal channel in all three waveform bands. The
complete production GUI transmit path is not lossless: its intentional 0.5 dB
envelope clipper introduces 8.0–9.0% normalized latent MSE before the channel
emulator. This distinction was previously hidden by clean-loopback tests that
required only correlation greater than 0.95.

Two channel-emulator defects were found and repaired:

1. Watterson fading used a separate noncausal Hilbert transform on every GUI
   GOP chunk. Equivalent audio therefore received a different impairment when
   process-call boundaries moved. The emulator now uses a stateful causal FIR
   analytic-signal transform with continuous filter, tap, and delay state.
2. AWGN power for a fading profile was calculated from the already-faded
   waveform. The receiver noise floor therefore fell during a fade. It is now
   referenced to the clean transmitted signal power.

## Method

The validation uses deterministic unit-RMS Gaussian latents and the production
interleaver, real passband OFDM matrices, pilot equalizer, GUI continuous TX,
`StreamingChannelEmulator`, GUI per-chunk level control, and
`StreamingDemodulator`. Five seeds were run for V0/N, V1/W, and V7/U.

The ideal-linear check deliberately omits only the nonlinear clip/filter TX
conditioner and channel impairment. It still converts the complete latent
vector to real passband audio and recovers it through the production pilot
demodulator.

Run the complete deterministic matrix with:

```bash
python scripts/validate_analog_modem.py --seeds 5
```

Run the executable invariants with:

```bash
python -m pytest -q tests/test_analog_validation.py
```

## Ideal linear transport

| Mode / band | Decode rate | Mean NMSE | Correlation | Gain |
|---|---:|---:|---:|---:|
| V0 / N | 100% | 1.18e-15 | 1.000000 | 1.000000 |
| V1 / W | 100% | 1.81e-15 | 1.000000 | 1.000000 |
| V7 / U | 100% | 1.83e-15 | 1.000000 | 1.000000 |

This establishes that carrier placement, real/complex scaling, I/Q pairing,
interleaving, cyclic-prefix timing, and pilot equalization do not lose or mix
analog latent values on an ideal linear channel.

## Production GUI path

| Mode | Profile | Decode | NMSE | Correlation | Mean confidence | Estimated SNR |
|---|---|---:|---:|---:|---:|---:|
| V0 | clean | 100% | 0.080 | 0.965 | 1.000 | 37.3 dB |
| V0 | AWGN 12 | 100% | 0.119 | 0.942 | 0.964 | 11.3 dB |
| V0 | AWGN 6 | 100% | 0.251 | 0.867 | 0.869 | 5.3 dB |
| V0 | AWGN 0 | 100% | 1.245 | 0.541 | 0.663 | -0.6 dB |
| V0 | MPP 12 | 100% | 0.318 | 0.834 | 0.793 | 4.4 dB |
| V0 | MPP 6 | 100% | 1.035 | 0.611 | 0.699 | 1.2 dB |
| V0 | MPP 0 | 100% | 1.266 | 0.427 | 0.574 | -3.3 dB |
| V1 | clean | 100% | 0.090 | 0.972 | 1.000 | 40.3 dB |
| V1 | AWGN 12 | 100% | 0.150 | 0.928 | 0.942 | 11.8 dB |
| V1 | AWGN 6 | 100% | 0.399 | 0.790 | 0.799 | 5.8 dB |
| V1 | AWGN 0 | 100% | 1.573 | 0.404 | 0.583 | -0.2 dB |
| V1 | MPP 12 | 100% | 0.441 | 0.763 | 0.758 | 6.0 dB |
| V1 | MPP 6 | 100% | 0.978 | 0.543 | 0.642 | 2.1 dB |
| V1 | MPP 0 | 100% | 1.689 | 0.281 | 0.524 | -2.7 dB |
| V7 | clean | 100% | 0.084 | 0.965 | 1.000 | 53.0 dB |
| V7 | AWGN 12 | 100% | 0.435 | 0.786 | 0.807 | 11.5 dB |
| V7 | AWGN 6 | 100% | 1.728 | 0.421 | 0.592 | 5.5 dB |
| V7 | AWGN 0 | 100% | 2.151 | 0.177 | 0.470 | -0.3 dB |
| V7 | MPP 12 | 80% | 1.072 | 0.523 | 0.626 | 7.0 dB |
| V7 | MPP 6 | 80% | 1.839 | 0.275 | 0.504 | 1.8 dB |
| V7 | MPP 0 | 40% | 2.553 | 0.075 | 0.443 | -3.2 dB |

## Interpretation

The AWGN conditions have the expected monotonic effect in every band: lower
configured SNR increases latent NMSE and decreases both latent correlation and
receiver confidence. The pilot SNR estimator tracks the 12/6/0 dB labels to
within roughly 0.7 dB without fading.

MPP adds time/frequency-selective damage beyond AWGN at the same nominal SNR.
Low-confidence latent positions correlate with larger squared errors (the
six-seed V1/MPP12 invariant requires correlation greater than 0.20), showing
that the receiver weights identify selectively damaged analog values. The SNR
display reads below the nominal MPP label because pilot-to-pilot channel motion
is intentionally counted as receive uncertainty.

V7/U has worse per-latent performance at the same 2.5 kHz-reference SNR because
its normalized transmit power is spread across 160 carriers and real white
noise spans a 12 kHz Nyquist bandwidth. That is a consequence of the defined
reference-SNR convention, not an SNR calibration error.

The word `clean` should continue to mean "no channel impairment," not
"lossless end-to-end modem." If lossless GUI clean-loopback is a product
requirement, the 0.5 dB clipping contract must be changed or bypassed for that
profile; raising the headroom affects PAPR and the channel distribution used to
train existing checkpoints.
