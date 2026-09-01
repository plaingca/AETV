# Analog composite power and PAPR audit

Date: 2026-08-25

Measured artifact: `runs/simpsons-analog-channel-60s/composite_transmit.wav`.

## Power allocation

`compose_delayed_stream` independently normalizes the complete active speech and AETV branches to RMS 0.22, sums them, and then applies one global peak scale. It does not normalize per GOP, so it does not pump the levels at GOP boundaries.

Over the 59-second steady overlap interval:

- voice RMS: 0.11565, -18.74 dBFS, 49.88% of composite power
- upper AETV RMS: 0.11590, -18.72 dBFS, 50.09% of composite power
- voice/AETV ratio: -0.019 dB
- filter/cross-term residue: 0.03%
- composite RMS: 0.16376, -15.72 dBFS
- real-sample peak: 0.94998, -0.45 dBFS

Thus the current prototype is effectively a 50/50 average-power split. This is a whole-branch allocation, not 50% for every OFDM carrier. Within the W branch, the 44 latent carriers share almost all AETV power; pilots occupy every fifth symbol and the beacon has its own carrier.

## Crest factor and PAPR

For an SSB transmitter, the analytic-envelope value is the relevant PEP-to-average ratio:

- steady analytic-envelope PAPR: 13.08 dB
- complete 61-second PAPR: 13.14 dB
- real-sample crest factor: 15.27 dB steady / 15.33 dB complete
- 99% envelope level: 7.10 dB above RMS
- 99.9%: 9.17 dB
- 99.99%: 10.86 dB
- 99.999%: 12.14 dB

The real PCM samples do not clip, but the analytic envelope peaks at 1.044. A downstream SSB modulator or ALC can therefore reach envelope limiting even though the audio file itself peaks at 0.95.

At a 100 W PEP limit, 13.08 dB PAPR permits only about 4.9 W total average power. With the current equal split, voice and AETV each average about 2.45 W. Raising the average to 25 W would demand roughly 510 W PEP and necessarily compress a 100 W transmitter.

## Current implementation caveat

The experiment builds V8 payloads with `_payload_wave`, translates them, masks the upper slice, and combines them with speech. It bypasses the production `_ContinuousTxConditioner`. Therefore the final mixture is peak-scaled but not crest-factor reduced.

Applying the existing 0.5 dB AETV conditioner only to the upper branch does not solve the composite PAPR: measured PAPR remains 13.18 dB because speech peaks and cross-service addition dominate. It also introduces 0.094 latent NMSE relative to the unconditioned waveform. A composite-aware CFR design is needed instead of simply reusing the data-only clipper.

## Ideal envelope-limiter sensitivity

A single-pass hard complex-envelope limiter was applied to the complete waveform. This is a diagnostic bound, not a calibrated transmitter model.

| Limiter threshold above mean envelope power | Samples limited | Latent NMSE vs unlimited | Latent corr. | Voice SI-SDR | Voice STOI |
|---:|---:|---:|---:|---:|---:|
| 12 dB | 0.0018% | 2.0e-7 | 1.000000 | 66.7 dB | 1.0000 |
| 10 dB | 0.0346% | 3.2e-5 | 0.999984 | 44.2 dB | 0.9997 |
| 8 dB | 0.412% | 4.9e-4 | 0.999758 | 33.1 dB | 0.9969 |
| 6 dB | 2.54% | 3.9e-3 | 0.998089 | 24.8 dB | 0.9810 |
| 4 dB | 8.83% | 1.75e-2 | 0.991384 | 19.0 dB | 0.9434 |

The waveform does not instantly collapse under occasional ideal envelope limiting. Real ALC, speech compression, PA memory, bias movement, IMD, and transmit filters can be substantially less benign, especially because intermodulation couples voice into the OFDM slice and data into the voice slice.

## OTA operating implication

This should be treated like a wide digital waveform, not ordinary processed SSB speech:

- use a flat approximately 5 kHz transmit/receive passband;
- disable speech compression, EQ, noise reduction, and aggressive ALC;
- set drive from PEP, with approximately 13 dB average-power backoff for the current waveform;
- keep any external amplifier in linear, continuous-duty service;
- verify two-tone/IMD and occupied bandwidth on a dummy load before OTA use.

A standard 2.7 kHz SSB filter will fail regardless of PA linearity because it removes the upper AETV slice. A 5 kHz-capable, linear transmitter operated around 13 dB below PEP should reproduce this prototype without becoming a distorted mess; driving it to normal speech-like average power will not.
