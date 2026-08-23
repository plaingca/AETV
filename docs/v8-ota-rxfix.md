# V8 OTA receiver correction

The default checkpoint is `models/v8-flex8k-ota-rxfix.pt`, SHA-256
`294987591b8ece1cb6fd6ad10349a160192e4e6fefc26d47bbbefd9cce9a778f`.
It is a 250-step adaptation of V8 OTA-perceptual to the corrected receiver
confidence and equalization contract.

## Receiver corrections

- Pilot SNR subtracts estimated noise power instead of reporting signal plus
  noise, removing the former 5.05 dB noise-only floor.
- Payload confidence is H^2/(H^2+N), and therefore falls with pilot
  uncertainty instead of remaining near 0.89 in noise.
- Channel estimates interpolate toward the following frame pilot, removing up
  to 100 ms of avoidable pilot age.
- Continuous acquisition requires independent payload-pilot confirmation
  before a GOP is displayed, and weak tracked GOPs are dropped rather than
  released after recovery.

The same estimator/equalizer is implemented in the differentiable stage-2
training channel.

## Recorded OTA replay

The 2026-08-23 71-GOP Kiwi capture was paired with its exact saved TX waveform.
The corrected SNR spanned -3.49 through +12.49 dB. Against the clean transmitted
decode, the receiver correction improved mean PSNR from 21.71 to 23.79 dB and
reduced effective-latent NMSE from 1.625 to 0.638.

| Corrected SNR segment | Legacy RX PSNR | Fixed RX PSNR |
|---|---:|---:|
| 8.48 dB | 24.94 | 26.91 |
| 10.27 dB | 26.69 | 28.47 |
| 5.52 dB | 21.96 | 24.42 |
| -0.60 dB | 15.19 | 17.24 |

The adapted checkpoint then improved the fixed-receiver replay from 23.79 to
24.16 dB overall, with gains in every segment. Its five-clip corrected-channel
evaluation gained 0.25 dB at 6 dB and 1.10 dB at 0 dB, while losing 0.21 dB
clean. Full replay data and videos are under `runs/ota-redecode/`.
