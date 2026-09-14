# HackRF One throughput correction

HackRF One, clones and PortaPack units using standard One firmware do not have
HackRF Pro's filtered FPGA decimation/interpolation. The sample-rate divider
changes the converter clock; it is not an independent decimation stage.
See the [official gateware description](https://hackrf.readthedocs.io/en/latest/gateware.html)
and [sample-rate guidance](https://hackrf.readthedocs.io/en/latest/sampling_rate.html).

This change lowers One's converter/USB sample rate from 9.6 to 8.064 MS/s,
remaining above the recommended 8 MHz minimum. USB sample traffic falls 16%,
from 19.2 to 16.128 MB/s per active direction. RX host decimation by eight
produces 1.008 MS/s. No firmware change is required.

HackRF TX uses a periodic oscillator lookup and a shorter narrowband
interpolation filter. Signed 8-bit packing retains the previous rounding while
avoiding redundant allocations. RX uses a filter protecting +/-150 kHz around
baseband, including the offset signal and frequency-search range. Tests require
less than 0.002 dB passband ripple and greater than 80 dB alias/image rejection.
The RX filter does not promise alias protection across the entire output Nyquist
band. The waterfall still uses the original wideband samples.

TX primes fifteen 100 ms buffers before opening the stream, unless a shorter
source has already finished. This trades startup latency for tolerance of CPU
jitter and prevents the header alone from starting USB before payload encoding.
The queue remains bounded at sixteen buffers, underruns still fail visibly, and
RX overload still discards stale data and reacquires. TX health now records
conversion/packing timing and queue depth; RX overflow logs report conversion
cost. The trailing silence is 200 ms in every mode rather than accidentally
1.2 seconds for an 8 kHz source.

## Validation

Base revision: b3199498fc61a085c4bd5dc6f56e02add290c4a0 (v0.1.20).
Measured locally with one OpenBLAS/OMP thread, warm-up excluded:

| DSP and signed-IQ conversion | Before, s/signal second | After, s/signal second |
| --- | ---: | ---: |
| TX | 0.630 | 0.187 |
| RX | 0.523 | 0.199 |

These are host conversion measurements, excluding neural inference, USB and
hardware. They are not measurements of the reporter's Ryzen 1600.

Paired transport/reconstruction simulation covered V7, V8, AC16, V8 A/V and
AC16 A/V, six previously used validation GOPs per mode, clean and 8 dB waveform
AWGN. Both arms decoded every GOP in order. PSNR differences ranged from
-0.00174 to +0.00564 dB. This is waveform AWGN, not calibrated RF SNR; the paired
quality test uses known LO correction. Separate blind frequency acquisition
regressions cover 960 kS/s and the new 1.008 MS/s, at strong and noisy levels.

The full suite passed 478 tests with two skips. After adding the new-rate
acquisition cases, the focused stress suite passed all 50 tests. Other targeted
tests cover filter rejection, arbitrary buffer boundaries, exact I/Q rounding,
and a paced callback with delayed first payload. Evidence is in
[validation/hackrf-throughput-20260914](validation/hackrf-throughput-20260914).
Detailed local scripts and source snapshots are retained under
`/pool0/AETV-runs/hackrf-throughput-20260914`.

No HackRF is attached here. Physical One/clone/PortaPack streaming, long-duration
USB stability and Ryzen 1600 CPU inference remain unverified. In particular,
transport optimizations cannot guarantee real-time V7 CPU encoding/decoding.
