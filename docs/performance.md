# Performance and hardware guide

## Can it run live?

AETV transmits one video GOP per second. The station is half duplex, so live
operation requires the active direction—encode while transmitting or decode
while receiving—to complete in less than one second. A combined encode+decode
time is useful stress data, but it is not the normal radio workload.

Release-model medians on Windows 11, PyTorch 2.13, after two warmup runs:

| Hardware | Mode | Encode | Decode | Combined | Peak model workload |
|---|---|---:|---:|---:|---:|
| Ryzen 7 5800X, 16 threads | Standard channel | 414 ms | 629 ms | 1,041 ms | CPU |
| Ryzen 7 5800X, 16 threads | Wide 8 kHz | 1,182 ms | 1,962 ms | 3,227 ms | CPU |
| RTX 4080 | Standard channel | 12.5 ms | 19.1 ms | 33.8 ms | 403 MB CUDA allocated |
| RTX 4080 | Wide 8 kHz | 33.9 ms | 71.5 ms | 105.3 ms | 1,362 MB CUDA allocated |

The Standard channel model is therefore viable on a modern 8-core/16-thread
desktop CPU for normal half-duplex use. It is close enough to the deadline that
background load, laptop power limits, and slower memory can matter. Wide 8 kHz
is not real time on this CPU and should use CUDA.

CUDA memory figures are peak tensor allocations during inference, not total GPU
process usage. Driver/runtime overhead and camera/display buffers require extra
room. The measurements support a conservative recommendation of a modern
CUDA-capable NVIDIA card with at least 4 GB VRAM; only the RTX 4080 has been
timed here, so slower cards should be checked with the benchmark before going
on air.

### Reproduce the runtime test

```powershell
uv run python scripts/benchmark_inference.py --mode V8 `
  --checkpoint models/v8-hf3k-face-gan.pt --device cpu --threads 16

uv run python scripts/benchmark_inference.py --mode V7 `
  --checkpoint models/v8-flex8k-ota-rxfix.pt --device cuda
```

The benchmark uses deterministic random input, warms up the model, synchronizes
CUDA around every sample, and reports medians. Set `AETV_CPU_THREADS` to override
the app's default of using all logical processors.

## Standard channel validation

The release checkpoint is `v8-hf3k-face-gan.pt` (SHA-256
`f218376af9f9916050c9e345353da0c0970c392f58755efaa81d01e7ded8fc40`).
A production-GUI loopback at a nominal 12 dB recovered 1/1 GOP and measured
27.22 dB PSNR, 0.862 SSIM, and 0.059 LPIPS on the held-out validation clip. The
displayed pilot SNR was 11.1 dB. That run also passed all 11 targeted modem/GUI
contract tests.

The complete captured result is retained at
`runs/gui-loopback-validation-20260824/validation.json` in the research tree.

## Wide 8 kHz channel validation

The release checkpoint is `v8-flex8k-ota-rxfix.pt` (SHA-256
`294987591b8ece1cb6fd6ad10349a160192e4e6fefc26d47bbbefd9cce9a778f`).

In a 71-GOP recorded KiwiSDR replay paired to the exact saved transmit waveform,
the corrected receiver covered -3.49 to +12.49 dB SNR. Receiver corrections
raised mean PSNR from 21.71 to 23.79 dB; adapting the release checkpoint to that
receiver raised it again to 24.16 dB. Gains were present in every measured SNR
segment.

| Corrected SNR segment | Earlier receiver | Release receiver |
|---|---:|---:|
| 10.27 dB | 26.69 dB | 28.47 dB |
| 8.48 dB | 24.94 dB | 26.91 dB |
| 5.52 dB | 21.96 dB | 24.42 dB |
| -0.60 dB | 15.19 dB | 17.24 dB |

The three clickable clips in the main README come from this receiver-adapted
checkpoint's held-out evaluation and show clean, AWGN, and multipath cases.
See [v8-ota-rxfix.md](v8-ota-rxfix.md) for the equalizer and confidence changes.

## Interpretation

PSNR, SSIM, and LPIPS describe different aspects of reconstruction and should
not be read as a single universal quality score. The most useful release gate
is end-to-end recovery through the production modem under the intended channel,
followed by visual review of motion, faces, texture, and failure behavior. The
demo panels keep low-SNR failures visible for that reason.
