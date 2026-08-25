<p align="center">
  <img src="aetv/assets/aetv-logo.png" alt="AETV logo" width="280">
</p>

<h1 align="center">AETV</h1>

<p align="center"><strong>Live video that keeps going when the radio channel gets ugly.</strong></p>

<p align="center">
  <a href="https://github.com/plaingca/AETV/actions/workflows/ci.yml"><img src="https://github.com/plaingca/AETV/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/plaingca/AETV/actions/workflows/release-packages.yml"><img src="https://github.com/plaingca/AETV/actions/workflows/release-packages.yml/badge.svg" alt="Release packages"></a>
</p>

AETV sends moving color video over amateur-radio audio paths. It uses a learned
video codec and an OFDM waveform designed to fade gracefully as noise and
multipath increase—more like analog television than an all-or-nothing video
call.

The station app handles camera or file playback, reception, soundcards,
FlexRadio network audio, CAT/PTT, public KiwiSDR monitoring, and local channel
simulation from one window.

## Two release modes

| Mode | Best for | Picture | Radio audio |
|---|---|---:|---:|
| **Standard channel** | Typical HF/VHF SSB paths and CPU-only stations | 192×108 at 6 fps | Fits a standard transmit channel |
| **Wide 8 kHz** | Higher-detail links through FlexRadio or other wide audio paths | 256×144 at 12 fps | 8 kHz occupied bandwidth, 24 kHz audio |

Only these checksum-verified modes appear in the release GUI. Both stations
must select the same mode.

## See it over the channel

These are held-out evaluation clips from the validated Wide 8 kHz receiver
model. Each grid shows the source, clean loopback, 18/12/6/0 dB paths, and two
multipath fading profiles. Click a frame to play the one-second clip.

| Moving water | Forest tracking | Fine detail |
|---|---|---|
| [![Ocean evaluation](docs/demos/ocean-motion.png)](docs/demos/ocean-motion.mp4) | [![Forest evaluation](docs/demos/forest-motion.png)](docs/demos/forest-motion.mp4) | [![Flower evaluation](docs/demos/flower-motion.png)](docs/demos/flower-motion.mp4) |

The Wide 8 kHz model was also exercised over a 40 m FlexRadio path. A 71-GOP
off-air receiver-validation sweep and the full measurement notes are in
[the performance report](docs/performance.md).

## Download and run on Windows

1. Download the current Windows zip from
   [Releases](https://github.com/plaingca/AETV/releases).
2. Extract it anywhere.
3. Run `AETV.exe`.

The portable builds include Python, the app libraries, and both validated model
files. It does not install Python or download weights on first launch. Windows
may show a SmartScreen prompt until release binaries are code-signed.

FlexRadio control, serial PTT, VOX/manual PTT, soundcards, and the models are
self-contained. The optional **Hamlib — connect directly** backend still needs
a Hamlib 4.x DLL beside the app until that third-party runtime is cleared for
redistribution in AETV releases.

Choose the CPU build for the Standard channel mode. Choose the CUDA build for
Wide 8 kHz or for extra processing headroom; it still falls back to CPU when a
compatible NVIDIA GPU is unavailable.

Linux, Windows CPU, and Windows CUDA packages are built and smoke-tested on
GitHub Actions workers. Run the **Release packages** workflow manually for
short-lived downloadable artifacts; version tags such as `v0.1.0` publish the
files to GitHub Releases. Because GitHub limits each Release asset to 2 GiB, the
larger CUDA zip is published in numbered parts with PowerShell reassembly
instructions, checksums for the downloaded parts, and a checksum for the
reconstructed zip.

## What hardware works?

One GOP represents one second on air, so encode or decode must finish inside
one second for live operation.

Measured on a Ryzen 7 5800X (8 cores/16 threads):

| Release mode | Encode | Decode | Result |
|---|---:|---:|---|
| Standard channel, CPU | 414 ms | 629 ms | Real-time half-duplex transmit and receive |
| Wide 8 kHz, CPU | 1,182 ms | 1,962 ms | Not real time |

Measured on an RTX 4080:

| Release mode | Encode | Decode |
|---|---:|---:|
| Standard channel, CUDA | 13 ms | 19 ms |
| Wide 8 kHz, CUDA | 34 ms | 71 ms |

For CPU-only use, a recent 8-core/16-thread desktop is a practical baseline for
Standard channel mode. Slower machines can still decode recordings, but may not
keep up live. Wide 8 kHz should be treated as a CUDA mode for now. Run the
included benchmark on another rig for a direct answer:

```powershell
AETV-Benchmark.exe --mode V8 --device cpu
```

Detailed methodology and current machine results live in
[docs/performance.md](docs/performance.md).

## First contact

- Enter your callsign in **Settings**.
- Pick **Standard channel** unless both ends have an 8 kHz audio path.
- Select the radio input/output devices and configure CAT/PTT if desired.
- Use **Local loopback** first. It exercises the complete modem without keying a radio.
- Confirm your licence, regional band plan, occupied bandwidth, frequency, and power before transmitting.

AETV identifies transmissions, but it cannot decide whether a frequency or
bandwidth is legal at your station.

## Run from source

Source installs are for development; most operators should use the Windows
release.

```powershell
git clone https://github.com/plaingca/AETV.git
cd AETV
uv sync --extra gui
uv run aetv-gui
```

Without a bundled release model, the source build downloads the selected pinned
checkpoint from [AETV/AETV on Hugging Face](https://huggingface.co/AETV/AETV)
and verifies its size and SHA-256 hash. Set `AETV_OFFLINE=1` to forbid network
model downloads or `AETV_MODEL_DIR` to choose the cache location.

## Build, test, and contribute

```powershell
uv sync --extra gui --extra dev
uv run pytest -q
uv run python scripts/benchmark_inference.py --mode V8 --device cpu
./scripts/build_windows.ps1 -Runtime cpu
./scripts/build_linux.sh
```

The build creates a clean Python environment, fetches and verifies both pinned
models when needed, bundles the runtime, and smoke-tests the packaged Standard
model in offline mode before producing the zip. Use `-Runtime cuda` for the
larger NVIDIA build.

The modem contract, model experiments, OTA notes, training commands, and
hardware integrations are kept in [docs](docs/) so the main page can stay
focused on operating the software. Start with the
[ham operator guide](docs/ham-guide.md), [model index](models/README.md), and
[propagation planner](docs/propagation-planner.md).

## Licence

AETV is released under the [Artistic License 2.0](LICENSE). Third-party notices
are collected in [NOTICE](NOTICE).
