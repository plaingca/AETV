# AETV — Autoencoder Television

Analog video over HF. A spatiotemporal autoencoder maps each one-second group
of pictures to OFDM carriers; a ham station plays that audio into a radio and
the far end turns the recovered latents back into video.

This is the standalone Flex-8k AETV project. It ships the modem, the published
V7 checkpoint recipe, training and evaluation tools, and a ham-station GUI with
direct Hamlib control, native FlexRadio discovery/PTT/VITA-49 audio,
soundcard or KiwiSDR receive, webcam transmit, and a live audio waterfall.

## What was measured

Mode **V7** is 256×144 color at 12 fps in an 8 kHz channel (24 kHz
sample rate, 160 carriers). The published weights are the stage-2 no-GAN
checkpoint that was transmitted on 40 m and evaluated against VVC over
advanced single-tone / OFDM digital video paths. Clean codec PSNR on the
held-out OpenVid suite sits in the low-to-mid 20 dB range; the OTA 30-GOP
loopback of that same checkpoint recovered 20.98 dB mean PSNR with a
21.07 dB codec ceiling on animation.

AETV is analog: quality falls off as SNR falls. It does not cliff the way a
digital codec plus modem does.

## Install

You need Python 3.10+, [uv](https://docs.astral.sh/uv/), and FFmpeg on `PATH`.

```powershell
git clone https://github.com/plaingca/AETV.git
cd AETV
uv sync --extra gui
```

That extra is the redistributable station app: PyTorch, PortAudio,
OpenCV, PySide6, and the KiwiSDR client. CLI-only send/receive is
`--extra ham` (no Qt). Training extras (`--extra train`) add
torchvision, LPIPS, datasets, and Lance. GPU PyTorch is recommended
for V7 encode/decode; CPU works, slowly.

On an NVIDIA station, replace the default CPU wheel with the CUDA build after
the normal sync (choose the CUDA index supported by your driver; this checkout
has been tested with `cu130`):

```powershell
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu130
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Set **Torch device** to `cuda` in AETV Settings. The status bar includes the
GPU name when the checkpoint is actually running there.

Then launch the app:

```powershell
uv run aetv gui
# or, on Windows:
.\aetv-gui.ps1
```

On first use, AETV downloads the selected mode's approximately 206 MiB default
checkpoint from [AETV/AETV](https://huggingface.co/AETV/AETV), verifies its
SHA-256, and stores it in the per-user cache. See
[`models/README.md`](models/README.md) for alternate checkpoints, offline use,
and cache configuration.

## Send and receive

Identify with your own callsign. Confirm you are authorized for the
frequency, bandwidth, and power before keying.

```powershell
# Webcam -> Flex-8k WAV, then play locally
uv run aetv send --source webcam --callsign N0CALL --gops 10 --out tx.wav --play

# File -> waveform
uv run aetv send --source clip.mp4 --callsign N0CALL --gops 30 --out tx.wav

# WAV or soundcard capture -> video
uv run aetv receive --wav capture.wav --out rx.mp4 --display
uv run aetv receive --duration 35 --out rx.mp4

# Channel sim, no radio
uv run aetv simulate --source clip.mp4 --gops 4 --snr 12 --out sim.mp4
```

The GUI is the operator surface: receive on the left, compose and send
on the right, waterfall along the top. Choose a Hamlib model and radio
device—AETV loads Hamlib directly, with no `rigctld` to configure—or let it
discover a Flex 6000. Flex PTT and 24 kHz audio travel directly over the
SmartSDR TCP and VITA-49 UDP APIs, so SmartSDR and Windows DAX devices are
not required. Soundcards, serial-line PTT, and public KiwiSDR receive remain
available for other station layouts.

`aetv devices` lists soundcards. `aetv kiwi-list` reads the canonical
[KiwiSDR directory](http://rx.linkfanel.net/) and lists nearby receivers.
Selecting **Public KiwiSDR** in the GUI automatically ranks and selects the
best currently usable path, with the recommendation and confidence visible in
the Receive pane. **Path planner…** shows predicted and station-calibrated AETV
SNR over the next twelve hours. It uses
the official ITU-R P.533/P.372 runtime when installed and automatically saves
calibration residuals from matching-callsign OTA decodes. See
[`docs/propagation-planner.md`](docs/propagation-planner.md).
For rapid station calibration, **Receive → FT8 propagation calibration…** can
send an individually confirmed CQ on each selected band and import its
time/frequency-matched PSK Reporter SNR reports after five minutes.
Radio setup details are in
[`docs/ham-guide.md`](docs/ham-guide.md). Frequency and mode are checked,
never set for normal AETV transmission—tune the radio first. The separately
confirmed FT8 calibration action tunes its displayed Flex dial and DIGU mode.

The transmit pane's **Route** buttons also provide deterministic local modem
loopbacks. **Clean**, **12/6/0 dB**, and **MPP 12/6/0** bypass PTT and the audio
device, pass each completed modem chunk through the HF channel simulator, and
show recovered GOPs with measured pilot SNR in the Receive pane while later
GOPs are still being captured and encoded. **Radio** is the normal
on-air/audio-output route. SNR is signal power relative to white noise in a
2.5 kHz reference bandwidth; the 24 kHz V7 simulator includes the corresponding
12 kHz-to-2.5 kHz noise-bandwidth conversion.

## Train and eval

```powershell
uv sync --extra train
uv run aetv train -- --mode V7 --stage 2 --out runs/v7 --steps 10000 --amp
uv run aetv eval -- --checkpoint models/v8-flex8k-ota-rxfix.pt --out runs/eval --clips 8
```

The trainer's default `native` backend streams the full OpenVid-1M Lance
dataset with bounded, direct Lance scanners feeding a separate FFmpeg decode
pool. `--lance-batch-size` controls the number of encoded video rows per range
read, `--lance-fetch-threads` controls network concurrency, and `--threads`
controls FFmpeg concurrency. `--compile reduce-overhead` compiles the
encoder/decoder training hot path while keeping checkpoint keys
inference-compatible. The legacy `--data-backend viewer`
only sees the Dataset Viewer's currently converted window and is intended for
short smoke tests, not full epochs. Use `--init-checkpoint` to continue a
curriculum and `--reset-steps` when the objective changes. The published V7
checkpoint predates the corrected reference-bandwidth conversion: its stored
training labels are 11.86 dB below the equivalent physical SNR (for example,
the old training label 0 dB represents 11.86 dB in the UI and radio convention).
The exact severe-channel warm-start command and reference results are in
[`docs/v7-severe-finetune.md`](docs/v7-severe-finetune.md).

## Waveform

| Mode | Band | Video | Rate | Audio |
|---|---|---|---|---|
| V0 | N, 1.2 kHz | 64×48 | 6 fps | 8 kHz |
| V1 | W, 2.25 kHz | 96×72 | 6 fps | 8 kHz |
| V7 | U, ~8 kHz | 256×144 | 12 fps | 24 kHz |
| V8 | W, ~2.25 kHz | 192×108 | 6 fps | 8 kHz |

V7 is the published Flex-8k mode. Each GOP is exactly one RF second: 8 OFDM
frames and 10112 analog latents. Live station transmissions send lead-in,
preamble, and Golay mode header once, followed by back-to-back one-second GOPs
and one final lead-out. Protocol v4 uses a twelve-symbol repeated preamble and
eight repeated soft-scored Golay headers for calibrated low-SNR acquisition.
Thus 30 seconds of video occupies 30.65 seconds rather
than 40.2 seconds and sustains 12 fps. A receiver entering after the initial
header acquires symbol timing from cyclic prefixes, identifies pilot/frame
phase and the GOP boundary from the beacon, then jumps to the newest complete
GOP. Encoding, modulation, acquisition, and decoding all run incrementally.

V8 is the standard-channel counterpart: it trades half the frame
rate and one quarter of V7's latent throughput for 192×108 16:9 video on
450–2650 Hz carriers. Its default was trained on OpenVid-1M with the complete
stage-2 HF channel and localized face-detail fine-tuning. See
[`docs/v8-hf3k.md`](docs/v8-hf3k.md) for the waveform budget and commands.

## Credits

AETV began as the moving-video/Flex-8k work in Andrew Rodland's
[SSTVAE](https://github.com/arodland/SSTVAE) project and retains its analog
latent-over-OFDM approach, framing, Golay coding, and HF channel simulation.

The radio-autoencoder concept and two-stage training approach originate with
[FreeDV RADE](https://freedv.org/radio-autoencoder/), developed by David Rowe,
Jean-Marc Valin, and the FreeDV team. The compact spatiotemporal backbone also
takes architectural inspiration from
[LTX-Video](https://github.com/Lightricks/LTX-Video). Training uses
[OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M); the dataset and model
weights are not included in the source distribution and retain their own terms.
See [`NOTICE`](NOTICE) for the full attribution and distribution notes.

## Tests

```powershell
uv sync --extra dev
uv run pytest
uv build
```

The suite covers N/W/U numerology, clean modem loopbacks including V7 and V8,
checkpoint loading when the release model is present, the receive ring buffer,
Kiwi IQ-to-passband conversion, and fail-safe PTT.

## Codex worktrees

This repository includes a local Codex environment at
`.codex/environments/environment.toml`. Select **AETV development** when
starting a worktree task. Codex creates a worktree-local `.venv` from the
locked dependencies and exposes **Test**, **Build**, and **Run GUI** actions.

The ignored model checkpoints are not duplicated into every worktree. Tests
that require the default `models/v8-flex8k-ota-rxfix.pt` skip there; use a local
task when validating it or copy the checkpoint into that specific worktree.

## License

Artistic License 2.0. See `LICENSE` and `NOTICE`.
