# AETV — Autoencoder Television

Analog video over HF. A spatiotemporal autoencoder maps each one-second group
of pictures to OFDM carriers; a ham station plays that audio into a radio and
the far end turns the recovered latents back into video.

This repository is the standalone Flex-8k release of the AETV work that began
inside [SSTVAE](https://github.com/arodland/SSTVAE). It ships the modem, the
published V7 checkpoint recipe, training and evaluation, and a ham-station
GUI with CAT PTT, soundcard or KiwiSDR receive, webcam transmit, and a
live audio waterfall.

## What was measured

Mode **V7** is 256×144 color at 12 fps in an 8 kHz Flex DAX channel (24 kHz
sample rate, 160 carriers). The published weights are the stage-2 no-GAN
checkpoint that was transmitted on 40 m and evaluated against VVC over
advanced single-tone / OFDM digital video paths. Clean codec PSNR on the
held-out OpenVid suite sits in the low-to-mid 20 dB range; the OTA 30-GOP
loopback of that same checkpoint recovered 20.98 dB mean PSNR with a
21.07 dB codec ceiling on animation.

AETV is analog: quality falls off as SNR falls. It does not cliff the way a
digital codec plus modem does.

## Install

You need Python 3.10+, [uv](https://docs.astral.sh/uv/), FFmpeg on `PATH`,
and the Flex-8k weights in `models/v7-flex8k.pt` (see `models/README.md`).

```powershell
git clone https://github.com/arodland/AETV.git
cd AETV
uv sync --extra gui
```

That extra is the redistributable station app: PyTorch, PortAudio,
OpenCV, PySide6, and the KiwiSDR client. CLI-only send/receive is
`--extra ham` (no Qt). Training extras (`--extra train`) add
torchvision, LPIPS, datasets, and Lance. GPU PyTorch is recommended
for V7 encode/decode; CPU works, slowly.

Copy `models/v7-flex8k.pt` into place (see `models/README.md`), then:

```powershell
uv run aetv gui
# or, on Windows:
.\aetv-gui.ps1
```

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
on the right, waterfall along the top. Settings cover Hamlib `rigctld`,
Flex 6000 PTT, serial RTS/DTR, soundcards, webcam index, and a public
KiwiSDR picker (IQ centred on dial + 5 kHz so the 8 kHz V7 waveform
fits the 12 kHz stream).

`aetv devices` lists soundcards. `aetv kiwi-list` probes public
receivers. Flex DAX TX and CAT details are in
[`docs/ham-guide.md`](docs/ham-guide.md). Frequency and mode are checked,
never set — tune the radio first.

## Train and eval

```powershell
uv sync --extra train
uv run aetv train -- --mode V7 --stage 2 --out runs/v7 --steps 10000 --amp
uv run aetv eval -- --checkpoint models/v7-flex8k.pt --out runs/eval --clips 8
```

The trainer streams OpenVid-1M. Use `--init-checkpoint` to continue a
curriculum and `--reset-steps` when the objective changes.

## Waveform

| Mode | Band | Video | Rate | Audio |
|---|---|---|---|---|
| V0 | N, 1.2 kHz | 64×48 | 6 fps | 8 kHz |
| V1 | W, 2.25 kHz | 96×72 | 6 fps | 8 kHz |
| V7 | U, ~8 kHz | 256×144 | 12 fps | 24 kHz |

V7 is the published Flex-8k mode. Each GOP is one second: 8 OFDM frames,
10112 analog latents, a Golay header, and a repeating callsign beacon.

## Tests

```powershell
uv run pytest tests/test_core.py tests/test_station.py
```

The suite covers N/W/U numerology, a clean modem loopback including V7,
the receive ring buffer, Kiwi IQ-to-passband conversion, and fail-safe PTT.

## License

Artistic License 2.0. See `LICENSE` and `NOTICE`.
