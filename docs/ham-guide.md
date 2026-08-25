# Operating AETV on HF

AETV is analog video. There is no bitstream and no FEC in the usual digital
sense. The decoder reconstructs pictures from noisy latents, so a weak copy
looks soft rather than blocky or silent.

## Legal

- You must hold a license that authorizes the band, power, and emission.
- The beacon carries an 8-character callsign. Set `--callsign` to yours.
- Flex-8k occupies about 8 kHz of audio. That is not a standard 2.7 kHz SSB
  channel. Use a wide digital slice (DIGU) and stay inside the band plan.
- Standard channel (protocol mode V8) occupies about 2.25 kHz on 450-2650 Hz
  audio carriers and is intended for a nominal 3 kHz SSB channel.
- Yesterday's clear frequency is not a check today. Listen first.

## Station app

```powershell
uv sync --extra gui
uv run aetv gui
```

The AETV station window has a receive waterfall across the top, decoded video
on the left, and webcam or file transmit on the right. Settings persist under
`%APPDATA%\AETV\settings.json` on Windows and, following the XDG configuration
location, `$XDG_CONFIG_HOME/AETV/settings.json` or
`~/.config/AETV/settings.json` elsewhere.

**Receive** can come directly from a FlexRadio over VITA-49, from a local
soundcard or from a public KiwiSDR. **Find Kiwis** reads the canonical
[KiwiSDR directory](http://rx.linkfanel.net/) and shows nearby receivers
with their advertised API availability — `ext_api=0`
grants a socket for about ten seconds and then drops it. V7 is taken as
IQ centred on *USB dial + 5 kHz*; mixing after upsample is required so
the top of the 8 kHz waveform is not aliased.

**Transmit** captures one-second GOPs from a webcam or video file. Once PTT is
up, webcam capture, encoding, modulation, and radio output run as a rolling
pipeline instead of preparing the whole camera clip first. Acquisition and
mode header are sent once; payload GOPs then occupy exactly one RF second each.
A late receiver uses cyclic-prefix timing, pilots, and the continuous beacon
to recover without inserting gaps into the transmitted video. PTT is released
in a `finally` block and a watchdog unkeys if the pipeline overruns.

With **Save TX waveform, Kiwi IQ, and modem debug logs** enabled, AETV writes
timestamped `.tx.wav`, `.iq.wav`, metadata JSON, and modem JSONL files under
the received-video folder's `debug` directory. Compare a TX/RX pair offline:

```powershell
uv run python scripts/analyze_ota_debug.py --tx path\trial.tx.wav --iq path\trial.iq.wav
```

For a repeatable Voicemeeter or virtual-cable soak using the station's saved
audio endpoints, run the following with radio PTT disabled:

```powershell
uv run python scripts/cable_loopback_soak.py --mode V8 --gops 100 `
  --output runs/cable-loopback.json
```

The report separates the modem's normal latent distortion from error added by
the cable, and records first-result latency, callback burst size, receive
backlog, timing drift, realignments, lock loss, and audio discontinuities.

For late joins, the receiver now defers weak preamble/header coincidences from
background audio and preserves the buffered signal for blind acquisition. V8
blind acquisition uses its dedicated beacon carrier and can take about 12
seconds while it gathers enough self-identifying evidence.

**CAT / PTT**

| Method | Use |
|---|---|
| None | VOX or a manual PTT switch |
| Hamlib direct | Pick a model and COM/network device; AETV loads Hamlib in-process |
| FlexRadio 6000 | Discovered automatically; native TCP PTT plus VITA-49 UDP RX/TX audio |
| Serial RTS / DTR | SignaLink-style interfaces |

For Hamlib rigs, tune and select the operating mode at the radio. AETV's native
Flex session can create or tune its own DIGU slice when a frequency is entered.
Use **Audio-only** to test the audio path before you put RF on the air.
Test CAT / Test PTT run on a worker thread so the dialog cannot hang
the UI.

## Radio

AETV now ships two checksum-verified release modes. **Wide 8 kHz** (protocol
mode V7) is intended for a Flex 6000-series radio or another genuinely wide
audio path:

- Sample rate 24 kHz over native VITA-49 or a soundcard
- 160 OFDM carriers, 50 Hz spacing, first carrier 1000 Hz
- Transmit filter about 800-9200 Hz
- 256x144 color at 12 frames per second, one GOP per second

A receiver needs the same checkpoint and the same mode. A narrow SSB filter
will cut the upper carriers and the picture will collapse.

For a standard channel, select **Standard channel** (protocol mode V8) at both
ends. It uses 8 kHz audio, 45 carriers from 450 through 2650 Hz, and 192x108
color at 6 frames/s. Its dedicated, checksum-pinned release checkpoint is
`v8-hf3k-face-gan`; do not substitute a checkpoint trained for V7/Wide 8 kHz.
See
[`v8-hf3k.md`](v8-hf3k.md).

## Send

```powershell
uv run aetv send --source webcam --callsign YOURCALL --gops 15 --out tx.wav --play
uv run aetv send --source clip.mp4 --callsign YOURCALL --gops 30 --out tx.wav
```

To key a Flex after encoding:

```powershell
uv run aetv send --source clip.mp4 --callsign YOURCALL --gops 30 --out tx.wav `
  --flex-host 192.168.88.239 --freq-mhz 7.088 --require-mode DIGU --power 5
```

`--audio-only` plays DAX without keying. Use it to confirm the Flex meter
moves before you put RF on the air. PTT is released in a `finally` block.

## Receive

```powershell
uv run aetv receive --wav capture.wav --out rx.mp4 --display
uv run aetv receive --duration 35 --record-device "DAX RX" --out rx.mp4
```

A KiwiSDR IQ capture must be resampled to 24 kHz before it is heterodyned
back to the transmitter audio band. Doing the mix first aliases the top of
the 8 kHz waveform. The station GUI does that conversion live; `aetv
kiwi-list --lat 49.26 --lon -123.11` is the headless equivalent of the
picker.

## Simulate without a radio

```powershell
uv run aetv simulate --source clip.mp4 --gops 4 --snr 12 --out sim.mp4
```

The comparison video is source on the left, decoded copy on the right.
