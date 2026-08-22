# Operating AETV on HF

AETV is analog video. There is no bitstream and no FEC in the usual digital
sense. The decoder reconstructs pictures from noisy latents, so a weak copy
looks soft rather than blocky or silent.

## Legal

- You must hold a license that authorizes the band, power, and emission.
- The beacon carries an 8-character callsign. Set `--callsign` to yours.
- Flex-8k occupies about 8 kHz of audio. That is not a standard 2.7 kHz SSB
  channel. Use a wide digital slice (DIGU) and stay inside the band plan.
- Yesterday's clear frequency is not a check today. Listen first.

## Station app

```powershell
uv sync --extra gui
uv run aetv gui
```

The window is the SSTVAE-style shack layout: a receive waterfall across
the top, decoded video on the left, webcam or file transmit on the
right. Settings persist under `%APPDATA%\AETV\settings.json` on Windows
and `~/.config/AETV/settings.json` elsewhere.

**Receive** can be a local soundcard (Flex DAX RX, a virtual cable, or
any input) or a public KiwiSDR. **Find Kiwis** probes the public list
and keeps receivers that still advertise an API channel — `ext_api=0`
grants a socket for about ten seconds and then drops it. V7 is taken as
IQ centred on *USB dial + 5 kHz*; mixing after upsample is required so
the top of the 8 kHz waveform is not aliased.

**Transmit** encodes one-second GOPs from a webcam or a video file,
plays the Flex-8k waveform into the selected output device, and keys
the rig only for that duration. PTT is released in a `finally` block
and a watchdog unkeys if playback overruns.

**CAT / PTT**

| Method | Use |
|---|---|
| None | VOX or a manual PTT switch |
| Hamlib `rigctld` | The same TCP daemon WSJT-X already uses (`127.0.0.1:4532`) |
| FlexRadio 6000 | SmartSDR TCP; binds the GUI client, checks frequency/mode, keys `xmit` |
| Serial RTS / DTR | SignaLink-style interfaces |

Frequency and mode are checked, never set. Use **Audio-only** in
settings to confirm the DAX meter moves before you put RF on the air.
Test CAT / Test PTT run on a worker thread so the dialog cannot hang
the UI.

## Radio

The published mode is **V7** on a Flex 6000-series radio:

- Sample rate 24 kHz into DAX TX
- 160 OFDM carriers, 50 Hz spacing, first carrier 1000 Hz
- Transmit filter about 800-9200 Hz
- 256x144 color at 12 frames per second, one GOP per second

A receiver needs the same checkpoint and the same mode. A narrow SSB filter
will cut the upper carriers and the picture will collapse.

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
