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

## Station

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
the 8 kHz waveform.

## Simulate without a radio

```powershell
uv run aetv simulate --source clip.mp4 --gops 4 --snr 12 --out sim.mp4
```

The comparison video is source on the left, decoded copy on the right.
