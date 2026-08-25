# Simpsons 60-second analog voice + V8 channel run

Date: 2026-08-25

The first 60 seconds of the local user-supplied Simpsons test asset were processed as 60 consecutive V8 GOPs. The composite carries 0--2.2 kHz analog speech and the shifted V8/W waveform above a 300 Hz guard. Speech is delayed by one GOP. The complete 61-second transmit waveform was passed through the same `StreamingChannelEmulator` used by the GUI, at a 12 kHz composite sample rate, with one continuous channel state across every GOP.

The copyrighted source and derived clips remain local and must not be published or committed.

## Results

| Profile | Mean video PSNR | P10 video PSNR | Voice SI-SDR | Voice STOI | Alignment |
|---|---:|---:|---:|---:|---:|
| Clean | 21.29 dB | 16.06 dB | 40.34 dB | 0.949 | 0 samples |
| AWGN 12 | 20.61 dB | 15.70 dB | 9.72 dB | 0.665 | 0 samples |
| AWGN 6 | 18.91 dB | 14.83 dB | 3.73 dB | 0.470 | 0 samples |
| AWGN 0 | 15.21 dB | 12.71 dB | -2.26 dB | 0.279 | 0 samples |
| MPP 12 | 19.78 dB | 15.35 dB | -40.56 dB | 0.519 | 43 samples |
| MPP 6 | 18.04 dB | 14.67 dB | -42.15 dB | 0.375 | 43 samples |
| MPP 0 | 14.44 dB | 12.28 dB | -45.98 dB | 0.233 | 43 samples |

The very negative whole-clip MPP SI-SDR is expected: Watterson fading continuously changes analog gain and phase, which violates SI-SDR's single-gain model. STOI and listening are more useful for the faded speech. The emulator's causal Hilbert path contributes 43 native-rate samples (5.375 ms) of measured delay; the renderer compensates this in addition to the intentional one-second GOP delay.

The AWGN 6 mean latent NMSE contains equalizer blow-up outliers and is therefore less representative than rendered quality and PSNR. All 60 tracked GOP calls returned for every profile, but successful framing does not imply usable content at 0 dB.

## Renders

Artifacts are under `runs/simpsons-analog-channel-60s`.

- `simpsons_60s_channel_grid.mp4`: source plus all seven conditions; audio is source on the left and MPP 6 recovered speech on the right
- `simpsons_60s_<profile>.mp4`: side-by-side source/reconstruction for each condition; source audio left, matching recovered audio right
- `channel_<profile>.wav`: the actual impaired 12 kHz composite waveform for each profile
- `composite_transmit.wav`: unimpaired delayed-voice/upper-AETV waveform
- `channel_grid_midpoint.png`: representative visual frame
- `metrics.json`: full measurements

MP4 monitor audio uses one positive whole-clip RMS AGC gain per recovered channel. It makes channel damage audible without removing time-varying fades or noise. Both stereo channels in the MPP 6 render measure approximately -20.8 dB mean and -3 dB peak.

All MP4s are H.264 Constrained Baseline, `yuv420p`, AAC-LC stereo at 48 kHz, and fast-start. The grid and representative individual files were fully decoded with FFmpeg. Nine focused composite/modem tests pass.

## Reproduction

```bash
.venv/bin/python scripts/experiment_simpsons_analog_channel.py \
  --duration 60 \
  --out runs/simpsons-analog-channel-60s
```
