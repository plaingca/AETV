# Native PortAudio validation, 2026-10-12

## What actually ran

The cloud environment initially had sounddevice 0.5.6 but no PortAudio shared
library; importing sounddevice raised `OSError: PortAudio library not found`.
Installed Debian trixie's `libportaudio2` 19.6.0-1.2+b3 into a workspace-local
runtime from Debian's signed APT repository. System-wide installation was not
available because this executor has no root package-management access.

The new `scripts/validate_portaudio_path.py` uses real sounddevice and PortAudio
19.6 with ALSA's file PCM backed by the null PCM. It writes an isolated ALSA
configuration before importing sounddevice and explicitly selects only its named
file endpoint. It never selects physical audio hardware or changes host audio
settings. No endpoint objects or callbacks are monkeypatched.

The chain tested is:

1. Production modulation and streaming TX resampling/write loop
2. Real PortAudio/ALSA output stream, persisted float32 native samples
3. Whole-trace clean/AWGN/Watterson HF simulation at the native sample rate
4. Real PortAudio/ALSA input stream and production input callback/resampler
5. Production ring buffer, streaming modem and recovered latent measurements

The reported native default is 44.1 kHz. Modes use their actual 8/12/24/48 kHz modem
rates. Native output bytes matched the intended resampled waveform within
3e-8 absolute error.

## Results

Seed 0, four independent GOPs per mode/profile:

| Mode | Clean | AWGN 6 dB | MPP 12 dB |
|---|---:|---:|---:|
| V8 | 4/4 | 4/4 | 4/4 |
| V9 | 4/4 | 4/4 | 0/4 |
| V7 | 4/4 | 4/4 | 0/4 |
| AC16 | 4/4 | 4/4 | 4/4 |

All four clean baselines passed the original correlation >0.9 and NMSE <0.2
criteria, with no ring overrun or reported input discontinuity. Fading rows are
measurements, not guaranteed-delivery assertions. The V9/V7 MPP outcome must not
be presented as all impairment profiles passing.

Also exercised native streams at 8, 12, 24, 44.1 and 48 kHz; production callback
playback completion/cancellation; production blocking-write cancellation;
bounded-ring overrun reporting; three complete capture stop/close/reopen cycles;
and genuine PortAudio errors for a nonexistent input and output endpoint.

Machine-readable results: [native PortAudio report](validation/portaudio-linux-20261012.json).

Windows native audio and the discovered layout fix are documented in the
[Windows follow-on report](windows-native-validation-20261012.md).

## Broad regression

With the installed PortAudio runtime and existing Hamlib 4.6.2 runtime:
**596 passed, 2 skipped in 191.04 seconds**. The skips require optional FT8/V7
assets. As in the preceding cloud run, six Torch-dependent modules were
explicitly excluded because Torch is unavailable locally: `test_attention`,
`test_audio_metrics`, `test_core`, `test_gop_boundary`, `test_shared_eval`, and
`test_wide4k`. This is a broad non-Torch pass, not an unqualified full-suite pass.
The command uses the writable HOME/XDG directories and one-thread BLAS settings
shown in the [preceding validation report](final-validation-20261011.md).

This final run includes the integrated Windows selector-width fix and six new
layout regressions. Python byte-compilation and `git diff --check` also passed. [JUnit evidence](validation/portaudio-regression-20261012.xml).

## Important boundaries

This is accelerated file-backed replay, **not simultaneous, clocked loopback**.
ALSA null PCM has no physical sampling clock. Therefore clean callbacks and
false native xrun flags do not establish real-time schedulability, device latency,
USB drift, hardware underrun/overrun handling, or unplug/reconnect behavior.
The ring-overrun test really overruns the application ring; it does not induce a
native input overflow. Repeated reopen really restarts streams; it is not a
physical disconnect test. Existing deterministic tests separately exercise the
production native-status/underrun handling with injected statuses.

An attempted private PulseAudio null-sink fixture could not establish its Unix
socket (`Operation not permitted`, including after approved escalation). No
clocked sound-server results are claimed. No sound server was left running.

This follow-on has no physical capture, speakers, RF, Windows WASAPI, learned
model inference, or full Station/RxEngine orchestration. See the prior audio/HF
and Hamlib reports for those separately measured software paths; physical RF
release remains unverified.

## Reproduce on Linux

With ordinary package-management access:

```sh
sudo apt-get install libportaudio2 libasound2-data
uv sync --locked --extra gui --extra train --extra dev
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python scripts/validate_portaudio_path.py --output /tmp/portaudio-validation.json
```

The harness itself creates and removes its own temporary ALSA file fixture. It
requires no PulseAudio service, kernel loopback module, audio device, or root
privileges. To use a local extracted PortAudio package instead of system-wide
installation, expose its library directory through `LD_LIBRARY_PATH`; include
an unversioned `libportaudio.so` symlink so Python's `ctypes.util.find_library`
can discover the extracted library. Debian APT authenticates repository
metadata and package checksums before extraction.

Linux CI now installs `libportaudio2` and runs this same native harness before
the ordinary test suite. Windows uses its own native test evidence rather than
pretending this ALSA fixture establishes WASAPI behavior.
