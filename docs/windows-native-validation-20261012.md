# Native Windows follow-on validation, 2026-10-12

## Native audio

Tested on actual Windows, Python 3.12.8, sounddevice 0.5.6, SoundCard 0.4.6 and
the installed PortAudio 19.7 DLL. This was not a Linux emulation of Windows.

**15/15 native audio checks passed** at base commit
`a2fc8dde7f59f79c7e7941400d8991cf9ef00ddd`:

- Real PortAudio playback resampled 8 kHz into a 48 kHz digital output, verified
  through software loopback.
- Production WASAPI playback/capture worker framing, resampling/downmix and
  lifecycle at 8/24/48 kHz, repeated twice at each rate.
- Cancellation inside a long chunk returned false within the one-second bound,
  followed by successful fresh playback.
- Nonexistent playback/capture endpoints produced bounded helper errors.
- All four modems recovered **2/2 GOPs each**, with latent correlation
  **0.957–0.974**, and no reported capture error.
- Forced capture-child termination was reported; fresh capture/playback worked.

Only a specifically selected digital output and its software-loopback endpoint
were used, with synthetic signals at -60 dBFS. No microphone, physical radio,
RF transmission, or system-default endpoint change was involved.

**Capture selection was test-adapted.** Production intentionally excludes output
loopback from its microphone inventory. The harness changed only its own worker
endpoint selector to return the selected digital output's loopback recorder;
production framing, downmix, subprocess lifecycle, and error handling stayed
unchanged. Production's loopback exclusion remains unchanged in this PR.
This establishes a real Windows software-loopback path, not unmodified physical
microphone selection, hardware-input capture, radio wiring, native driver xrun
recovery, USB unplug/replug, or long-running real-time/learned-video performance.

[Audio measurements](validation/windows-20261012/audio-evidence.json).
Local filesystem/device-inventory details were removed; measurements were not
changed. The original validation bundle was verified after transfer with SHA256
`dbaae93a7ce0a8ec5b47fad551519005887586f686b983c5b502f1aea88f2342`.

## A real layout issue found and fixed

At 1280×800 with native Windows styling, long source/device names forced the
transmit form to **764 pixels** inside a **603-pixel viewport**. Region/Refresh
and audio-drive controls required horizontal scrolling. Stop/PTT remained
visible, but the control row was still too wide.

The fix gives six selectors a bounded content-based size hint while preserving
full choices in their popup. The fixed native form measures **603 pixels**,
matching its viewport, with **horizontal-scroll maximum 0**. Stop and PTT UNKNOWN
remain inside the window. Vertical scrolling is retained for optional controls.

Six new regression cases cover Windows/Fusion styles and V8/V8_AV/AC16_AV with
long names. All six failed on the original source and passed after the fix.
The post-fix affected GUI suite passed **42/42 on native Windows** and **42/42 with
Windows offscreen Qt**.

[Before metrics](validation/windows-20261012/native-layout-before.json),
[after metrics](validation/windows-20261012/native-layout-evidence.json),
[native GUI results](validation/windows-20261012/pytest-native-gui-fixed.log),
[offscreen GUI results](validation/windows-20261012/pytest-offscreen-gui-fixed.log).
Screenshots were inspected privately; machine-specific monitor labels are not
published here.

During cloud integration, switching the real station mode inside the new test
opened a missing-model dialog on a clean machine. The final test blocks that
selector's station-loading signal and explicitly refreshes the AV controls,
matching the existing layout-only test pattern. This is test isolation only;
the production selector fix is unchanged. The final cloud GUI/widget subset
passed **37/37** without installed AC16 weights.

## Full suite scope

The actual Windows full suite at the base commit passed **709 tests, 1 skipped**
in **303.00 seconds**; the skip needs an optional codec checkpoint.
[Original full-suite log](validation/windows-20261012/pytest-windows.log).
This full pass predates the layout patch; the 42/42 focused results above are
post-fix. Final integrated-commit CI and desktop retesting must be reported
separately rather than attributing the earlier 709-pass result to the new commit.
