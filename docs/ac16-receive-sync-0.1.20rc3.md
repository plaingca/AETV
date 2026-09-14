# AC16 synchronization and DirectML color validation — 0.1.20rc3

A Windows RTL-SDR receive log showed 11 rejected startup candidates, 40 pending
preamble searches, two failed blind acquisitions, and no accepted or decoded
GOPs. The failure therefore occurred before neural inference. The log covers
17:38–17:40 on September 13, 2026, before the final repeat transmission; it does
not establish the outcome of that later attempt. A diagnostic log without the
recovered waveform cannot determine the exact received impairment.

Offline replay of the transmitted waveform exposed a reproducible blind-entry
bug: residual offsets around ±12.5 Hz cancel the real signal's cyclic-prefix
correlation. The receiver rejected clean AC16 as low timing confidence (0.07).
Timing now uses analytic audio, keeping correlation magnitude independent of
carrier rotation. The payload equalizer and independent header/beacon checks
retain their existing rules.

The startup scanner now limits each AC16 preamble search to the newly covered
quarter-second interval and caches the deterministic preamble template. It
still validates against the complete following GOP. Pending blind-acquisition
history is bounded to 12 seconds after each unsuccessful scan. A matched local
replay of 14.875 seconds of payload took 13.588 seconds of processing before and
6.924 seconds after, recovering the same four GOPs. These are host-specific
profiled timings, not measurements of the user's Windows CPU.

The GUI reports acquisition progress instead of remaining at `starting`.
Direct SDR debug capture now saves the recovered modem waveform as well as
JSONL events, and includes driver/tuning messages and blind-search duration.

Validation before packaging: 364 tests passed, two skipped. Added coverage
checks late entry at ±12.5 Hz for all modem bands, AC16 payload chronology with
100 ms and 2 s callbacks, startup-history bounds followed by a fresh preamble,
and Pluto/RTL waveform capture and status changes. Existing unused-import lint
findings are unchanged; no new findings were introduced. Each packaged GUI and
benchmark executable also runs the ±12.5 Hz AC16 fixture through its own frozen
modem during the native SDR runtime smoke check. Physical Windows RF reception
still requires confirmation on the user's receiver.

## DirectML color corruption

The user also reported a severe red cast during clean local loopback on
DirectML, while CPU loopback had normal colors. This isolates the reported
color problem to the accelerator path; it does not identify the faulty GPU
operator or driver. No Windows GPU is attached to the development host.
The rc2 Windows GPU artifact's embedded report identifies ORT 1.24.4: the
packaging step had installed DirectML without respecting the application's
`onnxruntime<1.24` constraint. The new build pins DirectML 1.23.0, the available
Windows DirectML wheel in the application's 1.23 release family. This corrects
version drift; it is not by itself proof that 1.24.4 caused the red cast.

AC16 now qualifies DirectML against CPU sessions of the same checksum-pinned
model before selecting it. Two generated RGB clips exercise texture, motion
and color, each at full and half receiver confidence. The check covers encoder
latents and the complete decoded GOP. Limits are 0.2% relative latent RMSE,
one 8-bit level of mean RGB error, and eight levels of maximum RGB error.

If normal DirectML fails, the loader retries with vendor metacommands and
graph fusion disabled. These controls are supported by the pinned-runtime-era
[DirectML provider factory](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/dml/dml_provider_factory.cc)
and [session configuration](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/dml/dml_session_options_config_keys.h).
If the retry fails or cannot initialize, the loader selects its CPU reference
sessions and reports the actual CPU device, reasons in the model tooltip, and
a log message. The benchmark JSON retains all qualification attempts.

This is a runtime compatibility check and explicit fallback, not a claim that
an unidentified DirectML kernel has been repaired. The compatible GPU profile
still needs validation on the affected Windows hardware. Tests inject red
decoder bias, encoder corruption, nonfinite outputs and missing providers to
verify retry/fallback behavior. The real AC16 models also pass the same
numerical check on this host's CUDA provider against CPU.

After the color-qualification changes, the full suite passed 371 tests with
two skipped. The DirectML GPU hardware limitation above remains explicit;
successful CPU fallback on a build runner is not GPU accuracy evidence.

## Downloaded artifact validation

All four builds and both Windows/Linux CI jobs passed at binary commit
`c85bf250826b5c54f837c84156323ca0f35e625c`. Both CI platforms passed 371 tests
with two skips. The downloaded Windows archives contain the expected pinned
RTL/Pluto runtime files and offline USB setup tools. Their frozen GUI and
benchmark entry points passed the late-entry fixture and native library checks.
The Windows GPU report confirms DirectML 1.23.0 and explicit CPU fallback on
the runner without a usable GPU adapter.

The downloaded Linux CPU executable passed a fresh native runtime check and a
60-second 439 MHz Pluto → RTL 1001 trial: 60/60 GOPs, 600/600 displayed frames,
correct chronology, minimum latent cosine 0.95237, and no recorded errors.
Independent device readback confirmed Pluto TX powered down at −89.75 dB gain,
with zero remaining RTL capture processes. The new modem WAV was then replayed
from 4.125 seconds into reception with an additional 12.5 Hz tuning offset;
blind entry recovered four sequential payloads at minimum cosine 0.95408.

[Checksums, package reports, and trial evidence](evidence/ac16-receive-sync-0.1.20rc3.json)
are recorded separately from the unchanged model provenance. The six portable
assets and machine-readable validation report accompany the
[Stable release downloads](https://github.com/plaingca/AETV/releases/tag/v0.1.20).
