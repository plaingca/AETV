# AC16 receive synchronization correction — 0.1.20rc3

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
