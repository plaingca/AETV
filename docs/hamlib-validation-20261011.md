# Hamlib/CAT hardening validation, 2026-10-11

**Update:** actual Hamlib 4.6.2 dummy-driver validation has now been completed.
See the new real-dummy section below. The original mock-only results are retained
as a historical baseline; their native-runtime limitation is superseded only for
Debian x86-64 Hamlib 4.6.2, not for real radio hardware.

## Scope and result

This is software-only validation, not a certification for all amateur stations.
No real radio, serial adapter, antenna, amplifier, or RF output was used. No
hardware was keyed. TCP tests use an ephemeral loopback listener, not a user's
rigctld daemon. Direct-library tests use Python mocks, not a loaded Hamlib driver.

Command run from the repository root:

```sh
.venv/bin/python -m pytest tests/test_cat.py tests/test_station.py::test_rigctld_ptt_and_freq -q
```

Result: **41 passed in 15.42 seconds** (40 dedicated CAT tests and one existing
station integration test). A subsequent native-focused run after final ABI
signature cleanup passed 10 tests. `git diff --check` was clean.

## Covered behaviors

- Classic rigctld mode and passband responses split across multiple TCP packets;
  both lines are consumed before accepting the next command's acknowledgement.
- Exact numeric status parsing. Negative codes ending in zero, including -10 and
  -20, are errors. Positive/non-numeric status values, arbitrary text, non-ASCII
  bytes, missing values, and oversized responses do not count as success.
- Missing passband, EOF midway through a value, timeout after a possibly executed
  key-on, malformed frequency, and nonfinite/negative/zero frequencies.
- Serialized concurrent reads; invalid transport state cannot leak stale reply
  bytes into the next transaction. A well-formed unsupported-capability response
  remains usable for other operations.
- No automatic replay of key-on or tuning after a failed transaction. An explicit
  OFF may reconnect and make a bounded retry because OFF is idempotent. Closing
  requests OFF, closes the transport, and is safe to repeat.
- Direct-library native ABI widths, readable mode labels, unsupported PTT and
  configuration tokens, failed radio opening, cleanup after a close exception,
  invalid tuning inputs, and concurrent native operation/close sequencing.
- Mock configuration forwarding for a dummy rig, Windows COM-style device,
  Linux stable serial pathname, and a network device, with and without baud.
  These establish argument forwarding only, not actual device support.
- A failed tune during client creation releases the client. Radio model discovery
  timeout does not crash settings discovery.

## Official protocol and ABI references

- [Hamlib rig.h](https://raw.githubusercontent.com/Hamlib/Hamlib/master/include/hamlib/rig.h)
- [Hamlib rigctld manual](https://hamlib.sourceforge.net/html/rigctld.1.html)

The ABI bindings follow `rmode_t = uint64_t`, `pbwidth_t = shortfreq_t = signed
long`, configuration token = `long`, and `vfo_t = unsigned int`. Python `ctypes`
uses the platform's native `long` width rather than assuming Linux and Windows
have the same layout. Native mode strings use `rig_strrmode` when available.

## Safety boundaries and remaining validation

A successful command acknowledgement does not prove that physical RF stopped.
Loss of a network connection, stuck interface, radio firmware failure, or an
external amplifier can defeat software release attempts. Operators should use a
radio-side transmit timeout and retain an independent way to stop transmission.
PTT OFF errors must be treated as requiring an immediate radio check.

Direct Hamlib calls run synchronously in the process and are serialized to avoid
concurrent C-library access and cleanup of a handle still in use. A native call
that hangs can therefore block an OFF request behind the same lock. The station
watchdog begins after key-on returns, so it cannot recover a key-on call that
never returns. Its cancellation signal also requires the audio output path to
cooperate. Driver timeouts and radio-side safeguards remain necessary; this
patch does not provide process isolation or a hard deadline for C calls.

Not validated here:

- Actual Hamlib shared-library loading, installed Hamlib versions, or radio driver
  behavior; direct calls were mocked.
- Windows DLL dependency loading, macOS dynamic-library loading, 32-bit targets,
  USB/serial permissions, disconnect/re-enumeration, or real COM devices.
- Icom CI-V addressing, Yaesu/Kenwood model-specific settings, serial handshake,
  parity/stop bits, shared CAT ports, unusual VFO/split operation, or network-radio
  authentication.
- Distinct microphone/data PTT selections (`RIG_PTT_ON_MIC`, `RIG_PTT_ON_DATA`),
  inverted hardware keying, separate PTT devices, amplifier sequencing, and
  radio-specific USB audio routing. Existing direct CAT uses ordinary PTT ON and
  the current VFO; installations needing other controls require explicit setup
  or additional settings/API work rather than assumed compatibility.
- Enforcing a Hamlib mode requirement. The existing `require_mode` setting is
  applied to Flex; it is not newly enforced for Hamlib. USB, PKTUSB, DIGU, and
  other labels are not assumed interchangeable. The operator must select the
  correct radio mode, data input, and passband for their station.
- Native Flex and RTS/DTR hardware operation; those are separate backend paths.

A safe next validation stage is an actual supported Hamlib installation with its
dummy backend, then controlled bench checks of each target station without RF
radiation. Hardware validation should record OS, Hamlib version, radio/firmware,
connection type, baud/handshake, audio routing, PTT method, and observed behavior
under disconnect and failed OFF conditions before any on-air use.

## Actual Hamlib dummy-radio validation

### Safe, repeatable operator/developer check

Install official Hamlib 4.x utilities and its shared library. On Debian 13 the
packages are `libhamlib-utils` and `libhamlib4t64`. Run from the AETV checkout:

```sh
.venv/bin/python scripts/validate_hamlib_dummy.py --json docs/validation/hamlib-dummy-local.json
.venv/bin/python -m pytest tests/test_hamlib_dummy.py tests/test_cat.py -q
```

The smoke utility has no model, device, host or port options: native control is
hardcoded to **Hamlib model 1**, without any serial device, and TCP control starts
its own model-1 daemon bound to loopback on an ephemeral port. It cannot be pointed
at an existing radio or daemon. All PTT state changes are simulated. It does not
play audio or access a soundcard. It requests OFF and closes clients in cleanup,
then terminates its owned daemon with a bounded kill fallback. Do not replace the
hardcoded model/connection parameters with a real rig for these tests.

When Hamlib is missing, the smoke utility exits 1 with installation/search-path
instructions. The integration tests explicitly skip (not pass); existing mock
unit tests remain available without Hamlib. Neither route installs dependencies
automatically. Model 1 automatically selects simulated `ptt_type=RIG`; Hamlib's
default dummy PTT is NONE and cannot provide useful ON/OFF readback.

### Environment and package provenance

Validated on Debian 13, x86-64, glibc 2.41, Python 3.12. Native tool output:

```text
rigctld Hamlib 4.6.2 2025-02-09T21:03:50Z SHA=870364 64-bit
```

System package installation was unavailable: `sudo` was absent and apt package
indexes were not present. Official Debian packages were downloaded and extracted
with `dpkg-deb -x` into `/workspace/scratch/d4a4673d7215/hamlib-runtime`. No system
package, system settings or repository-bundled runtime was changed. Download URLs:

- https://deb.debian.org/debian/pool/main/h/hamlib/libhamlib-utils_4.6.2-1+b1_amd64.deb
- https://deb.debian.org/debian/pool/main/h/hamlib/libhamlib4t64_4.6.2-1+b1_amd64.deb

Locally measured SHA-256 of the HTTPS downloads, recorded for reproducibility:

```text
cc2498d54ab00c370a8acb8ce4f166e959d8cf8c882048f038dbe268e927ff24  libhamlib-utils_4.6.2-1+b1_amd64.deb
2c2819d587df16fc9bcc00f76b327982c79e15308d388c2aea0a877f4fec7937  libhamlib4t64_4.6.2-1+b1_amd64.deb
```

These are recorded download digests, not a separate signature verification. The
official package description is https://packages.debian.org/trixie/libhamlib4t64.
The extracted runtime used:

```sh
export HAMLIB_RUNTIME=/workspace/scratch/d4a4673d7215/hamlib-runtime
export PATH="$HAMLIB_RUNTIME/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HAMLIB_RUNTIME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Set loader paths **before** launching Python. AETV's library lookup now also asks
the OS loader for the Hamlib SONAME, so a portable library on `LD_LIBRARY_PATH`
works even when absent from the system's `ldconfig` database.

### Readback and fault coverage

Saved actual native/TCP trace: [hamlib-dummy-20261011.json](validation/hamlib-dummy-20261011.json).
Both native and TCP returned FM, initial PTT false, exact readbacks for 7,088,000,
14,230,000 and 1,296,000,000 Hz, and ON/OFF/ON/OFF state changes. A separate TCP
client read PTT false after the controlling client closed while keyed. The smoke
report records the daemon as stopped.

Real-dummy integration tests cover:

- Twenty repeated native and TCP ON/OFF cycles with state readback.
- Native cleanup requests OFF before releasing its handle; repeated close and OFF
  are safe, and ON after close is rejected.
- TCP connection loss while keyed; ON is not replayed, and explicit OFF reconnects
  and independently reads back false through a second client.
- Abrupt owned-daemon death; OFF fails visibly within the configured bound.
- Production `TxEngine` complete, cancel-during-playback and audio-player-error
  exits against TCP; cancel and player-error exits against native Hamlib. Tests
  observe real dummy PTT true during the injected player and false before/after
  cleanup. Audio is an injected software player, not physical audio hardware.
- Actual model discovery preserves multiword names such as `NET rigctl`.

The initial real-runtime run exposed a harness issue: graceful rigctld termination
can wait on an open client. The daemon-crash test now deliberately kills only its
owned dummy process; ordinary fixture cleanup closes clients and has a bounded
terminate/kill fallback. No real radio was involved.

### Remaining limits

The dummy establishes real shared-library ABI, protocol, state transitions and
application cleanup behavior. It does not establish physical RF release, USB/audio
routing, Windows/macOS loading, serial hardware recovery or compatibility with
any manufacturer's radio firmware. Synchronous native-driver hangs and transport
failures remain unable to guarantee physical OFF; retain radio-side transmit
timeouts and an independent shutdown method. Model-specific mode/data-input/PTT
requirements and unsupported get-PTT capabilities still need bench validation.

Final focused verification with the extracted runtime on PATH/LD_LIBRARY_PATH:
**60 passed in 26.31 seconds** (11 actual-dummy integration tests, 49 CAT unit
checks). With those environment variables absent, the actual-dummy file reports
**11 skipped in 0.24 seconds**, and the standalone utility exits 1 with the
expected actionable missing-runtime message. `git diff --check` and Python
byte-compilation of the changed CAT module, script and integration tests passed.
