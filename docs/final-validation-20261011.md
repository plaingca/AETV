# Final software validation, 2026-10-11

Base: `23ec89d9bb1101bb249e17565e325079b84d42c0`.
This follow-on includes the earlier GUI/CAT/RF hardening and the new production
audio-path, HF-channel and actual Hamlib dummy-radio checks.

## Final regression

The final broad local suite passed **585 tests**, with **2 skipped**, in
**187.78 seconds**. Six Torch-dependent modules were excluded because Torch
is unavailable. No Torch installation was performed. The two skipped checks
require the optional FT8 decoder runtime and V7 checkpoint. This is not a full
training-suite pass.

```sh
PATH=/path/to/hamlib-runtime/usr/bin:$PATH \
LD_LIBRARY_PATH=/path/to/hamlib-runtime/usr/lib/x86_64-linux-gnu \
QT_QPA_PLATFORM=offscreen AETV_OFFLINE=1 HOME=/tmp/aetv-home \
XDG_CONFIG_HOME=/tmp/aetv-config XDG_CACHE_HOME=/tmp/aetv-cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  --ignore=tests/test_attention.py --ignore=tests/test_audio_metrics.py \
  --ignore=tests/test_core.py --ignore=tests/test_gop_boundary.py \
  --ignore=tests/test_shared_eval.py --ignore=tests/test_wide4k.py
```

The Hamlib path above is only necessary for a workspace-extracted runtime.
System installations normally expose these binaries/libraries automatically.
The Linux CI dependency step installs `libhamlib-utils` so the real dummy tests
run rather than silently skip. Windows/macOS native radio loading was not
validated in this cloud run.

Focused final checks:

- **66 CAT/Hamlib tests**, including **17 actual-dummy integrations**.
- **159 station/GUI/QRL tests** after the final cancellation regression update.
- **55 audio callback/rate-converter/overflow/cancellation tests**.
- `uv build` produced both source distribution and wheel.
- Python byte-compilation and `git diff --check` passed.

The first broad pass found a stale QRL test that expected an already-cancelled
request to key briefly. It now asserts that startup cancellation never keys;
a separate test retains cancellation during PTT lead with ON → OFF → close and
no playback. The complete suite was rerun after that test-only correction.

## Bench evidence

- [Audio/HF report](audio-path-validation-20261011.md): 120 synthetic trials,
  96/96 clean-path GOPs, explicit fading losses and delayed capture-gap recovery.
  Verified V8 ONNX encode/decode recovered 9/9 learned GOPs (54 frames).
- [Hamlib report](hamlib-validation-20261011.md): actual Hamlib 4.6.2 native and
  loopback TCP dummy control. Six combined tests run the production TxEngine,
  production audio adapters, virtual HF path, capture ring and modem, with real
  dummy PTT readback on completion, cancellation and endpoint errors.
- [GUI report](gui-usability-20261011.md): actual Qt renders and interaction
  regressions, including local-test cancellation and latched PTT UNKNOWN.
- [Earlier RF sweep](rf-simulation-20261011.md): 180 modem-level trials retained
  as complementary evidence, not relabeled as learned-video or hardware tests.

## Remaining boundaries

No physical audio device, serial radio, amplifier, antenna or RF transmission
was used. PortAudio is absent here; native audio endpoints were deterministic
software substitutes. The dedicated audio bench does not run RxEngine or prove
OS/driver deadlines. CAT acknowledgements and dummy state do not prove physical
RF stopped. Native-driver hangs still require independent hardware recovery.

The learned-path run with two CPU threads decoded in 1.52–1.73 s/GOP. A separate
five-repeat isolated benchmark measured default/four/eight-thread decode medians
of 0.81/0.93/0.66 s; eight threads gave the most margin in that small probe.
Those per-operation timings do not establish sustained real-time station or
full-duplex operation. No inference defaults were changed.

All changes were tested locally before opening a draft PR. Publication and CI
status are reported separately; no merge, release or deployment was performed.

## CI follow-up to the initial publication

The initial draft-PR commit passed Linux CI with **703 passed, 2 skipped**,
including all six Torch-dependent modules, plus the package build. Windows CI
reported **685 passed, 19 skipped**, with one failure: its native style expanded
the default window to 1290 px. Two compact control-row grids fix the width
pressure without loosening the 1280 px assertion. Windows/Fusion large-font
regressions and the affected suite passed **161 tests** locally. Both CI jobs
are rerun for that follow-up; see the draft PR for their latest status.

The next Windows run confirmed the default layout fix and exposed a separate
high-DPI/large-font width constraint in the added regression. Independent action
rows, concise labels and eliding status fields address that case without reducing
font size or weakening the laptop bounds. The final affected suite passed
**164 tests**, including four large-font cases repeated at 192 DPI. The final
broad local rerun passed **590 tests, 2 skipped** in **188.78 seconds**, with the
same six local Torch exclusions. CI now runs GUI layout checks before the full
suite so platform layout failures surface early. Final hosted results are linked
from the PR and included in the delivered report.
