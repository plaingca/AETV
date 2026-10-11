# Amateur-operator GUI review — 2026-10-11

## What changed

- The default station window fits 1280 × 800. Optional content scrolls rather than forcing the window beyond a laptop screen. Receive/transmit actions, progress and status stay pinned outside those scroll areas.
- Radio details, the prepared-clip bank and diagnostic log start collapsed. `View > Log` restores the log. Radio details show only controls relevant to the chosen hardware.
- A prepared-clip click selects and previews it; it never starts transmission. Press **Transmit** explicitly. The separate **Transmit QRL?** button makes clear that checking whether a frequency is in use also keys the radio.
- Eight adjacent route buttons are replaced by one dropdown. **Radio / audio output** and **Local test** have distinct explanations and action labels. A local test disables radio-output selection and QRL transmission.
- Modes retain their protocol IDs and show bandwidth alongside a friendly name. The adjacent hint gives picture size, frame rate and the actual audio-filter range; nominal occupied bandwidth and the filter's guard margins are deliberately distinguished. Both stations need the same mode.
- Manual receive-frequency correction is disabled while automatic correction is enabled. Collapsed audio/CAT details explain where to configure the rig.
- Truncated status messages retain their complete text in a hover tooltip. Rig settings distinguish **Check connection** from **Key transmitter for 0.25 s**, including a tooltip warning about the brief PTT test.

The initial settings remain V8, the existing audio drive, and the saved radio route. This review does not silently change an operator's configured hardware or RF settings.

## A straightforward first use

1. Open `File > Settings`: enter your callsign, select the radio connection and audio devices. Use **Check connection** before a PTT test. Install the needed release model through `File > Model Manager`.
2. Start receiving and listen. Select the same mode at both ends, and confirm that the radio filter and your permitted operating bandwidth accommodate it.
3. Choose a webcam, screen or video file. For a hardware-free practice decode, choose **Local test: Clean** and **Run local test**. For on-air use, select **Radio / audio output**, check the frequency, callsign and output path, then press **Transmit**.

## Evidence and limits

These are real offscreen Qt renders of the application, not drawn mockups. Hardware enumeration, camera preview and model startup are stubbed deliberately; no camera, microphone, radio, CAT/PTT, RF transmission, model download or external receiver was used. The screenshots label the absent model and stubbed devices. Thus they validate layout and GUI state, not hardware connectivity or decode quality.

- [Default window](validation/gui/main.png)
- [Local-test route](validation/gui/local-test.png)
- [Expanded hardware and clip controls](validation/gui/advanced.png)
- [Station settings](validation/gui/settings.png)

Reproduce from the repository root with the GUI/dev dependencies installed:

```sh
QT_QPA_PLATFORM=offscreen AETV_OFFLINE=1 XDG_CACHE_HOME=/tmp/aetv-cache \
  .venv/bin/python scripts/render_gui_review.py --output docs/validation/gui
QT_QPA_PLATFORM=offscreen AETV_OFFLINE=1 XDG_CACHE_HOME=/tmp/aetv-cache \
  .venv/bin/python -m pytest tests/test_gui_usability.py tests/test_tx_panel.py tests/test_gui_logging.py -q
```

The focused suite passed **40 tests**. It covers laptop fit, pinned actions with expanded content, clip selection without transmission, local/radio route recovery, all six mode filter hints, relevant radio controls and complete error tooltips. A broader GUI/TX/HackRF regression run also passed **70 tests** before the final pinned-action layout adjustment; the focused suite was rerun afterward.


## Interrupted-flow and PTT safety follow-up

The 2026-10-11 follow-up adds explicit stopping and failure states:

- Local-emulator playback never lights the radio PTT lamp.
- A failed PTT OFF remains a red **PTT UNKNOWN** warning and prevents automatic receive resumption. Check and manually unkey the radio immediately.
- Terminal TX state does not release controls or resume RX until the worker has finished cleanup. Repeated cancellation before half-duplex handoff does not start the job. Unexpected worker errors restore usable controls and retain the error.
- Quit is refused while the TX worker is still stopping. Stalled encoders retain ownership until they exit, while radio OFF happens before waiting for encoder cleanup. An in-flight watchdog OFF must finish before ownership passes to another send.
- Receive gaps/overruns explain that synchronization is being searched for and restarting transmission can speed recovery. A continuous stream does not carry a fresh preamble at each GOP.

Additional actual-Qt renders with injected engine states (not real radio failures):

- [Local test running, radio PTT idle](validation/gui/local-test-running.png)
- [Cancelled local test](validation/gui/local-test-cancelled.png)
- [PTT OFF failure warning](validation/gui/ptt-off-warning.png)

Focused station/GUI/widget regression command (93 tests passed after these safety changes):

```sh
HOME=/tmp/aetv-home QT_QPA_PLATFORM=offscreen AETV_OFFLINE=1 \
  .venv/bin/pytest -q tests/test_station.py tests/test_gui_usability.py \
  tests/test_gui_logging.py tests/test_widgets.py
```

The new checks cover cancelled startup, blocked encoder cancellation with prompt OFF, CAT setup producer cleanup, failed OFF, delayed watchdog OFF ownership, repeated GUI Stop/retry, unexpected worker exceptions, safe quit, and deferred terminal-state handoff. Hardware remains stubbed in these tests; see the separate audio/Hamlib validation for transport evidence.

The final combined regression run added `tests/test_tx_panel.py`, `tests/test_radio_frequency.py`, and `tests/test_model_manager.py` to that command: **140 tests passed**. `git diff --check` also passed.

PTT uncertainty is latched until an explicit operator acknowledgment that the radio is in receive. Transmit, QRL, manual receive, Apply tuning, and radio-settings access prompt for that check, defaulting to Cancel. Cancelling does not send, retune, start RX, or clear the red warning. Unrelated idle/cancelled/done updates cannot clear it. This acknowledgment records a physical operator check; it is not an automatic proof of radio state.

After adding the uncertainty acknowledgment regressions, the same combined command passed **144 tests**. The warning screenshot remains visually unchanged; its enabled actions now enter the confirmation gate before doing work. The rendering script explicitly simulates acknowledgment when moving from that injected failure to its next independent screenshot scenario.

Ownership guarantees here cover the TX producer and in-flight watchdog OFF command. Existing nested screen/camera capture shutdown still uses bounded waits; a hung OS/device capture thread is not guaranteed to terminate and can require external process/device shutdown.

## Windows CI sizing follow-up

The first PR CI run exposed a native Windows minimum-width difference: the
window expanded to 1290 px rather than the required 1280 px. The radio header
and pinned receive-action strip now use compact two-row grids, keeping buttons
fully visible without weakening the laptop-fit requirement. New tests exercise
Windows and Fusion styles at 18 pt and check exact 1280 × 800 size, minimum
button sizes, containment and non-overlap. The affected suite passed **161
tests** locally; actual Windows CI is rerun on the follow-up commit.

A second CI pass confirmed the default Windows size, while the added large-font
checks exposed further physical-font scaling pressure. QRL now has its own row,
receive rows size independently, the radio label can wrap, and non-safety status
fields elide with complete hover text. PTT remains fully visible. Regression
coverage includes both 18 pt and explicit 48 px fonts, detailed DPI/layout
failure diagnostics, and a 192-DPI subprocess check. **164 affected tests** and
all four large-font cases passed locally. CI runs this GUI file before the full
suite to surface platform layout problems promptly.
