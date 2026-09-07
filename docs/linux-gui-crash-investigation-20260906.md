# Linux GUI file-operation crash investigation

## Fix implementation for 0.1.19

`LogPane.append_line` now emits a queued signal to a decorated slot on the
widget's owning thread. This covers direct Station callbacks as well as messages
forwarded by the TX/RX panels. The native text append, layout, and scrollbar
updates all execute in that slot. `MainWindow._log` is also a registered Qt slot
for panel signals; its direct worker-callback path only formats and enqueues text.

The audit covered Station logs (TX debug, microphone, prepared cache, RX save),
RX/TX panel logs, propagation-calibration messages, model workers, device
enumeration, CAT testing, and FT8 status callbacks. The station callback was the
direct worker-to-widget path. Other worker notifications already use Qt signals;
the remaining QThread lambda receiver behavior was checked with a live Qt event
loop and delivered on the GUI thread. No second independent text-log widget was
found. Regression tests exercise station, RX, TX, and direct widget entry points
from workers and assert that the native append occurs only on the GUI thread.

Linux folder-opening actions now use a detached `xdg-open` process with its own
environment: restore the pre-PyInstaller library path and remove AETV/OpenCV Qt
plugin and font overrides. The application's environment is never modified, and
Windows retains QDesktopServices behavior. Both receive-folder and model-folder
actions report filesystem/launch failures at the UI boundary.

The baseline evidence below describes the unfixed 0.1.18 release. PR and release
validation must use the GitHub Actions artifact built from the fix branch.

Investigated revision: `559ad74` (version 0.1.18). The reported version and exact
action are unknown. The baseline investigation made no application changes.
The initial Windows-only investigation was followed by real Lubuntu VM testing
on beastmode on September 7, 2026; see the results immediately below.

The strongest hard-crash explanation is a confirmed cross-thread GUI log update.
A disk-permission failure can specifically trigger this path during prepared-clip
caching. There are also confirmed unhandled filesystem errors and a synchronous
save operation that can explain interrupted actions or an unresponsive window.
These defects are not exclusive to Linux; their presence does not establish the
cause of every reported Lubuntu crash.

## Lubuntu VM results, September 7

An official Lubuntu 24.04.4 live ISO was booted using KVM on beastmode, with
4 virtual CPUs, 8 GiB RAM, and the actual LXQt/X11 desktop. The existing beastmode
development checkout and existing VM were not used or changed. The isolated
workspace is `/home/plaing/aetv-lubuntu-investigation-20260907`. The guest SSH port
was bound only to beastmode's loopback interface.

Both downloaded artifacts matched the publishers' SHA-256 checksums:

```text
Lubuntu 24.04.4 ISO:
5ca3ab769f1538fec7c7d8a5af2e73d3f06ea22f979f6560a9cc4acaf042a5fa
AETV v0.1.18 published Linux archive:
abc353624827f0d4e3f7dd391393a5cd3701fbd89c72cebd504016eaeedfd767
```

### Released binary: normal GUI interaction reproduced a native crash

The unmodified release started, downloaded and verified the standard V8 model,
opened the video chooser, and successfully ran a two-second Clean loopback using
a generated test-pattern MP4. This involved no radio transmission.

Clicking **Save video** after that loopback made the application disappear.
The kernel recorded `AETV[3231]: segfault ... in libQt6Gui.so.6`. The attempted
save had not created an MP4, and the script had not reached the step that removes
write permission. Repeating the same GUI sequence under gdb produced another
SIGSEGV, with the following main-thread frames:

```text
QFontEngineFT::shouldUseDesignMetrics(...)
... Qt font/text shaping frames ...
QWidgetTextControl::selectionRect()
QWidgetTextControl::processEvent(...)
```

The fault is in GUI text processing; it is not a Python PermissionError or an
FFmpeg subprocess failure. Saving is the triggering interaction in these runs,
not proof that the video writer itself corrupts memory.

### Logger isolation and control

A separate harness built the real MainWindow and used its actual
`Station.log -> MainWindow._log -> LogPane.append_line` connection, with hardware
and model startup disabled. A single worker emitted up to 10,000 cache-error
messages with a 1 ms pause between them while the normal GUI event loop ran.
This accelerated workload crashed under gdb:

```text
Thread 1 "python" received signal SIGSEGV, Segmentation fault.
QTextLine::layout_helper(int)
QPlainTextDocumentLayout::layoutBlock(QTextBlock const&)
QPlainTextDocumentLayout::blockBoundingRect(QTextBlock const&) const
```

The same workload with only the logger connection changed to an explicit queued
Qt signal survived and exited normally after 15 seconds. This control was made
in the investigation harness, not in production source or the release binary.
It confirms that the existing logger can cause a native GUI crash. Together with
the packaged crash's font/text stack and the worker log generated during
loopback, it makes the logging bug the leading explanation for the release
failure. A patched release still needs the same GUI sequence tested before
claiming that every packaged crash is fixed.

### Released binary: folder-opening helpers abort

Both **Open model folder** and **Open saved video directory** failed in the
unmodified package while AETV itself remained running. The child process emitted:

```text
QObject::moveToThread: Current thread ... is not the object's thread ...
Could not load the Qt platform plugin "xcb" in
"/home/lubuntu/aetv-test/AETV/_internal/cv2/qt/plugins"
Aborted (core dumped)
```

Lubuntu generated `/var/crash/_usr_bin_pcmanfm-qt.1000.crash`. Launching the same
model folder via `xdg-open` from the clean LXQt session succeeded. This supports
inherited Qt/OpenCV environment contamination as a separate packaging defect.
The failure persisted after the source-test Qt system dependencies were installed.
Fix external desktop launches to use a clean environment, and evaluate using
OpenCV's headless distribution since AETV provides its own PySide GUI.

### Permission checks and source-install prerequisite

All nine initial probes also passed under Lubuntu's real X11 session. Three
additional tests used actual filesystem mode bits as the non-root `lubuntu` user:

- A denied settings replacement preserved the previous settings file.
- A denied MP4 write was caught by ReceivePanel and displayed Permission denied.
- A permitted MP4 write produced a nonempty video successfully.

Thus basic save permissions alone did not reproduce a native crash in those
isolated tests. The packaged permission test was interrupted by the earlier
native crash before permissions were changed; do not count that as a passed
packaged save test.

The pip/source installation initially aborted at QApplication construction
because the live desktop lacked `libxcb-cursor0`; installing it and
`libxkbcommon-x11-0` allowed the source probes to run. The distributed binary
already launched before those packages were added. This is a separate source
installation prerequisite, not the cause of the demonstrated in-GUI release
crash. Source tests used Python 3.12.3 and PySide6 6.11.2.

### Saved evidence and test scope

The local `.build/gui-crash-investigation/evidence/` directory contains
`packaged-save-backtrace.txt`, `native-log-backtrace.txt`,
`native-log-control.txt`, `packaged-desktop.log`, `guest-kernel.log`,
`lubuntu-probes.log`, `real-permission-tests.log`, and environment records.
The harnesses and screenshots are in `.build/gui-crash-investigation/`;
matching artifacts remain in beastmode's dedicated workspace. These are ignored
investigation artifacts rather than production regression tests.

The guest was used only for this investigation. Its runtime was shut down after
evidence collection; downloaded images and reproduction scripts were retained.

Recommended order: fix queued GUI logging first; repair external desktop launch
environments; harden persistence and recorder cleanup; add persistent diagnostics.
Retest a patched Linux release on the same normal file-selection/loopback/save
sequence, including a genuinely unwritable destination.

## Findings

### 1. A cache write error reaches a GUI widget from the background thread

The actual call path is:

```text
TransmitPanel._prepare_clip_batch (aetv-clip-preparer worker)
  -> TxEngine.prepare_clip
  -> save_prepared_clip raises OSError / PermissionError
  -> Station.log("Prepared clip is ready but could not be cached: ...")
  -> MainWindow._log
  -> LogPane.append_line
  -> QPlainTextEdit.appendPlainText / scrollbar update
```

References: `aetv/gui/tx_panel.py:715`, `aetv/station.py:469`,
`aetv/station.py:327`, `aetv/gui/app.py:163`, `aetv/gui/app.py:594`, and
`LogPane.append_line` in `aetv/gui/widgets.py`.

`MainWindow._build` installs a direct Python callback using
`self.station.set_logger(self._log)`. Unlike the panel signals, this callback
does not queue delivery onto the GUI thread. Successful receive autosaving and
TX debug logging also use this connection from workers.

**Initially verified on Windows:** built the actual MainWindow with hardware/model startup disabled,
injected a cache PermissionError while preparing a clip on a worker, and
intercepted `append_line` immediately before widget mutation. The error message
arrived on the worker, not the widget's owning thread. Clip preparation itself
completed. Deliberately corrupting the native widget to force a crash was not
necessary to establish the violation.

Qt specifies that GUI widgets must be used only from the main thread. The
subsequent Lubuntu stress reproduction above confirmed a native crash along this
path. See
[Qt threading documentation](https://doc.qt.io/qt-6/threads-qobject.html).

**Fix priority: first.** Route station logging through a Qt signal connected to a
main-thread slot, and test the real logger connection's thread affinity.

### 2. Settings filesystem errors interrupt GUI transitions

`MainWindow` calls `save_settings` without an error boundary when activating a
model, accepting Settings, changing mode, and closing:
`aetv/gui/app.py:405`, `:426`, `:474`, and `:637`.

**Verified:** injecting PermissionError into mode-change persistence leaves
`settings.mode` changed and its checkpoint cleared, but skips the subsequent
model reload. A real Qt signal invoking this failing callback reached
`sys.excepthook` and returned without terminating the test process. Thus this
explains inconsistent/stuck operation, but is not proof of a native crash.

Startup is different: `MainWindow` calls `load_settings` outside any exception
handler before the event loop. Config-directory creation failure or malformed
JSON escapes startup. Both were verified in isolated probes.

`save_settings` already uses atomic replacement. A denied write need not corrupt
the previous settings file; the missing GUI recovery is the defect.

**Proposed fix:** handle persistence errors explicitly at the GUI boundary,
keep runtime changes coherent, and report whether changes are only in memory.
Recover startup from unreadable/corrupt configuration without silently
overwriting the original file.

### 3. Autosave directory errors bypass cleanup during receive stop

`RxEngine._autosave` resolves/creates the receive directory before its `try`
(`aetv/station.py:1784`). `RxEngine.stop` invokes it before clearing the ring,
closing debug recorders, and publishing the final stopped state (`:1557`).

**Verified:** a directory PermissionError escapes `stop`, does not reach the
autosave error callback, and leaves the debug recorder unclosed and ring retained.
The GUI stop worker catches the exception and emits stop completion, so this can
leave the UI reporting stopped despite incomplete engine cleanup. The blocking
shutdown path has no equivalent catch around `engine.stop`.

TX also performs WAV close and debug JSON writes in an unguarded `finally`
(`aetv/station.py:650`), outside its normal transmission error handler. That
additional path was identified by inspection, not a dynamic probe.

**Proposed fix:** include path creation inside autosave error handling and make
resource cleanup independent of all save/recorder failures, including disk-full
errors during close.

### 4. Manual saving blocks the GUI and duplicates the video buffer

`ReceivePanel.save_current` calls the encoder synchronously
(`aetv/gui/rx_panel.py:416`). `write_mp4` passes `frames.tobytes()` to
`subprocess.run(..., timeout=600)` (`aetv/source.py:663`).

**Verified:** the encoder is called on the QApplication thread. A manual-save
PermissionError is already caught and displayed. Slow encoding/storage can
instead make the window unresponsive while the subprocess runs.

The receiver retains up to 300 GOPs (`aetv/station.py:1702`). At the configured
RGB8 dimensions this is approximately 107 MiB in V8 or 380 MiB in V7. Saving
makes another complete bytes copy, in addition to the existing array, model,
FFmpeg process, and other buffers. The arithmetic is from the configured frame
sizes; peak process memory was not measured. An OOM kill on a low-memory machine
is a separate possibility, not a confirmed explanation of these reports.

**Proposed fix:** save asynchronously and stream bounded chunks to FFmpeg instead
of making a complete bytes copy; keep resource use bounded while RX continues.

### 5. Portable callsigns produce invalid autosave paths

`_autosave` inserts `result.callsign` directly into the filename. A legitimate
callsign such as `VE7ABC/P` becomes a subdirectory component, whose directory is
not created. Verified the constructed path in a probe. This is normally a caught
save failure, not a process crash. Sanitize the filename component as the debug
filename path already does.

## Linux-specific context

Default writable locations are per-user: `${XDG_CONFIG_HOME:-~/.config}/AETV`,
`${XDG_CACHE_HOME:-~/.cache}/aetv`, and `~/AETV/received`. A read-only application
installation alone should not prevent those writes. Incorrect ownership of these
directories, an unwritable configured destination, or a full filesystem can.
There is no evidence that any particular user's directories have these problems.

The file pickers use Qt's default native dialog behavior. If a future report
pins the failure to opening the picker, compare against
`QFileDialog.Option.DontUseNativeDialog` on that Lubuntu installation. Qt documents
this option in its [file dialog reference](https://doc.qt.io/qt-6/qfiledialog.html).
Current Linux CI and packaging smoke tests use `QT_QPA_PLATFORM=offscreen`, which
does not exercise the actual Lubuntu desktop/dialog integration.

Opening a saved-video/model folder also launches the desktop's external handler
through `QDesktopServices`. Frozen Linux applications can pass bundled library
search paths to external programs, a documented
[PyInstaller compatibility issue](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#launching-external-programs-from-the-frozen-application).
The VM results above now confirm a folder-helper abort in the released package;
the full cleanup needed for each subprocess environment remains to be implemented.

## Verification and next evidence

The initial nine isolated probes passed on Windows, Python 3.12.8 / PySide6 6.11.2. Passing
here means the assertions confirmed the observed behavior, including the defects;
it does not mean the defects have been fixed. In that initial phase no local
Linux distribution was available and permission failures were injected. The
later beastmode VM phase added real Linux permissions, the published package,
and native debugging as described above. All actual permission changes were
limited to disposable guest directories.

The local probe file is `.build/gui-crash-investigation/test_observed_failures.py`
(an ignored investigation artifact, separate from the regression suite). Run it
with the project's Python environment:

```text
python -m pytest .build/gui-crash-investigation/test_observed_failures.py -v
```

No persistent Python-exception or Qt crash logging is installed in the GUI.
The existing in-window station log disappears with the process, and waveform
debug captures do not substitute for a traceback. Persistent rotating diagnostics
should include Python/worker exceptions and Qt messages, with a fallback if the
normal log directory is unwritable; native-fault reporting should retain its file
handle for the process lifetime.

For a subsequent report, the simplest first capture from the extracted Linux
application directory is:

```sh
./AETV > "$HOME/aetv-gui.log" 2>&1
```

Record the application version, last action, whether the window vanished or
froze, and terminal exit status immediately after termination. If it vanished,
a native backtrace or kernel OOM record would distinguish the main remaining
possibilities. Fix the confirmed log thread violation and filesystem error
boundaries regardless of whether that evidence becomes available.
