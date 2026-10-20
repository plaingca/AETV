# Desktop layout review — 2026-10-20

Layout fix for [issue #39](https://github.com/plaingca/AETV/issues/39), approved for release in v0.1.26.

## Behavior

- Send, Cancel, QRL, mode selection, and receive actions stay outside the scrollable setup pages. Mode fields and receive buttons wrap to additional rows when wider font metrics require it.
- Transmit setup uses Source, Clips, Audio, and Route tabs. Only the selected live-source controls are shown.
- The receive setup area sizes itself to the selected source, leaving more room for received video.
- RF details expand on demand; the short station log can be hidden through View → Log.
- Clip thumbnails retain their aspect ratio and source pixels. The bank reflows from four to eight columns as the available logical width grows.
- Initial window size respects the available desktop. Larger windows allocate additional space to video.

## Captures

These are actual Qt/X11 window captures using the Fusion style and 9-point fonts on Linux virtual desktops. Video and clip previews are generated test cards. Device discovery, model startup, camera/audio/RF operations, and settings writes are disabled. These captures do not establish native Windows behavior.

Each capture leaves 64 logical pixels for a title bar and taskbar. The PNG therefore shows the application client area rather than pretending the entire monitor is available. At 200% scaling this reserves 128 physical pixels. Native PNGs are preserved, with dimensions, visible-control bounds, clip counts, and SHA-256 in adjacent JSON files.

All capture checks require Send/Cancel/QRL, receive actions, mode selection, both previews, PTT status, and the RF-details button to fit in the window. Clips captures additionally require all eight cells to fit simultaneously without scrolling.

| Desktop | Scaling | Page | Captured pixels | Screenshot |
| --- | --- | --- | --- | --- |
| 1366×768 | 100% | Clips | 1366×704 | [PNG](evidence/gui-layout-20261020/1366x768-clips.png) · [geometry](evidence/gui-layout-20261020/1366x768-clips.json) |
| 1366×768 | 100% | Source | 1366×704 | [PNG](evidence/gui-layout-20261020/1366x768-source.png) · [geometry](evidence/gui-layout-20261020/1366x768-source.json) |
| 1366×768 | 100% | A/V audio | 1366×704 | [PNG](evidence/gui-layout-20261020/1366x768-av-audio.png) · [geometry](evidence/gui-layout-20261020/1366x768-av-audio.json) |
| 1366×768 | 100% | Expanded RF details | 1366×704 | [PNG](evidence/gui-layout-20261020/1366x768-rf-details.png) · [geometry](evidence/gui-layout-20261020/1366x768-rf-details.json) |
| 1920×1080 | 100% | Clips | 1920×1016 | [PNG](evidence/gui-layout-20261020/1920x1080-clips.png) · [geometry](evidence/gui-layout-20261020/1920x1080-clips.json) |
| 1920×1080 | 125% | Clips | 1920×1000 | [PNG](evidence/gui-layout-20261020/1920x1080-scale125-clips.png) · [geometry](evidence/gui-layout-20261020/1920x1080-scale125-clips.json) |
| 2560×1440 | 100% | Clips | 2560×1376 | [PNG](evidence/gui-layout-20261020/2560x1440-clips.png) · [geometry](evidence/gui-layout-20261020/2560x1440-clips.json) |
| 3840×2160 | 100% | Clips | 3840×2096 | [PNG](evidence/gui-layout-20261020/3840x2160-clips.png) · [geometry](evidence/gui-layout-20261020/3840x2160-clips.json) |
| 3840×2160 | 200% | Clips | 3840×2032 | [PNG](evidence/gui-layout-20261020/3840x2160-scale200-clips.png) · [geometry](evidence/gui-layout-20261020/3840x2160-scale200-clips.json) |

## Reproduce

From a source environment with the GUI and development extras:

```sh
xvfb-run -a -s '-screen 0 1366x768x24 -dpi 96' env \
  QT_QPA_PLATFORM=xcb QT_SCALE_FACTOR=1 PYTHONPATH=. \
  python scripts/capture_gui_layout.py --output /tmp/aetv-laptop.png --page Clips
```

Use `--page Source`, `--page Audio --av`, or `--rf-details` for the other states. For scaled 4K, use a 3840×2160 Xvfb screen and `QT_SCALE_FACTOR=2`.

## Regression checks

Layout tests cover repeated laptop/1080p/1440p/4K resizing, 9/12/18-point fonts, Fusion/Windows Qt styles, video/A/V modes, RF-detail expansion, source-row visibility, and clicking a thumbnail after the bank reflows. The Windows Qt style on Linux is not a native Windows test.

Initial approved layout regression: **550 passed, 2 skipped** in 148.29 seconds.
After integrating v0.1.25 HF tuning for the v0.1.26 release, the complete suite
passed **569 tests, with 2 skipped**, in 146.84 seconds. Both runs used bounded
CPU threads and the offscreen Qt platform:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  QT_QPA_PLATFORM=offscreen python -m pytest -q --tb=short
```

All **nine** native X11 captures separately passed their visible-control checks;
every Clips capture showed all eight slots at once. PNG dimensions and SHA-256
values match the adjacent geometry reports. Whitespace checks passed.

The short-transmission GUI regression waits for playback completion with a
one-second bound instead of assuming the hosted runner dispatches both Qt
timers within 60 ms. Production playout timing is unchanged. The focused
widget/layout rerun passed all 20 checks.

Hosted GUI checks run before the remaining suite, use installed Segoe UI
metrics on Windows, and explicitly dispose test windows between cases.
