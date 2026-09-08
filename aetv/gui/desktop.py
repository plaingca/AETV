"""Open system desktop applications without exporting AETV's Qt runtime."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, QUrl
from PySide6.QtGui import QDesktopServices


def _desktop_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # OpenCV sets these even in source installs. Its Qt plugins/fonts belong to
    # OpenCV, not to the system's file manager (which may use another Qt version).
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        environment.pop(name, None)
    if getattr(sys, "frozen", False):
        original = environment.pop("LD_LIBRARY_PATH_ORIG", "")
        if original:
            environment["LD_LIBRARY_PATH"] = original
        else:
            environment.pop("LD_LIBRARY_PATH", None)
        for name in ("QT_PLUGIN_PATH", "QML_IMPORT_PATH", "QML2_IMPORT_PATH"):
            environment.pop(name, None)
    return environment


def open_directory(folder: Path) -> bool:
    """Launch the desktop file manager without changing this process's env."""
    folder = folder.resolve()
    return _open_external(QUrl.fromLocalFile(str(folder)), str(folder))


def open_web_url(address: str) -> bool:
    """Open a browser with system libraries, just like the folder opener."""
    url = QUrl(address)
    if not url.isValid() or url.scheme() not in {"http", "https"} or not url.host():
        return False
    return _open_external(url, url.toString(QUrl.ComponentFormattingOption.FullyEncoded))


def _open_external(url: QUrl, argument: str) -> bool:
    if not sys.platform.startswith("linux"):
        return QDesktopServices.openUrl(url)

    environment = QProcessEnvironment()
    for name, value in _desktop_environment().items():
        environment.insert(name, value)
    process = QProcess()
    process.setProgram("xdg-open")
    process.setArguments([argument])
    process.setProcessEnvironment(environment)
    return process.startDetached()
