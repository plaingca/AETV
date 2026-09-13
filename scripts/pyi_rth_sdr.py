"""Find bundled libiio even when the host's ldconfig cache has no SDR drivers."""

import ctypes.util
import sys
from pathlib import Path

_original_find_library = ctypes.util.find_library


def _find_library(name):
    if name in {"iio", "libiio.dll"}:
        root = Path(sys._MEIPASS)
        for filename in ("libiio.so.0", "libiio.dll", "libiio.dylib"):
            path = root / filename
            if path.is_file():
                return str(path)
    return _original_find_library(name)


ctypes.util.find_library = _find_library
