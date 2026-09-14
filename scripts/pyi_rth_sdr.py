"""Prefer the bundled libiio and its DLL dependencies on every platform."""

import ctypes.util
import os
import sys
from pathlib import Path

_original_find_library = ctypes.util.find_library
_root = Path(sys._MEIPASS)
_pluto = _root / "aetv" / "bin" / "pluto"
# Keep the handles alive: closing them removes the DLL search directories.
_dll_directories = []
if sys.platform == "win32":
    for _directory in (_root, _pluto):
        if _directory.is_dir():
            _dll_directories.append(os.add_dll_directory(str(_directory)))


def _find_library(name):
    if name in {"iio", "libiio.dll"}:
        for root in (_pluto, _root):
            for filename in ("libiio.so.0", "libiio.dll", "libiio.dylib"):
                path = root / filename
                if path.is_file():
                    return str(path)
    return _original_find_library(name)


ctypes.util.find_library = _find_library
