"""Portable SDR discovery must not depend on host PATH or installed libraries."""

import ctypes.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from aetv import sdr


def test_nested_bundled_rtl_precedes_host_and_legacy_paths(tmp_path, monkeypatch):
    package = tmp_path / "aetv"
    nested = package / "bin" / "rtlsdr" / "rtl_sdr.exe"
    nested.parent.mkdir(parents=True)
    nested.touch()
    (package / "bin" / "rtl_sdr").touch()
    monkeypatch.setattr(sdr, "__file__", str(package / "sdr.py"))
    monkeypatch.setattr(sdr.shutil, "which", lambda _: "/unrelated/rtl_sdr")
    assert sdr.rtl_executable() == str(nested)


@pytest.mark.parametrize("windows", [False, True])
def test_hook_finds_bundled_iio_without_host_drivers(tmp_path, monkeypatch, windows):
    directory = tmp_path / "aetv/bin/pluto" if windows else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / ("libiio.dll" if windows else "libiio.so.0")
    library.touch()
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "platform", "win32" if windows else "linux")
    monkeypatch.setattr(ctypes.util, "find_library", lambda _: None)
    added = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda p: added.append(p) or object(), raising=False
    )
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/pyi_rth_sdr.py")
    )
    assert ctypes.util.find_library("libiio.dll" if windows else "iio") == str(library)
    assert ctypes.util.find_library("unrelated") is None
    if windows:
        assert str(directory) in added
        assert len(namespace["_dll_directories"]) == 2


def test_frozen_rtl_child_can_find_bundled_vc_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setenv("PATH", "")
    options = sdr._rtl_process_options()
    assert options["env"]["PATH"].split(os.pathsep)[0] == str(tmp_path)
    assert options["creationflags"] == 0x08000000
    assert os.environ["PATH"] == ""
