#!/usr/bin/env python3
"""Build the pinned native HackRF library; never install it into the host OS."""

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "2026.01.3"
URL = f"https://github.com/greatscottgadgets/hackrf/releases/download/v{RELEASE}/hackrf-{RELEASE}.zip"
SHA256 = "601599e14d03cc1900cf9a1c832957a997dc0d488f8299581bd73b40718481a5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"hackrf-{RELEASE}.zip"
    if not archive.is_file():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("HackRF source checksum mismatch")
    source = root / f"hackrf-{RELEASE}"
    with zipfile.ZipFile(archive) as package:
        # Copy only the required, authenticated source members, not firmware.
        for member in package.namelist():
            relative = Path(member)
            if member.endswith("/") or ".." in relative.parts or relative.is_absolute():
                continue
            if not member.startswith(f"hackrf-{RELEASE}/host/") and member != f"hackrf-{RELEASE}/COPYING":
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(member))
    build = root / "build"
    subprocess.run(["cmake", "-S", str(source / "host/libhackrf"), "-B", str(build),
                    "-DENABLE_STATIC_LIB=OFF", "-DINSTALL_UDEV_RULES=OFF",
                    "-DCMAKE_BUILD_TYPE=Release", f"-DRELEASE={RELEASE}"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
    shutil.copyfile(build / "src/libhackrf.so.0", root / "libhackrf.so.0")
    rules = (source / "host/libhackrf/53-hackrf.rules.in").read_text()
    (root / "53-hackrf.rules").write_text(rules.replace("@HACKRF_GROUP@", "plugdev"))
    shutil.copyfile(source / "host/libhackrf/src/hackrf.h", root / "libhackrf-BSD-header.h")
    shutil.copyfile(source / "COPYING", root / "HackRF-COPYING.txt")


if __name__ == "__main__":
    main()
