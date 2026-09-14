#!/usr/bin/env python3
"""Stage checksum-pinned official SDR runtimes and offline USB setup tools."""

import argparse
import hashlib
import io
import json
import time
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def fetch(entry):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(entry["url"], timeout=60) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    actual = hashlib.sha256(data).hexdigest()
    if actual != entry["sha256"]:
        raise RuntimeError(f"{entry['name']} checksum mismatch: {actual}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).with_name("sdr_windows_dependencies.json")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in entries:
        data = fetch(entry)
        if "members" in entry:
            if entry.get("format") == "tar.zst":
                import zstandard
                with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as stream:
                    with tarfile.open(fileobj=stream, mode="r|") as archive:
                        files = [(entry["members"][member.name], archive.extractfile(member).read())
                                 for member in archive if member.name in entry["members"]]
                if len(files) != len(entry["members"]):
                    raise RuntimeError(f"Missing archive members: {entry['name']}")
            else:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    files = [
                        (target, archive.read(member))
                        for member, target in entry["members"].items()
                    ]
        else:
            files = [(entry["destination"], data)]
        for relative, content in files:
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        print(f"Verified {entry['name']}")
    # Windows resolves a DLL basename once per process. Both native backends
    # must use the same libusb, including HackRF's 1.0.30 raw-I/O API exports.
    usb = (args.output / "runtime/hackrf/libusb-1.0.dll").read_bytes()
    (args.output / "runtime/pluto/libusb-1.0.dll").write_bytes(usb)
    (args.output / "dependencies.json").write_text(
        manifest.read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
