#!/usr/bin/env python3
"""Stage checksum-pinned official SDR runtimes and offline USB setup tools."""

import argparse
import hashlib
import io
import json
import time
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
    (args.output / "dependencies.json").write_text(
        manifest.read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
