#!/usr/bin/env python3
"""Exercise real Hamlib without any radio, serial port, or RF hardware.

Hardcoded model 1, CAT-simulated PTT, loopback-only owned daemon. No option can
select a real radio or an existing rigctld server. Dependencies: Hamlib 4.x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aetv.cat import CatError, HamlibDirect, RigctldClient, _find_hamlib_library


def exercise(client):
    result = {"mode": client.get_mode(), "initial_ptt": client.get_ptt(), "frequencies_hz": [], "ptt_states": []}
    if result["initial_ptt"]:
        raise RuntimeError("fresh dummy unexpectedly started keyed")
    for frequency in (7_088_000, 14_230_000, 1_296_000_000):
        client.set_frequency_hz(frequency)
        observed = client.get_frequency_hz()
        if observed != frequency:
            raise RuntimeError(f"frequency mismatch: requested {frequency}, read {observed}")
        result["frequencies_hz"].append(observed)
    for state in (True, False, True, False):
        client.set_ptt(state)
        observed = client.get_ptt()
        if observed != state:
            raise RuntimeError(f"PTT mismatch: requested {state}, read {observed}")
        result["ptt_states"].append(observed)
    return result


def validate():
    executable = shutil.which("rigctld")
    if not executable:
        raise RuntimeError("rigctld is missing. Install official Hamlib 4.x utilities (Debian: libhamlib-utils and libhamlib4t64), or add your extracted runtime's usr/bin to PATH and usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH before starting Python.")
    result = {"platform": platform.platform(), "model": 1, "rf_hardware_used": False, "library": _find_hamlib_library(), "version": subprocess.check_output([executable, "--version"], text=True, timeout=5).strip()}
    native = HamlibDirect(1, "")
    try:
        result["native"] = exercise(native)
    finally:
        native.close()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen([executable, "-m", "1", "-P", "RIG", "-T", "127.0.0.1", "-t", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = observer = None
    try:
        deadline = time.monotonic() + 5
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"dummy daemon exited with code {process.returncode}")
            try:
                client = RigctldClient("127.0.0.1", port, timeout=0.5)
                break
            except CatError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("dummy daemon did not start within 5 seconds")
                time.sleep(0.05)
        result["tcp"] = exercise(client)
        observer = RigctldClient("127.0.0.1", port, timeout=0.5)
        client.set_ptt(True)
        client.close()
        result["tcp"]["ptt_after_close"] = observer.get_ptt()
        if result["tcp"]["ptt_after_close"]:
            raise RuntimeError("PTT remained on after close")
    finally:
        if client is not None:
            client.close()
        if observer is not None:
            observer.close()
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    result["daemon_stopped"] = process.poll() is not None
    result["status"] = "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also save the validation report")
    args = parser.parse_args()
    try:
        result = validate()
    except (OSError, CatError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Hamlib dummy validation failed: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
