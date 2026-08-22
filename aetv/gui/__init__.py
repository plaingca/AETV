"""AETV ham-station GUI (PySide6)."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .app import main as _main

    return _main(argv)
