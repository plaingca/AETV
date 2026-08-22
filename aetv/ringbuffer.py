"""Thread-safe circular audio buffer shared by capture, decode, and the waterfall.

Copied in spirit from SSTVAE: the audio callback must never block on a
full-buffer copy. `snapshot` publishes two integers under the lock and
copies outside it. The waterfall uses `tail` so a 20 fps display does
not clone tens of seconds of history every frame.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Fixed-length circular float64 audio buffer."""

    def __init__(self, seconds: float, fs: int):
        self.n = max(1, int(seconds * fs))
        self.fs = int(fs)
        self.buf = np.zeros(self.n, dtype=np.float64)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1)
        n = len(chunk)
        if n == 0:
            return
        pos = self.write_pos
        if n >= self.n:
            self.buf[:] = chunk[-self.n :]
            new_pos = 0
        else:
            end = pos + n
            if end <= self.n:
                self.buf[pos:end] = chunk
            else:
                k = self.n - pos
                self.buf[pos:] = chunk[:k]
                self.buf[: end - self.n] = chunk[k:]
            new_pos = end % self.n
        with self.lock:
            self.write_pos = new_pos
            self.total_written += n

    def snapshot(self) -> tuple[np.ndarray, int]:
        with self.lock:
            write_pos = self.write_pos
            total = self.total_written
        if total < self.n:
            return self.buf[:total].copy(), total
        return np.concatenate([self.buf[write_pos:], self.buf[:write_pos]]), total

    def tail(self, n: int) -> np.ndarray:
        n = min(int(n), self.n)
        with self.lock:
            n = min(n, self.total_written)
            if n == 0:
                return np.zeros(0, dtype=np.float64)
            start = self.write_pos - n
            if start >= 0:
                return self.buf[start : self.write_pos].copy()
            return np.concatenate([self.buf[start:], self.buf[: self.write_pos]])

    def clear(self) -> None:
        with self.lock:
            self.buf[:] = 0.0
            self.write_pos = self.total_written % self.n
