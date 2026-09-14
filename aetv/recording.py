"""Bounded background writes for optional live debug recordings."""

import queue
import threading
import time


class BufferedWriter:
    """Keep disk stalls off capture/decoder threads; retain a valid prefix.

    After an overflow or disk failure, stop accepting data rather than silently
    joining nonadjacent audio. The owner exposes ``health`` in its diagnostics.
    Only the worker closes the file, including after a timed-out close call.
    """

    def __init__(self, write, close, *, max_bytes=8 * 1024 * 1024):
        self._write, self._close = write, close
        self._max_bytes = max_bytes
        self._queued = 0
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._end = object()
        self._closed = False
        self.health = dict(queue_high_water_bytes=0, write_max_ms=0.,
                           rejected_bytes=0, error="")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="aetv-debug-recorder")
        self._thread.start()

    def write(self, data):
        with self._lock:
            if self._closed or self.health["error"]:
                self.health["rejected_bytes"] += len(data)
                return False
            if self._queued + len(data) > self._max_bytes:
                self.health["error"] = "Debug recording queue exceeded its byte limit; retained prefix only"
                self.health["rejected_bytes"] += len(data)
                return False
            self._queued += len(data)
            self.health["queue_high_water_bytes"] = max(
                self.health["queue_high_water_bytes"], self._queued)
            self._queue.put_nowait(data)
            return True

    def _run(self):
        failed = False
        try:
            while True:
                data = self._queue.get()
                if data is self._end:
                    return
                with self._lock:
                    self._queued -= len(data)
                if failed:
                    continue
                before = time.monotonic()
                try:
                    self._write(data)
                except Exception as error:
                    self.health["error"] = f"Debug recording failed: {error}"
                    failed = True
                self.health["write_max_ms"] = max(
                    self.health["write_max_ms"], 1000 * (time.monotonic() - before))
        finally:
            try:
                self._close()
            except Exception as error:
                self.health["error"] = f"Debug recording close failed: {error}"

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put_nowait(self._end)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self.health["error"] = "Debug recording still draining after five seconds"
