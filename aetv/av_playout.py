"""Bounded, clocked playout of complete audio/video GOP pairs."""

from collections import deque
import time


class PairedAVPlayout:
    """One queue owns both streams' latency and drops.

    Blind acquisition can release several old GOPs at once. Keep at most four
    complete pairs and release one per second, so video cannot skip ahead of
    an independently growing audio queue. The caller retains recordings before
    enqueueing; this queue only controls live presentation.
    """

    def __init__(self, max_gops=4, *, clock=None):
        self._pending = deque(maxlen=max_gops)
        self._clock = time.monotonic if clock is None else clock
        self._deadline = None
        self.dropped = 0

    def __len__(self):
        return len(self._pending)

    def push(self, item):
        if len(self._pending) == self._pending.maxlen:
            self.dropped += 1
        self._pending.append(item)

    def pop(self, now=None):
        now = self._clock() if now is None else now
        if not self._pending or (self._deadline is not None and now < self._deadline):
            return None
        # Do not burst to catch up after a decoder or GUI stall. Audio hardware
        # still needs one second to render the pair released now.
        self._deadline = now + 1.0
        return self._pending.popleft()

    def clear(self):
        self._pending.clear()
        # Retain the deadline: the already released audio may still be playing.
