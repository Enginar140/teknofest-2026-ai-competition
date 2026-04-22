"""Sunucu tahmin gönderimi için dakikalık üst sınır (ör. 80/dk)."""

from __future__ import annotations

import time
from collections import deque


class PredictionThrottle:
    """Kayan 60 saniyelik pencerede en fazla `max_per_minute` gönderime izin verir."""

    def __init__(self, max_per_minute: int = 80) -> None:
        self.max_per_minute = max_per_minute
        self._t: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._t and now - self._t[0] > 60.0:
            self._t.popleft()
        if len(self._t) >= self.max_per_minute:
            delay = 60.0 - (now - self._t[0])
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            while self._t and now - self._t[0] > 60.0:
                self._t.popleft()
        self._t.append(time.monotonic())
