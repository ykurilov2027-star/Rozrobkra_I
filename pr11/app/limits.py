"""Ковзне обмеження частоти запитів на клієнта."""

import os
import threading
import time
from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
WINDOW_SECONDS = 60.0
_requests: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


class RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    now = time.monotonic()
    with _lock:
        stale = [key for key, times in _requests.items()
                 if not times or now - times[-1] >= WINDOW_SECONDS]
        for key in stale:
            del _requests[key]
        queue = _requests[client]
        while queue and now - queue[0] >= WINDOW_SECONDS:
            queue.popleft()
        if len(queue) >= PER_MINUTE:
            raise RateLimited(max(1.0, WINDOW_SECONDS - (now - queue[0])))
        queue.append(now)
