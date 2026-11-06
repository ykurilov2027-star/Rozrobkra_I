"""Обмеження частоти звертань до власного API.

Ліміт провайдера — спільний на весь застосунок: якщо один користувач
сторінки натисне «скласти чернетку» двадцять разів за хвилину, 429 від
провайдера отримають усі. Тому веб-рівень обмежує кожного клієнта ще до
того, як запит дійде до шлюзу. Це інше обмеження, ніж `MAX_CONCURRENT`
у шлюзі: там — скільки звертань до провайдера виконується одночасно на
всіх, тут — скільки запитів за вікно часу дозволено одному.

Хто такий «клієнт», вирішує веб-рівень: сторінка надсилає свій
ідентифікатор у полі `client`. Обмеження використовує ковзне вікно
і повідомляє час до наступної доступної спроби.
"""

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
    """Ліміт вичерпано. `retry_after` — через скільки секунд пробувати."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    """Чинні обмеження — для сторінки."""
    return {"per_minute": max(1, PER_MINUTE)}


def acquire(client: str) -> None:
    """Зарахувати одне звертання клієнта або підняти `RateLimited`.

    Викликається веб-рівнем перед створенням кожної чернетки.
    """
    now = time.monotonic()
    with _lock:
        history = _requests[client]
        while history and now - history[0] >= WINDOW_SECONDS:
            history.popleft()
        if len(history) >= max(1, PER_MINUTE):
            raise RateLimited(max(1, WINDOW_SECONDS - (now - history[0])))
        history.append(now)
