"""Підставний провайдер: поводиться як справжній, але без мережі й
ключа — і збоїть тоді, коли ви просите.

Навіщо він потрібен:

* застосунок запускається й демонструється без ключа і без доступу до
  провайдера — у поїзді, на захисті, в автоматичних тестах;
* збої відтворюються на вимогу. Дочекатися справжнього 429 чи обриву
  посеред streaming можна довго; тут вони стаються за налаштуванням;
* безкоштовний ліміт ключа не витрачається на перевірку кнопок.

Поводитися він має так само, як справжній: ті самі `Result`, `Chunk`,
`ProviderError`; фрагменти — із затримкою, а не всі одразу, інакше
сторінка не покаже, як виглядає streaming. Текст відповіді не мусить
бути розумним — досить правдоподібного: у `samples/requests.json` для
кожного звернення є `reply_example`, а яке саме звернення зараз
обробляється, видно з `Request.metadata`.

Затримки та частка помилок керуються параметрами `MOCK_*` у `.env`.
"""

import os
import random
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    @staticmethod
    def _wait(milliseconds: int, request: Request) -> None:
        deadline = time.monotonic() + max(0, milliseconds) / 1000
        cancelled = request.metadata.get("is_cancelled", lambda: False)
        while time.monotonic() < deadline:
            if cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    @staticmethod
    def _reply(request: Request) -> str:
        request_id = request.metadata.get("request_id")
        for sample in request.metadata.get("samples", []):
            if sample.get("id") == request_id:
                return str(sample.get("reply_example") or "")
        return "Дякуємо за звернення. Ми перевіримо надану інформацію та повідомимо вас про результат."

    @staticmethod
    def _failure(request: Request) -> str | None:
        requested = request.metadata.get("mock_failure_kind")
        rate = min(1.0, max(0.0, float(request.metadata.get("mock_failure_rate", FAILURE_RATE))))
        if requested and random.random() < rate:
            return requested
        if not requested and random.random() < rate:
            return FAILURE_KIND
        return None

    @staticmethod
    def _raise_failure(kind: str) -> None:
        if kind == "mid_stream":
            kind = "unavailable"
        messages = {
            "timeout": "Час очікування відповіді вичерпано.",
            "rate_limit": "Провайдер тимчасово обмежив кількість запитів.",
            "unavailable": "Провайдер тимчасово недоступний.",
            "auth": "Не вдалося авторизуватися у провайдера.",
            "bad_request": "Провайдер не прийняв запит.",
        }
        if kind not in messages:
            kind = "unavailable"
        retryable = kind in {"timeout", "rate_limit", "unavailable"}
        raise ProviderError(kind, messages[kind], retryable=retryable,
                            retry_after=1.5 if kind == "rate_limit" else None)

    def complete(self, request: Request) -> Result:
        """Повернути відповідь цілком — після тих самих затримок і з
        тими самими збоями, що й у `stream`."""
        text = "".join(chunk.text for chunk in self.stream(request))
        return Result(text=text, provider=self.name, model=self.model,
                      finish_reason="stop")

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати текст фрагментами з затримкою `DELAY_MS` між ними."""
        reply = self._reply(request)
        failure = self._failure(request)
        if failure == "timeout":
            self._wait(int((request.metadata.get("attempt_timeout", 20) + 1) * 1000), request)
            self._raise_failure(failure)
        self._wait(FIRST_DELAY_MS, request)
        parts = [reply[i:i + 16] for i in range(0, len(reply), 16)] or [""]
        for index, part in enumerate(parts):
            if request.metadata.get("is_cancelled", lambda: False)():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            if failure and failure != "mid_stream" and index == 0:
                self._raise_failure(failure)
            cutoff = max(1, min(3, len(parts) // 2))
            if failure == "mid_stream" and index >= cutoff:
                self._raise_failure(failure)
            yield Chunk(text=part, finish_reason="stop" if index == len(parts) - 1 else None)
            if index < len(parts) - 1:
                self._wait(DELAY_MS, request)
        if failure == "mid_stream" and len(parts) <= 3:
            self._raise_failure(failure)
