"""Підставний провайдер для локальних перевірок."""

import json
import os
import random
import time
from typing import Iterator

from dotenv import load_dotenv

from .. import samples
from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    name = "mock"
    model = "mock-1"

    def _reply(self, request: Request) -> str:
        sample = samples.get(request.metadata.get("sample_id") or request.metadata.get("request_id", ""))
        if sample:
            return sample["reply_example"]
        return "Дякуємо за звернення. Ми перевіримо надану інформацію та повернемося з відповіддю."

    def _maybe_fail(self) -> None:
        if FAILURE_RATE > 0 and random.random() < FAILURE_RATE:
            kind = FAILURE_KIND
            messages = {
                "timeout": "Підставний провайдер не відповів вчасно.",
                "rate_limit": "Підставний провайдер тимчасово обмежив запити.",
                "auth": "Підставний провайдер відхилив доступ.",
                "unavailable": "Підставний провайдер тимчасово недоступний.",
            }
            if kind != "mid_stream":
                raise ProviderError(kind, messages.get(kind, "Підставний провайдер завершився помилкою."),
                                    retryable=kind in {"timeout", "rate_limit", "unavailable"},
                                    retry_after=1 if kind == "rate_limit" else None,
                                    detail="MockProvider")

    def complete(self, request: Request) -> Result:
        started = time.perf_counter()
        self._maybe_fail()
        time.sleep(max(0, FIRST_DELAY_MS) / 1000)
        text = self._reply(request)
        in_tokens = max(1, len(json.dumps(request.messages, ensure_ascii=False)) // 3)
        out_tokens = max(1, len(text) // 3)
        return Result(text=text, provider=self.name, model=self.model,
                      usage={"prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                             "total_tokens": in_tokens + out_tokens},
                      finish_reason="stop", elapsed=time.perf_counter() - started)

    def stream(self, request: Request) -> Iterator[Chunk]:
        self._maybe_fail()
        time.sleep(max(0, FIRST_DELAY_MS) / 1000)
        text = self._reply(request)
        pieces = [text[index:index + 36] for index in range(0, len(text), 36)]
        for index, piece in enumerate(pieces):
            if index:
                time.sleep(max(0, DELAY_MS) / 1000)
            if FAILURE_RATE > 0 and FAILURE_KIND == "mid_stream" and index == max(1, len(pieces) // 2):
                raise ProviderError("unavailable", "Потік від провайдера перервався.",
                                    retryable=True, detail="MockProvider")
            usage = None
            if index == len(pieces) - 1:
                in_tokens = max(1, len(json.dumps(request.messages, ensure_ascii=False)) // 3)
                out_tokens = max(1, len(text) // 3)
                usage = {"prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                         "total_tokens": in_tokens + out_tokens}
            yield Chunk(piece, usage=usage, finish_reason="stop" if index == len(pieces) - 1 else None)
