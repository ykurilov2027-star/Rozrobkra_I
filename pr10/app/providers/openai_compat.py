"""Адаптер OpenAI-сумісного SDK до контракту `providers.base`."""

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             timeout=timeout, max_retries=0)

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("retry-after")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_time = parsedate_to_datetime(value)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @classmethod
    def _convert_error(cls, exc: Exception) -> ProviderError:
        retry_after = cls._retry_after(exc)
        if isinstance(exc, APITimeoutError):
            kind, message, retryable = "timeout", "Час очікування відповіді вичерпано.", True
        elif isinstance(exc, RateLimitError):
            kind, message, retryable = "rate_limit", "Провайдер тимчасово обмежив кількість запитів.", True
        elif isinstance(exc, AuthenticationError):
            kind, message, retryable = "auth", "Не вдалося авторизуватися у провайдера.", False
        elif isinstance(exc, BadRequestError):
            kind, message, retryable = "bad_request", "Провайдер не прийняв запит.", False
        elif isinstance(exc, InternalServerError):
            kind, message, retryable = "unavailable", "Провайдер тимчасово недоступний.", True
        elif isinstance(exc, APIConnectionError):
            kind, message, retryable = "unavailable", "Не вдалося з’єднатися з провайдером.", True
        elif isinstance(exc, APIStatusError):
            status = exc.status_code
            if status == 408:
                kind, message, retryable = "timeout", "Час очікування відповіді вичерпано.", True
            elif status == 429:
                kind, message, retryable = "rate_limit", "Провайдер тимчасово обмежив кількість запитів.", True
            elif status in (401, 403):
                kind, message, retryable = "auth", "Не вдалося авторизуватися у провайдера.", False
            elif status >= 500:
                kind, message, retryable = "unavailable", "Провайдер тимчасово недоступний.", True
            else:
                kind, message, retryable = "bad_request", "Провайдер не прийняв запит.", False
        else:
            kind, message, retryable = "unknown", "Сталася непередбачена помилка провайдера.", False
        return ProviderError(kind, message, retryable=retryable,
                             retry_after=retry_after, detail=type(exc).__name__)

    def _create(self, request: Request, stream: bool):
        return self.client.chat.completions.create(
            model=request.model or self.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            timeout=request.metadata.get("attempt_timeout", self.timeout),
            stream=stream,
        )

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        started = time.monotonic()
        try:
            response = self._create(request, False)
            choice = response.choices[0]
            usage = response.usage.model_dump() if response.usage else None
            return Result(text=choice.message.content or "", provider=self.name,
                          model=response.model or request.model or self.model,
                          usage=usage, finish_reason=choice.finish_reason,
                          elapsed=time.monotonic() - started)
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise self._convert_error(exc) from None

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        stream = None
        try:
            stream = self._create(request, True)
            for item in stream:
                usage = item.usage.model_dump() if item.usage else None
                if not item.choices:
                    yield Chunk(text="", usage=usage)
                    continue
                choice = item.choices[0]
                yield Chunk(text=choice.delta.content or "", usage=usage,
                            finish_reason=choice.finish_reason)
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise self._convert_error(exc) from None
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close:
                    try:
                        close()
                    except Exception as exc:
                        raise self._convert_error(exc) from None
