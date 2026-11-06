"""Провайдер для OpenAI-сумісного Chat Completions API."""

import time
from typing import Iterator

from openai import (APIConnectionError, APITimeoutError, AuthenticationError,
                    BadRequestError, InternalServerError, OpenAI, RateLimitError)

from .base import Chunk, ProviderError, Request, Result


def _usage_dict(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage)


def _provider_error(exc: Exception) -> ProviderError:
    if isinstance(exc, APITimeoutError):
        kind, message, retryable = "timeout", "Провайдер не відповів вчасно.", True
    elif isinstance(exc, RateLimitError):
        kind, message, retryable = "rate_limit", "Провайдер тимчасово обмежив запити.", True
    elif isinstance(exc, (APIConnectionError, InternalServerError)):
        kind, message, retryable = "unavailable", "Провайдер тимчасово недоступний.", True
    elif isinstance(exc, AuthenticationError):
        kind, message, retryable = "auth", "Не вдалося пройти автентифікацію у провайдера.", False
    elif isinstance(exc, BadRequestError):
        kind, message, retryable = "bad_request", "Провайдер не прийняв запит.", False
    else:
        kind, message, retryable = "unknown", "Сталася помилка під час звернення до провайдера.", False
    retry_after = None
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            retry_after = float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    return ProviderError(kind, message, retryable=retryable,
                         retry_after=retry_after, detail=type(exc).__name__)


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def complete(self, request: Request) -> Result:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=request.model or self.model, messages=request.messages,
                temperature=request.temperature, max_tokens=request.max_tokens,
            )
        except Exception as exc:
            raise _provider_error(exc) from exc
        choice = response.choices[0]
        return Result(
            text=choice.message.content or "", provider=self.name,
            model=response.model or request.model or self.model,
            usage=_usage_dict(response.usage), finish_reason=choice.finish_reason,
            elapsed=time.perf_counter() - started,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        sent = False

        def consume(response) -> Iterator[Chunk]:
            nonlocal sent
            for item in response:
                text = ""
                finish_reason = None
                if item.choices:
                    choice = item.choices[0]
                    text = choice.delta.content or ""
                    finish_reason = choice.finish_reason
                usage = _usage_dict(getattr(item, "usage", None))
                sent = sent or bool(text)
                yield Chunk(text=text, usage=usage, finish_reason=finish_reason,
                            model=getattr(item, "model", None))

        try:
            response = self.client.chat.completions.create(
                model=request.model or self.model, messages=request.messages,
                temperature=request.temperature, max_tokens=request.max_tokens,
                stream=True, stream_options={"include_usage": True},
            )
            yield from consume(response)
        except BadRequestError as exc:
            if sent:
                raise _provider_error(exc) from exc
            try:
                response = self.client.chat.completions.create(
                    model=request.model or self.model, messages=request.messages,
                    temperature=request.temperature, max_tokens=request.max_tokens, stream=True,
                )
                yield from consume(response)
            except Exception as retry_exc:
                raise _provider_error(retry_exc) from retry_exc
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise _provider_error(exc) from exc
