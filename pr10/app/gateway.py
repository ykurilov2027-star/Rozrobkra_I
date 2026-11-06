"""Центральний шлюз провайдера з таймаутами, повторами й fallback."""

import os
import queue
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Chunk, Provider, ProviderError, Request, Result

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))
_provider_slots = threading.Semaphore(max(1, MAX_CONCURRENT))

# Параметри генерації — спільні для всіх провайдерів.
MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу. Ключа тут
    немає і бути не може."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Зібрати провайдера за назвою: `openai` — з налаштувань доступу в
    `.env`, `mock` — підставний. Інша назва — та сама адреса й ключ, але
    інша модель (для запасного варіанту)."""
    if name == "mock":
        from .providers.mock import MockProvider

        return MockProvider()
    from .providers.openai_compat import OpenAICompatProvider
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not base_url or not model:
        raise ProviderError("bad_request", "Не налаштовано адресу або модель провайдера.")
    if not api_key:
        raise ProviderError("auth", "Не налаштовано ключ доступу до провайдера.")
    selected_model = model if name == "openai" else name
    return OpenAICompatProvider(base_url, api_key, selected_model, TIMEOUT)


def _attempt(provider: Provider, request: Request, on_delta: Callable[[str], None],
             is_cancelled: Callable[[], bool], total_deadline: float,
             request_started: float) -> tuple[str, dict | None, str | None, float | None]:
    while True:
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            break
        if _provider_slots.acquire(timeout=min(0.1, remaining)):
            break
    if remaining <= 0:
        raise ProviderError("timeout", "Час очікування в черзі до провайдера вичерпано.",
                            retryable=True)
    if is_cancelled():
        _provider_slots.release()
        raise ProviderError("cancelled", "Генерацію скасовано.")

    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    metadata = dict(request.metadata)
    metadata.update({
        "is_cancelled": lambda: is_cancelled() or stop.is_set(),
        "attempt_timeout": min(TIMEOUT, max(0.01, remaining)),
    })
    attempt_request = Request(
        messages=request.messages, max_tokens=request.max_tokens,
        temperature=request.temperature, model=request.model,
        metadata=metadata,
    )
    started = time.monotonic()

    def produce() -> None:
        try:
            stream = iter(provider.stream(attempt_request))
            try:
                for chunk in stream:
                    if stop.is_set():
                        break
                    events.put(("chunk", chunk))
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()
            events.put(("done", None))
        except ProviderError as exc:
            events.put(("error", exc))
        except Exception as exc:
            events.put(("error", ProviderError(
                "unknown", "Сталася непередбачена помилка під час генерації.",
                detail=type(exc).__name__,
            )))
        finally:
            _provider_slots.release()

    thread = threading.Thread(target=produce, daemon=True)
    try:
        thread.start()
    except RuntimeError as exc:
        _provider_slots.release()
        raise ProviderError("unavailable", "Не вдалося запустити звернення до провайдера.",
                            retryable=True, detail=type(exc).__name__) from None
    text: list[str] = []
    usage = None
    finish_reason = None
    first_token = None
    last_chunk = started
    try:
        while True:
            if is_cancelled():
                stop.set()
                raise ProviderError("cancelled", "Генерацію скасовано.")
            now = time.monotonic()
            attempt_left = min(TIMEOUT - (now - started), total_deadline - now)
            idle_left = IDLE_TIMEOUT - (now - last_chunk)
            if attempt_left <= 0:
                stop.set()
                raise ProviderError("timeout", "Час очікування відповіді вичерпано.",
                                    retryable=True)
            if idle_left <= 0:
                stop.set()
                raise ProviderError("timeout", "Потік відповіді надто довго не надсилав фрагментів.",
                                    retryable=True)
            try:
                event, value = events.get(timeout=min(0.1, attempt_left, idle_left))
            except queue.Empty:
                continue
            if event == "chunk":
                chunk: Chunk = value
                last_chunk = time.monotonic()
                text.append(chunk.text)
                usage = chunk.usage or usage
                finish_reason = chunk.finish_reason or finish_reason
                if chunk.text:
                    if first_token is None:
                        first_token = time.monotonic() - request_started
                    on_delta(chunk.text)
            elif event == "error":
                raise value
            else:
                return "".join(text), usage, finish_reason, first_token
    finally:
        stop.set()


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    """Згенерувати відповідь, дотримуючись таймаутів, спроб, обмеження
    одночасності й запасного варіанту.

    `on_delta` отримує кожен фрагмент тексту, `on_status` — зміни стану
    (наприклад, `{"attempt": 2, "provider": "openai"}` або
    `{"fallback": True}`), `is_cancelled` опитується між фрагментами й
    між спробами. Сигнатура — відправна точка: якщо вам зручніше
    генератор чи черга, змініть і її, і `app/jobs.py`.

    Повертає `Result` або підіймає `ProviderError` — уже остаточний,
    після всіх спроб і запасного варіанту. Інших винятків назовні не
    виходить.
    """
    started = time.monotonic()
    deadline = started + max(0.01, TOTAL_TIMEOUT)
    attempts = 0
    last_error = ProviderError("unknown", "Не вдалося створити запит.")

    def run_provider(selected: Provider, fallback: bool, max_attempts: int) -> Result:
        nonlocal attempts, last_error
        for index in range(max_attempts):
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = ProviderError("timeout", "Вичерпано загальний час генерації.",
                                           retryable=True)
                break
            attempts += 1
            on_status({
                "status": "running", "attempt": attempts,
                "max_attempts": MAX_ATTEMPTS + (1 if fallback else 0), "provider": selected.name,
                "model": getattr(selected, "model", MODEL),
                "fallback": fallback,
            })
            try:
                text, usage, finish_reason, first_token = _attempt(
                    selected, request, on_delta, is_cancelled, deadline, started)
                return Result(
                    text=text, provider=selected.name,
                    model=getattr(selected, "model", MODEL) or "unknown",
                    usage=usage, finish_reason=finish_reason,
                    elapsed=time.monotonic() - started, fallback=fallback,
                    attempts=attempts, first_token=first_token,
                    max_attempts=MAX_ATTEMPTS + (1 if fallback else 0),
                )
            except ProviderError as exc:
                if exc.kind == "cancelled" or is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.")
                last_error = exc
                temporary = exc.kind in {"timeout", "rate_limit", "unavailable"}
                if exc.retryable and temporary and index + 1 < max_attempts:
                    on_status({"status": "running", "attempt": attempts,
                               "max_attempts": MAX_ATTEMPTS + (1 if fallback else 0), "provider": selected.name,
                               "fallback": fallback, "reset": True,
                               "detail": "Повтор після тимчасового збою"})
                    delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** index))
                    delay *= random.uniform(0.8, 1.2)
                    if exc.retry_after is not None:
                        delay = max(delay, exc.retry_after)
                    remaining = deadline - time.monotonic()
                    if delay >= remaining:
                        break
                    if is_cancelled():
                        raise ProviderError("cancelled", "Генерацію скасовано.")
                    end_wait = time.monotonic() + delay
                    while time.monotonic() < end_wait:
                        if is_cancelled():
                            raise ProviderError("cancelled", "Генерацію скасовано.")
                        time.sleep(min(0.05, end_wait - time.monotonic()))
                    continue
                break
        raise last_error

    try:
        provider = make_provider(PROVIDER)
        return run_provider(provider, False, max(1, MAX_ATTEMPTS))
    except ProviderError as primary_error:
        if primary_error.kind == "cancelled" or is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        if FALLBACK.lower() in {"", "none", PROVIDER.lower()}:
            raise primary_error
        if time.monotonic() >= deadline:
            raise primary_error
        try:
            fallback_provider = make_provider(FALLBACK)
            on_status({"status": "running", "attempt": attempts + 1,
                       "max_attempts": MAX_ATTEMPTS + 1, "provider": fallback_provider.name,
                       "model": getattr(fallback_provider, "model", FALLBACK),
                       "fallback": True, "reset": True,
                       "detail": "Використовується запасний варіант"})
            return run_provider(fallback_provider, True, 1)
        except ProviderError as fallback_error:
            if fallback_error.kind == "cancelled":
                raise
            raise last_error
