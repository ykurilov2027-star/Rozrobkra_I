"""Шлюз провайдерів із кешем, бюджетом, повторами та телеметрією."""

import hashlib
import logging
import os
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from . import budget, cache, costs
from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

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
MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
_slots = threading.BoundedSemaphore(max(1, MAX_CONCURRENT))
_log = logging.getLogger("app.gateway")


def settings() -> dict:
    return {"provider": PROVIDER, "fallback": FALLBACK, "model": MODEL, "timeout": TIMEOUT,
            "idle_timeout": IDLE_TIMEOUT, "total_timeout": TOTAL_TIMEOUT,
            "max_attempts": MAX_ATTEMPTS, "max_concurrent": MAX_CONCURRENT}


def make_provider(name: str) -> Provider:
    if name == "mock":
        return MockProvider()
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")
    if not base_url or not api_key:
        raise ProviderError("auth", "Налаштування доступу до провайдера неповні.", retryable=False)
    provider = OpenAICompatProvider(base_url, api_key, name if name not in {"openai", PROVIDER} else MODEL or "",
                                   TIMEOUT)
    if name not in {"openai", PROVIDER}:
        provider.model = name
    return provider


def _client_id(request: Request) -> str:
    raw = str(request.metadata.get("client", "unknown"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _log_attempt(request: Request, *, provider: str, attempt: int, started: float,
                 queued_ms: float, first_ms: float | None, result: Result | None,
                 error: ProviderError | None, is_fallback: bool) -> None:
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    usage = (result.usage if result else None) or (error.usage if error else None)
    prompt = "\n".join(str(m.get("content", "")) for m in request.messages)
    partial = result.text if result else error.partial_text if error else ""
    if error and error.kind in {"auth", "bad_request", "cancelled", "rate_limit"} and usage is None:
        cost, estimated = 0.0, False
    else:
        cost, estimated = costs.estimate(usage, provider=provider,
                                         model=result.model if result else None,
                                         text=partial, prompt=prompt)
    if result:
        result.cost, result.cost_estimated = cost, estimated
        usage_for_log, usage_estimated = costs.estimate_usage(usage, prompt=prompt, text=result.text)
        result.usage = usage_for_log
    elif usage is None and error and error.kind in {"auth", "bad_request", "cancelled", "rate_limit"}:
        usage_for_log = {"prompt_tokens": 0, "completion_tokens": 0}
        usage_estimated = False
    else:
        usage_for_log, usage_estimated = costs.estimate_usage(usage, prompt=prompt, text=partial)
    billable_cost = cost
    budget.record(billable_cost)
    request.metadata.setdefault("_attempt_costs", []).append(cost or 0.0)
    request.metadata["_estimated_cost"] = request.metadata.get("_estimated_cost", False) or estimated
    status = "success" if result else "failed"
    level = logging.INFO if result else (logging.WARNING if error and error.retryable else logging.ERROR)
    fields = {
        "draft_id": request.metadata.get("draft_id"),
        "request_id": request.metadata.get("http_request_id"),
        "client": _client_id(request),
        "provider": provider,
        "model": result.model if result else request.metadata.get("actual_model") or request.model,
        "prompt_version": request.metadata.get("prompt_version"),
        "attempt": attempt,
        "queue_ms": round(queued_ms, 1),
        "first_token_ms": round(first_ms, 1) if first_ms is not None else None,
        "duration_ms": elapsed_ms,
        "generation_ms": round(max(0.0, elapsed_ms - (first_ms or elapsed_ms)), 1),
        "status": status,
        "kind": error.kind if error else None,
        "fallback": provider if is_fallback else None,
        "tokens": {
            "input": usage_for_log.get("prompt_tokens"),
            "output": usage_for_log.get("completion_tokens"),
            "cached": (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if isinstance((usage or {}).get("prompt_tokens_details"), dict) else 0,
            "estimated": usage_estimated,
        },
        "cost": cost,
        "cost_estimated": estimated,
        "cached": False,
        "cache_key": request.metadata.get("cache_key"),
    }
    _log.log(level, "llm.attempt", extra=fields)


def _try_provider(provider: Provider, name: str, request: Request, attempt: int,
                  on_delta: Callable[[str], None], on_status: Callable[[dict], None],
                  is_cancelled: Callable[[], bool],
                  deadline: float, is_fallback: bool) -> Result:
    queue_start = time.perf_counter()
    remaining = max(0.0, deadline - queue_start)
    if not _slots.acquire(timeout=remaining):
        raise ProviderError("timeout", "Час очікування в черзі вичерпано.", retryable=True)
    queued_ms = (time.perf_counter() - queue_start) * 1000
    started = time.perf_counter()
    try:
        request.metadata["actual_model"] = getattr(provider, "model", name)
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
        budget.check()
        on_status({"status": "running", "attempt": attempt, "provider": name, "fallback": is_fallback})
        chunks = []
        usage = None
        response_model = getattr(provider, "model", name)
        first_ms = None
        last_chunk = time.perf_counter()
        try:
            for chunk in provider.stream(request):
                now = time.perf_counter()
                if now > deadline:
                    raise ProviderError("timeout", "Перевищено загальний час генерації.", retryable=True)
                if now - last_chunk > IDLE_TIMEOUT:
                    raise ProviderError("timeout", "Потік провайдера надто довго не надсилав даних.", retryable=True)
                if is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
                if first_ms is None and chunk.text:
                    first_ms = (now - started) * 1000
                chunks.append(chunk.text)
                response_model = chunk.model or response_model
                if chunk.text:
                    on_delta(chunk.text)
                if chunk.usage:
                    usage = chunk.usage
                last_chunk = now
            text = "".join(chunks)
            if not text:
                raise ProviderError("unavailable", "Провайдер повернув порожню відповідь.", retryable=True)
            result = Result(text=text, provider=name, model=response_model,
                            usage=usage, finish_reason="stop",
                            elapsed=time.perf_counter() - started)
            _log_attempt(request, provider=name, attempt=attempt, started=started, queued_ms=queued_ms,
                         first_ms=first_ms, result=result, error=None, is_fallback=is_fallback)
            return result
        except ProviderError as error:
            error.usage = usage or error.usage
            partial_text = "".join(chunks)
            error.partial_text = partial_text
            _log_attempt(request, provider=name, attempt=attempt, started=started, queued_ms=queued_ms,
                         first_ms=first_ms, result=None, error=error, is_fallback=is_fallback)
            raise
    except budget.BudgetExceeded as exc:
        _log.warning("llm.attempt", extra={
            "draft_id": request.metadata.get("draft_id"), "client": _client_id(request),
            "request_id": request.metadata.get("http_request_id"),
            "provider": name, "model": request.metadata.get("actual_model") or request.model,
            "prompt_version": request.metadata.get("prompt_version"),
            "attempt": attempt, "queue_ms": round(queued_ms, 1), "first_token_ms": None,
            "duration_ms": 0.0, "generation_ms": 0.0, "status": "rejected", "kind": "budget",
            "fallback": name if is_fallback else None, "tokens": None, "cost": 0.0,
            "cost_estimated": False, "cached": False, "cache_key": request.metadata.get("cache_key"),
        })
        raise ProviderError("budget", str(exc), retryable=False, detail="BudgetExceeded",
                            resets_at=exc.resets_at) from exc
    finally:
        _slots.release()


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    started = time.perf_counter()
    request_id = str(request.metadata.get("http_request_id") or "")
    draft_id = request.metadata.get("draft_id")
    prompt_version = str(request.metadata.get("prompt_version", "unknown"))
    primary = PROVIDER
    request.metadata["provider"] = primary
    key = cache.make_key(request, prompt_version)
    request.metadata["cache_key"] = key
    if request.metadata.get("fresh"):
        cache.bypassed()
        _log.info("llm.cache_bypassed", extra={
            "request_id": request.metadata.get("http_request_id"), "draft_id": draft_id,
            "client": _client_id(request), "cache_key": key, "cached": False,
        })
    else:
        cached = cache.get(key)
        if cached:
            cached.cached = True
            on_status({"status": "running", "provider": cached.provider, "cached": True})
            on_delta(cached.text)
            _log.info("llm.cache_hit", extra={
                "draft_id": draft_id, "request_id": request_id, "client": _client_id(request),
                "provider": cached.provider, "model": cached.model, "prompt_version": prompt_version,
                "cache_key": key, "cached": True, "cost": 0.0,
            })
            cached.cost = 0.0
            cached.cost_estimated = False
            return cached
        _log.info("llm.cache_miss", extra={
            "request_id": request.metadata.get("http_request_id"), "draft_id": draft_id,
            "client": _client_id(request), "cache_key": key, "cached": False,
            "enabled": cache.ENABLED,
        })

    primary_provider = make_provider(primary)
    last_error: ProviderError | None = None
    try:
        primary_provider = make_provider(primary)
        providers = [(primary_provider.name, primary_provider, False)]
    except ProviderError as error:
        last_error = error
        providers = []
        if FALLBACK not in {"none", "", primary}:
            fallback_provider = make_provider(FALLBACK)
            providers.append((fallback_provider.name, fallback_provider, True))
    attempt_total = 0
    deadline = started + TOTAL_TIMEOUT
    for name, provider, is_fallback in providers:
        if is_fallback:
            on_status({"status": "running", "provider": name, "fallback": True, "reset": True})
        attempts = 1 if is_fallback else max(1, MAX_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            attempt_total += 1
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.", retryable=False)
            try:
                result = _try_provider(provider, name, request, attempt_total, on_delta, on_status,
                                       is_cancelled, deadline, is_fallback)
                result.elapsed = time.perf_counter() - started
                result.cost = round(sum(request.metadata.get("_attempt_costs", [])), 9)
                result.cost_estimated = bool(request.metadata.get("_estimated_cost", False))
                if result.provider == primary:
                    cache.put(key, result)
                return result
            except ProviderError as error:
                last_error = error
                if error.kind in {"budget", "cancelled"}:
                    raise
                if not error.retryable or attempt >= attempts or time.perf_counter() >= deadline:
                    break
                wait = min(BACKOFF_MAX, error.retry_after or BACKOFF_BASE * (2 ** (attempt - 1)))
                wait *= random.uniform(0.8, 1.2)
                if time.perf_counter() + wait >= deadline:
                    break
                on_status({"status": "running", "attempt": attempt_total, "provider": name,
                           "detail": "повтор після тимчасового збою", "reset": True})
                time.sleep(wait)
        if FALLBACK not in {"none", "", primary} and not is_fallback and last_error \
                and last_error.kind not in {"budget", "cancelled"}:
            fallback_name = FALLBACK
            try:
                fallback_provider = make_provider(fallback_name)
                providers.append((fallback_provider.name, fallback_provider, True))
            except ProviderError as error:
                last_error = error

    if last_error is None:
        last_error = ProviderError("unknown", "Не вдалося отримати відповідь від провайдера.")
    raise last_error
