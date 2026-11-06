"""Розрахунок вартості викликів за використанням токенів."""

import json
import os

from dotenv import load_dotenv

load_dotenv()

PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def _token_estimate(text: str) -> int:
    return max(1, (len(text) + 2) // 3) if text else 0


def estimate(usage: dict | None, *, provider: str | None = None, model: str | None = None,
             text: str | None = None, prompt: str | None = None) -> tuple[float | None, bool]:
    """Return USD cost and whether token counts were estimated."""
    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    estimated = not usage or not isinstance(usage.get("prompt_tokens"), (int, float)) \
        or not isinstance(usage.get("completion_tokens"), (int, float))
    if estimated:
        prompt_tokens = _token_estimate(prompt or "")
        completion_tokens = _token_estimate(text or "")
        cached_tokens = 0
    else:
        prompt_tokens = max(0, int(usage["prompt_tokens"]))
        completion_tokens = max(0, int(usage["completion_tokens"]))
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = max(0, int(details.get("cached_tokens", 0) or 0))
        cached_tokens = min(cached_tokens, prompt_tokens)

    uncached_tokens = prompt_tokens - cached_tokens
    value = (uncached_tokens * PRICE_INPUT + cached_tokens * PRICE_CACHED_INPUT
             + completion_tokens * PRICE_OUTPUT) / 1_000_000
    return round(value, 9), estimated


def estimate_usage(usage: dict | None, *, prompt: str, text: str) -> tuple[dict, bool]:
    """Normalize usage data, estimating prompt and completion tokens if absent."""
    if usage and isinstance(usage.get("prompt_tokens"), (int, float)) \
            and isinstance(usage.get("completion_tokens"), (int, float)):
        return usage, False
    return {
        "prompt_tokens": _token_estimate(prompt),
        "completion_tokens": _token_estimate(text),
        "total_tokens": _token_estimate(prompt) + _token_estimate(text),
        "estimated": True,
    }, True
