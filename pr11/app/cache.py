"""Точний кеш відповідей із TTL та персистентністю."""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")
_lock = threading.RLock()
_entries: dict[str, dict] = {}
_hits = 0
_misses = 0
_bypassed = 0


def _save() -> None:
    if not CACHE_FILE:
        return
    path = Path(CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_entries, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


if ENABLED and CACHE_FILE and Path(CACHE_FILE).exists():
    loaded = json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"неправильний формат кешу: {CACHE_FILE}")
    _entries = loaded


def settings() -> dict:
    return {"enabled": ENABLED, "ttl_seconds": TTL_SECONDS, "max_entries": MAX_ENTRIES,
            "file": CACHE_FILE or None}


def make_key(request: Request, prompt_version: str) -> str:
    identity = {
        "provider": request.metadata.get("provider", os.getenv("LLM_PROVIDER", "openai")),
        "model": request.model or os.getenv("LLM_MODEL"),
        "prompt_version": prompt_version,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get(key: str) -> Result | None:
    global _hits, _misses
    with _lock:
        if not ENABLED:
            return None
        record = _entries.get(key)
        if record and time.time() - record["stored_at"] <= TTL_SECONDS:
            _hits += 1
            result = Result(**record["result"])
            result.cached = True
            return result
        if record:
            del _entries[key]
            _save()
        _misses += 1
        return None


def put(key: str, result: Result) -> None:
    if not ENABLED:
        return
    with _lock:
        value = asdict(result)
        value["cached"] = False
        _entries[key] = {"stored_at": time.time(), "result": value}
        if len(_entries) > MAX_ENTRIES:
            oldest = sorted(_entries, key=lambda item: _entries[item]["stored_at"])
            for expired in oldest[:len(_entries) - MAX_ENTRIES]:
                del _entries[expired]
        _save()


def bypassed() -> None:
    global _bypassed
    with _lock:
        _bypassed += 1


def stats() -> dict:
    with _lock:
        now = time.time()
        expired = [key for key, entry in _entries.items()
                   if now - entry["stored_at"] > TTL_SECONDS]
        for key in expired:
            del _entries[key]
        if expired:
            _save()
        return {"enabled": ENABLED, "hits": _hits, "misses": _misses,
                "bypassed": _bypassed, "entries": len(_entries)}
