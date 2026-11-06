"""Персистентний денний ліміт витрат."""

import datetime
import json
import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")
_lock = threading.RLock()
_log = logging.getLogger("app.budget")
_day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
_spent = 0.0
_warned = False


class BudgetExceeded(Exception):
    """Денний ліміт вичерпано."""

    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(f"денний ліміт витрат {limit:.2f} $ вичерпано; відновиться {resets_at}")
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _reset_at() -> str:
    tomorrow = datetime.datetime.now(datetime.timezone.utc).date() + datetime.timedelta(days=1)
    return datetime.datetime.combine(tomorrow, datetime.time(), datetime.timezone.utc).isoformat()


def _save() -> None:
    if not BUDGET_FILE:
        return
    path = Path(BUDGET_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps({"day": _day, "spent": _spent}), encoding="utf-8")
    os.replace(temp, path)


if BUDGET_FILE and Path(BUDGET_FILE).exists():
    state = json.loads(Path(BUDGET_FILE).read_text(encoding="utf-8"))
    if state.get("day") == _day:
        _spent = max(0.0, float(state.get("spent", 0)))


def _roll_day() -> None:
    global _day, _spent, _warned
    today = _today()
    if today != _day:
        _day, _spent, _warned = today, 0.0, False
        _save()


def settings() -> dict:
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def check(expected_cost: float | None = None) -> None:
    with _lock:
        _roll_day()
        if _spent + max(0.0, expected_cost or 0.0) >= DAILY_USD:
            raise BudgetExceeded(_spent, DAILY_USD, _reset_at())


def record(cost: float | None) -> None:
    global _spent, _warned
    if cost is None:
        return
    with _lock:
        _roll_day()
        _spent += max(0.0, cost)
        _save()
        if not _warned and DAILY_USD > 0 and _spent >= DAILY_USD * WARN_AT:
            _warned = True
            _log.warning("budget.warning", extra={
                "day": _day, "spent": round(_spent, 6), "limit": DAILY_USD,
                "remaining": max(0.0, DAILY_USD - _spent),
            })


def status() -> dict:
    with _lock:
        _roll_day()
        remaining = max(0.0, DAILY_USD - _spent)
        return {
            "day": _day,
            "spent": round(_spent, 9),
            "limit": DAILY_USD,
            "remaining": round(remaining, 9),
            "warning": DAILY_USD > 0 and _spent >= DAILY_USD * WARN_AT,
            "exceeded": _spent >= DAILY_USD,
            "resets_at": _reset_at(),
        }
