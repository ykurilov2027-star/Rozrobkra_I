"""Фонові задачі: чернетка генерується не в HTTP-запиті, а окремо.

`POST /api/drafts` повертається одразу з номером чернетки, а робота
триває у фоні: сторінка дізнається про стан через потік подій
(`GET /api/drafts/{id}/events`) або знімок (`GET /api/drafts/{id}`) і
може скасувати її (`POST /api/drafts/{id}/cancel`). Так чернетка не
прив'язана до одного HTTP-зʼєднання: сторінку можна перезавантажити,
а потім відкрити ту саму чернетку й побачити все, що встигло статися.

Модуль не знає, як влаштований шлюз і що саме генерується; він знає
лише `Draft`, події і те, що робота виконується не в потоці запиту.

Реєстр чернеток і події живе в памʼяті процесу. `emit` додає подію до
журналу чернетки й будить тих, хто чекає; `subscribe` віддає події
від указаного номера і далі в міру надходження, доки чернетка не
дійде до кінцевого статусу. Веб-рівень перетворює це на
server-sent events; реалізація фонової роботи до цього не торкається.

`start` створює потік на чернетку. Стан і події зберігаються в памʼяті
процесу; скасування передається через `threading.Event`.

Події, які розуміє сторінка (інші вона показує як є):

    status     {"status": "running", "attempt": 1, "provider": "openai",
                "fallback": false, "detail": "…"}
    delta      {"text": "…"}                      приріст тексту
    done       {"text": "…", "provider": "…", "model": "…",
                "usage": {...}, "elapsed": {...}, "attempts": 2,
                "fallback": false, "warnings": ["…"]}
    error      {"type": "timeout", "message": "…", "retryable": true,
                "request_id": "…"}
    cancelled  {}

Поля — відправна точка; змінюючи їх, змініть і сторінку.
"""

import datetime
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterator

from . import drafting, gateway
from .providers.base import ProviderError

# Статуси, після яких подій більше не буде.
TERMINAL = {"done", "failed", "cancelled"}


@dataclass
class Draft:
    """Стан однієї чернетки — те, що зберігається в реєстрі і йде на
    сторінку як знімок.

    `status` — `queued`, `running`, `done`, `failed`, `cancelled`.
    `text` — накопичений текст. `attempts`, `provider`, `model`,
    `fallback` — звідки взявся результат. `error` — остаточна помилка,
    якщо є: `kind`, `message`, `retryable`. `elapsed` — секунди, наприклад
    `{"first_token": 0.8, "total": 4.1}`. `warnings` — зауваження до
    чернетки з `drafting.check`. `events` — журнал подій для потоку;
    у знімок на сторінку не входить.
    """

    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    max_attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_lock = threading.Lock()
_changed = threading.Condition(_lock)
_cancellations: dict[str, threading.Event] = {}


class DraftNotFound(Exception):
    pass


class DraftConflict(Exception):
    pass


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    """Створити чернетку зі статусом `queued` і зареєструвати її.
    Роботу не запускає — це справа `start`."""
    draft = Draft(draft_id=uuid.uuid4().hex[:8], request_id=request_id, client=client,
                  created_at=_now(), updated_at=_now())
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def snapshot(draft: Draft) -> dict:
    with _lock:
        data = asdict(draft)
        data["events"] = len(draft.events)
        return data


def list_drafts(limit: int = 50) -> list[Draft]:
    """Чернетки, нові першими."""
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    """Додати подію до журналу чернетки й розбудити підписників.

    Статус і текст чернетки ця функція не змінює — це робить фонова
    робота разом із подією, під тим самим замком, якщо їй потрібна
    узгодженість між знімком і потоком.
    """
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
        draft.updated_at = _now()
        _changed.notify_all()


def _update_and_emit(draft: Draft, event: str, data: dict,
                     updates: dict | None = None) -> None:
    with _changed:
        if updates:
            for name, value in updates.items():
                setattr(draft, name, value)
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data})
        draft.updated_at = _now()
        _changed.notify_all()


def _finish(draft: Draft, cancelled: threading.Event, event: str, data: dict,
            updates: dict) -> str:
    with _changed:
        if cancelled.is_set():
            event, data, updates = "cancelled", {}, {"status": "cancelled"}
        for name, value in updates.items():
            setattr(draft, name, value)
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data})
        draft.updated_at = _now()
        _changed.notify_all()
        return event


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Віддавати події чернетки, починаючи з номера `since + 1`.

    Спершу — ті, що вже є, потім — нові в міру надходження. Завершується,
    коли чернетка в кінцевому статусі й усі події віддано, або коли
    чернетки немає. Якщо подій немає довше за `heartbeat` секунд, віддає
    `None`: веб-рівень надсилає порожній коментар, щоб зʼєднання не
    закрив проксі і щоб помітити, що клієнт відключився.
    """
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def start(sample: dict, client: str) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у
    фоні. Повертає чернетку одразу — найімовірніше, ще `queued`."""
    draft = new_draft(sample["id"], client)
    cancelled = threading.Event()
    with _lock:
        _cancellations[draft.draft_id] = cancelled
    emit(draft, "status", {"status": "queued", "detail": "Чернетку додано до черги"})

    def run() -> None:
        try:
            with _lock:
                already_cancelled = cancelled.is_set()
                if not already_cancelled:
                    draft.status = "running"
            if already_cancelled:
                _finish(draft, cancelled, "cancelled", {}, {"status": "cancelled"})
                return
            emit(draft, "status", {"status": "running", "detail": "Генерація почалася"})
            request = drafting.build_request(sample)

            def on_status(status: dict) -> None:
                updates = {"status": status.get("status", "running")}
                if "attempt" in status:
                    updates["attempts"] = status["attempt"]
                if "max_attempts" in status:
                    updates["max_attempts"] = status["max_attempts"]
                if "provider" in status:
                    updates["provider"] = status["provider"]
                if "model" in status:
                    updates["model"] = status["model"]
                if "fallback" in status:
                    updates["fallback"] = status["fallback"]
                if status.get("reset"):
                    updates["text"] = ""
                _update_and_emit(draft, "status", status, updates)

            def on_delta(text: str) -> None:
                if cancelled.is_set():
                    return
                with _lock:
                    if cancelled.is_set() or draft.status in TERMINAL:
                        return
                    draft.text += text
                emit(draft, "delta", {"text": text})

            result = gateway.generate(request, on_delta, on_status, cancelled.is_set)
            warnings = drafting.check(result.text, sample)
            elapsed = {}
            if result.elapsed is not None:
                elapsed["total"] = result.elapsed
            if result.first_token is not None:
                elapsed["first_token"] = result.first_token
            data = {
                "text": result.text, "provider": result.provider, "model": result.model,
                "usage": result.usage, "elapsed": elapsed, "attempts": result.attempts,
                "max_attempts": result.max_attempts,
                "fallback": result.fallback, "warnings": warnings,
            }
            _finish(draft, cancelled, "done", data, {
                "status": "done", "text": result.text, "provider": result.provider,
                "model": result.model, "fallback": result.fallback,
                "usage": result.usage, "elapsed": elapsed, "warnings": warnings,
                "attempts": result.attempts, "max_attempts": result.max_attempts,
            })
        except ProviderError as exc:
            if exc.kind == "cancelled":
                _finish(draft, cancelled, "cancelled", {}, {"status": "cancelled"})
                return
            error = {
                "type": exc.kind, "kind": exc.kind, "message": exc.message,
                "retryable": exc.retryable, "request_id": draft.draft_id,
            }
            event = _finish(draft, cancelled, "error", error, {"status": "failed", "error": error})
            if event == "cancelled":
                return
        except Exception:
            error = {
                "type": "unknown", "kind": "unknown",
                "message": "Сталася непередбачена помилка під час створення чернетки.",
                "retryable": False, "request_id": draft.draft_id,
            }
            _finish(draft, cancelled, "error", error, {"status": "failed", "error": error})
        finally:
            with _lock:
                _cancellations.pop(draft.draft_id, None)

    threading.Thread(target=run, name=f"draft-{draft.draft_id}", daemon=True).start()
    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися. Повертає чернетку; якщо її
    немає або вона вже завершена — вирішуєте ви."""
    with _lock:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise DraftNotFound(draft_id)
        if draft.status in TERMINAL:
            raise DraftConflict(draft_id)
        cancelled = _cancellations.get(draft_id)
        if cancelled is None:
            raise DraftConflict(draft_id)
        cancelled.set()
    emit(draft, "status", {"status": draft.status, "detail": "Скасування запитано"})
    return draft
