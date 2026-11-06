"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.

Цей файл не знає ані який провайдер відповідає, ані скільки було спроб,
ані в якому потоці йде генерація — усе це в `app/gateway.py`,
`app/providers/` і `app/jobs.py`. Тут вирішується інше: що застосунок
приймає від сторінки, що віддає їй і з яким HTTP-статусом, і як події
фонової роботи стають потоком server-sent events.

Запуск із папки pr10:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import json
import math
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import gateway, jobs, limits, samples

app = FastAPI(title="Помічник оператора — ПР10")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def api_error(request: Request, status: int, kind: str, message: str,
              retryable: bool = False, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "type": kind, "message": message, "retryable": retryable,
            "request_id": request.state.request_id,
        },
        headers=headers,
    )


@app.exception_handler(limits.RateLimited)
async def rate_limit_error(request: Request, exc: limits.RateLimited):
    return api_error(request, 429, "rate_limit", "Забагато запитів. Спробуйте пізніше.",
                     True, {"Retry-After": str(math.ceil(exc.retry_after))})


@app.exception_handler(jobs.DraftNotFound)
async def draft_not_found(request: Request, exc: jobs.DraftNotFound):
    return api_error(request, 404, "not_found", "Чернетку не знайдено.")


@app.exception_handler(jobs.DraftConflict)
async def draft_conflict(request: Request, exc: jobs.DraftConflict):
    return api_error(request, 409, "conflict", "Чернетка вже завершена або скасована.")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return api_error(request, 422, "bad_request", "Запит має некоректний формат.")


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        return api_error(request, 404, "not_found", "Ресурс не знайдено.")
    if exc.status_code == 405:
        return api_error(request, 405, "bad_request", "Метод запиту не підтримується.")
    return api_error(request, exc.status_code, "bad_request", "Запит не вдалося обробити.")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    return api_error(request, 500, "unknown", "Сталася внутрішня помилка.")


class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки.

    `request_id` — яке звернення зі зразків. `client` — ідентифікатор
    клієнта сторінки для обмеження частоти; у справжньому застосунку
    його взяв би із сесії веб-рівень, а не сторінка.
    """

    request_id: str
    client: str = "anonymous"


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    return jobs.snapshot(draft)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування — щоб сторінка показала, хто відповідає і які
    обмеження діють. Ключа тут немає."""
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    """Замовити чернетку у фоні та одразу повернути її стан."""
    sample = samples.get(payload.request_id)
    if sample is None:
        raise jobs.DraftNotFound(payload.request_id)
    limits.acquire(payload.client)
    return draft_to_dict(jobs.start(sample, payload.client))


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    """Знімок однієї чернетки."""
    draft = jobs.get(draft_id)
    if draft is None:
        raise jobs.DraftNotFound(draft_id)
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    """Потік подій чернетки у форматі server-sent events.

    Браузер при обриві зʼєднання перепідключається сам і надсилає
    заголовок `Last-Event-ID` — звідси продовжуємо, не повторюючи
    віддане. Порожній коментар раз на кілька секунд тримає зʼєднання
    живим; якщо клієнт відключився, спроба записати в нього завершує
    генератор.
    """
    try:
        since = max(0, int(request.headers.get("last-event-id") or 0))
    except ValueError:
        since = 0
    if jobs.get(draft_id) is None:
        raise jobs.DraftNotFound(draft_id)

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    """Запросити скасування активної чернетки."""
    return draft_to_dict(jobs.cancel(draft_id))
