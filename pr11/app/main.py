"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.

Відносно ПР10 тут додано:

* проміжний шар `request_context`: кожному HTTP-запиту — серверний
  ідентифікатор у `telemetry.request_id_var`, заголовок `X-Request-ID`
  у відповіді й запис `http.request` у журналі;
* `fresh` у замовленні чернетки — обійти кеш (кнопка «Повторити»);
* `GET /api/usage` — стан бюджету й кешу для сторінки;
* `/api/health` віддає ще й версію prompt, налаштування кешу й ліміту.

Помилки вхідних даних і обмеження частоти мають контрольовані HTTP-відповіді
з `request_id`. Помилки генерації після прийняття роботи передаються
подією `error` і доступні у знімку чернетки.

Запуск із папки pr11:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import budget, cache, costs, drafting, gateway, jobs, limits, samples, telemetry

telemetry.setup()
log = logging.getLogger("app.http")

app = FastAPI(title="Помічник оператора — ПР11")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки.

    `request_id` — яке звернення зі зразків. `client` — ідентифікатор
    клієнта сторінки для обмеження частоти. `fresh` — оминути кеш:
    оператор хоче іншу чернетку, а не ту саму.
    """

    request_id: str
    client: str = Field(default="anonymous", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    fresh: bool = False


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Ідентифікатор запиту в контексті й у відповіді; запис про запит у
    журнал. Для потоку подій тривалість — до початку відповіді, не до
    кінця потоку."""
    request_id = telemetry.new_request_id()
    token = telemetry.request_id_var.set(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        log.exception("http.unhandled", extra={
            "method": request.method, "route": getattr(request.scope.get("route"), "path", "/unmatched"),
            "error_type": type(exc).__name__,
        })
        status = 500
        response = JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "Внутрішня помилка сервера.",
                     "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
        return response
    finally:
        log.info("http.request", extra={
            "method": request.method,
            "route": getattr(request.scope.get("route"), "path", "/unmatched"),
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        telemetry.request_id_var.reset(token)


@app.exception_handler(limits.RateLimited)
async def rate_limit_error(request: Request, exc: limits.RateLimited) -> JSONResponse:
    request_id = getattr(request.state, "request_id", telemetry.request_id_var.get())
    return JSONResponse(status_code=429, content={
        "error": "rate_limit", "message": str(exc), "retry_after": exc.retry_after,
        "request_id": request_id,
    }, headers={"Retry-After": str(max(1, int(exc.retry_after))), "X-Request-ID": request_id or ""})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", telemetry.request_id_var.get())
    return JSONResponse(status_code=422, content={
        "error": "validation", "message": "Некоректні дані запиту.", "request_id": request_id,
    }, headers={"X-Request-ID": request_id or ""})


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування — щоб сторінка показала, хто відповідає, яка
    версія prompt і які обмеження діють. Ключа тут немає."""
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "cache": cache.settings(),
        "budget": budget.settings(),
        "prompt_version": getattr(drafting, "PROMPT_VERSION", None),
        "log_file": telemetry.settings()["log_file"],
    }


@app.get("/api/usage")
def api_usage() -> dict:
    """Стан витрат і кешу — для панелі на сторінці."""
    return {
        "budget": budget.status(),
        "cache": cache.stats(),
        "pricing": costs.settings(),
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    """Замовити чернетку. Повертає її стан одразу, не чекаючи на
    генерацію; далі сторінка слухає події. Вхідні помилки й обмеження
    частоти повертаються з request ID; помилки генерації надходять подією
    `error` і доступні у знімку чернетки.
    """
    sample = samples.get(payload.request_id)
    if sample is None:
        request_id = telemetry.request_id_var.get()
        return JSONResponse(status_code=404, content={
            "error": "not_found", "message": "Звернення не знайдено.", "request_id": request_id,
        })
    limits.acquire(payload.client)
    return draft_to_dict(jobs.start(sample, payload.client, fresh=payload.fresh))


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    """Знімок однієї чернетки; невідома чернетка повертає 404."""
    draft = jobs.get(draft_id)
    if draft is None:
        return JSONResponse(status_code=404, content={
            "error": "not_found", "message": "Чернетку не знайдено.",
            "request_id": telemetry.request_id_var.get(),
        })
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
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

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
    """Скасувати генерацію; невідома чернетка повертає 404."""
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except KeyError:
        return JSONResponse(status_code=404, content={
            "error": "not_found", "message": "Чернетку не знайдено.",
            "request_id": telemetry.request_id_var.get(),
        })
