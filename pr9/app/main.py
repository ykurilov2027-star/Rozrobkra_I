"""Веб-рівень застосунку: сторінка звернень, черга оператора і
JSON-ендпоінти.

Цей файл не знає ані які кроки є у workflow, ані як зберігається
стан, ані якою моделлю розібрано звернення — усе це в
`app/workflow.py`, `app/store.py` і `app/llm.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.

Для порівняння той самий текст звернення можна віддати агентові з ПР8
(`app/assistant.py`) — сторінка має перемикач способу обробки.

Запуск із папки pr9:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from shop import service

from . import assistant, llm, store, workflow

app = FastAPI(title="Звернення клієнтів — ПР9")
logger = logging.getLogger(__name__)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class RequestIn(BaseModel):
    """Нове звернення зі сторінки.

    `customer_id` — клієнт, обраний на сторінці. Тут він заміняє вхід у
    кабінет: у справжньому застосунку веб-рівень узяв би його із сесії.
    `mode` — хто обробляє: `workflow` або `agent` (помічник із ПР8).
    """

    customer_id: str
    text: str = Field(max_length=2001)
    mode: Literal["workflow", "agent"] = "workflow"


class DecisionIn(BaseModel):
    """Рішення оператора щодо дії, яка чекає підтвердження."""

    approve: bool
    comment: str = Field(default="", max_length=500)


def run_to_dict(run: workflow.Run) -> dict:
    """Перетворити стан звернення на те, що піде на сторінку."""
    data = asdict(run)
    data["mode"] = "workflow"
    data["limits"] = workflow.limits()
    return data


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат агента з ПР8 на те, що піде на сторінку.

    Якщо в ПР8 ви змінили склад `Answer` чи `ToolTrace`, перенесіть
    сюди й свою `answer_to_dict` з `pr8/app/main.py`.
    """
    return {
        "mode": "agent",
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.on_event("startup")
def recover_interrupted_runs() -> None:
    store.recover_interrupted()


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна «увійти» на сторінці."""
    return service.list_customers()


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану: створені повернення
    й скасування зникають. Збережені звернення не зачіпаються — їх
    видаляйте з `RUNS_DIR` самі, якщо потрібно."""
    service.reset()
    return {"ok": True}


@app.post("/api/requests")
def api_create(payload: RequestIn) -> dict:
    """Прийняти звернення й обробити його обраним способом.

    Workflow повертає стан звернення (див. `run_to_dict`), агент —
    відповідь і журнал викликів, як у ПР8 (див. `answer_to_dict`).

    Workflow-помилки зберігаються як контрольовані стани звернення;
    некоректний запит або невідомий клієнт отримують відповідний HTTP-код.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Звернення не може бути порожнім.")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Звернення занадто довге.")
    try:
        service.get_customer(payload.customer_id)
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail="Клієнта не знайдено.") from exc
    except service.Unavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Сервіс магазину тимчасово недоступний. Спробуйте пізніше.",
        ) from exc

    if payload.mode == "agent":
        try:
            return answer_to_dict(assistant.answer(text, payload.customer_id))
        except llm.LLMError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return run_to_dict(workflow.start(text, payload.customer_id))


@app.get("/api/runs")
def api_runs(status: str | None = None) -> list[dict]:
    """Звернення, оброблені workflow, — усі або з певним статусом.

    Черга оператора на сторінці — це `?status=awaiting_approval`. Якщо
    ваш статус очікування називається інакше, змініть його в сторінці.
    """
    return [run_to_dict(run) for run in store.list_runs(status)]


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    """Return one saved run, or 404 when it does not exist."""
    run = store.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Звернення не знайдено.")
    return run_to_dict(run)


@app.post("/api/runs/{run_id}/decision")
def api_decision(run_id: str, payload: DecisionIn) -> dict:
    """Рішення оператора: підтвердити чи відхилити дію.

    Повторне рішення не запускає дію вдруге: повертається поточний стан.
    """
    try:
        return run_to_dict(workflow.resume(run_id, payload.approve, payload.comment))
    except workflow.RunNotFound as exc:
        raise HTTPException(status_code=404, detail="Звернення не знайдено.") from exc
    except Exception as exc:
        logger.exception("Помилка рішення оператора для %s", run_id)
        raise HTTPException(status_code=500, detail="Не вдалося зберегти рішення оператора.") from exc
