"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані які інструменти є в моделі, ані як перевіряються
аргументи, ані якою моделлю й за якою інструкцією отримано відповідь —
усе це в `app/tools.py`, `app/llm.py` і поєднується в
`app/assistant.py`. Тут вирішується інше: що застосунок приймає від
сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr8:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from shop import service

from . import assistant, llm, tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник клієнта — ПР8")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка.

    `customer_id` — клієнт, обраний на сторінці. Тут він заміняє вхід у
    кабінет: у справжньому застосунку веб-рівень узяв би його із сесії
    після автентифікації, а не з тіла запиту.
    """

    customer_id: str
    question: str


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат циклу на те, що піде на сторінку."""
    return {
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
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна «увійти» на сторінці."""
    return service.list_customers()


@app.get("/api/tools")
def api_tools() -> list[dict]:
    """Описи інструментів у тому вигляді, в якому їх бачить модель."""
    return tools.specs()


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану: створені повернення
    й скасування зникають. Зручно між прогонами перевірки."""
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Відповісти на питання клієнта й повернути журнал викликів.

    Обробка збоїв:

    * порожнє питання — 400;
    * невідомий клієнт — 404;
    * збої моделі (таймаут, ліміт, автентифікація) — з ПР3–ПР7;
    * несподівана помилка — 500.
    """
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Питання занадто довге.")

    # Перевірка, що клієнт існує
    try:
        service.get_customer(payload.customer_id)
    except service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except service.Unavailable as e:
        raise HTTPException(
            status_code=503,
            detail="Сервіс магазину тимчасово недоступний. Спробуйте пізніше.",
        ) from e

    try:
        result = assistant.answer(question, payload.customer_id)
        return answer_to_dict(result)
    except llm.LLMError as e:
        logger.error("LLMError: %s (status=%d)", e.message, e.status_code)
        detail = e.message
        if e.status_code >= 500:
            detail = "Сервіс моделі тимчасово не може відповісти. Спробуйте пізніше."
        raise HTTPException(status_code=e.status_code, detail=detail) from e
    except Exception as e:
        logger.exception("Непередбачена помилка: %s", e)
        raise HTTPException(
            status_code=500, detail="Внутрішня помилка сервера."
        ) from e