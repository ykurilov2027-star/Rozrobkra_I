"""Веб-рівень застосунку: сторінка пошуку і JSON-ендпоінти.

Цей файл не знає ані якою моделлю отримано вектори, ані як влаштований
індекс, ані як ранжує пошук за словами — усе це лишається в модулях
`app/embeddings.py`, `app/index.py`, `app/keyword.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr5),
а тут лише читається при старті.

Запуск із папки pr5:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import search

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class SearchRequest(BaseModel):
    """Те, що надсилає сторінка.

    `mode` — `semantic`, `keyword` або `both`. `filters` — умови на
    метадані фрагментів, наприклад `{"category": "інструкція"}`; порожній
    словник означає «без фільтрів». `threshold` — поріг схожості для
    семантичного пошуку; `None` — узяти значення з конфігурації.
    """

    query: str
    mode: str = "both"
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)
    threshold: float | None = Field(default=None, ge=-1, le=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    app.state.search_service = search.SearchService()
    yield


app = FastAPI(title="Пошук у базі знань — ПР5", lifespan=lifespan)
app.state.search_service = None


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку пошуку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан пошуку та налаштування для сторінки.

    Сторінка викликає це при відкритті, щоб показати підказку, якщо
    індексу ще немає.
    """
    service = app.state.search_service
    if service is None:
        return {"ready": False, "hint": "Пошук ще не ініціалізовано."}
    return service.status()


@app.post("/api/search")
def api_search(payload: SearchRequest) -> dict:
    """Виконати пошук і повернути влучення обох способів.

    Сторінка очікує обʼєкт із полями `semantic` і `keyword` (список
    влучень або `null`, якщо спосіб не запитували) та `elapsed` — час
    кожного способу в секундах.

    Некоректний запит і відсутній індекс повертаються як зрозумілі
    HTTP-помилки, які сторінка показує користувачеві.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Введіть непорожній пошуковий запит.")
    if payload.mode not in {"semantic", "keyword", "both"}:
        raise HTTPException(
            status_code=400,
            detail="Невідомий режим пошуку. Доступні: semantic, keyword, both.",
        )
    service = app.state.search_service
    if service is None:
        raise HTTPException(status_code=503, detail="Пошук ще не ініціалізовано.")
    try:
        return service.search(
            payload.query,
            payload.mode,
            payload.top_k,
            payload.filters or None,
            payload.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except search.SearchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
