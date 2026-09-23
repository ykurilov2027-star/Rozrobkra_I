"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Цей файл не має знати про `ultralytics`, ваги моделі й формат її «сирого»
виводу — усе це лишається в `app/detector.py`. Тут вирішується інше: що
застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr2:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from . import detector

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Завантажити ваги один раз під час запуску застосунку."""
    detector.load_model()
    yield


app = FastAPI(title="Детекція обʼєктів — ПР2", lifespan=lifespan)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...)):
    """Прийняти зображення і повернути структурований результат детекції."""
    content = await image.read()
    try:
        return detector.detect(content)
    except detector.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except detector.ModelInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except detector.DetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
