"""Веб-рівень застосунку: сторінка розбору рахунків і JSON-ендпоінти.

Цей файл не знає ані як готується зображення, ані якою моделлю й за
якою інструкцією вилучено поля, ані якими правилами їх перевірено — усе
це в `app/images.py`, `app/llm.py`, `app/schema.py`, `app/rules.py` і
поєднується в `app/extraction.py`. Тут вирішується інше: що застосунок
приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr7:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import extraction, rules
from .images import ImageError, UPLOAD_MAX_BYTES
from .llm import LLMError

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Зразки віддаються як статичні файли, щоб сторінка могла їх показати й
# надіслати на розбір так само, як завантажений користувачем файл.
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def result_to_dict(result: extraction.Result) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [asdict(issue) for issue in result.issues],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
        "raw_response": result.raw_response,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/samples")
def api_samples() -> dict:
    """Перелік зразків за папками: `clean`, `degraded` і, якщо є, `own`."""
    groups = {}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[folder.name] = files
    return groups


@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)) -> dict:
    """Розібрати документ і повернути поля, проблеми й рішення.

    Збої підготовки зображення й запиту до моделі повертаються з
    контрольованим HTTP-статусом і зрозумілим повідомленням.
    """
    content = await image.read(UPLOAD_MAX_BYTES + 1)
    await image.close()
    if len(content) > UPLOAD_MAX_BYTES:
        limit_mb = UPLOAD_MAX_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Файл завеликий. Максимальний розмір — {limit_mb:g} МБ.",
        )
    try:
        result = await run_in_threadpool(extraction.process, content)
    except ImageError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except LLMError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except rules.ReferenceDataError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return result_to_dict(result)
