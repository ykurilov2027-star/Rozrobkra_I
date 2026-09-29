"""Веб-рівень застосунку: сторінка зі зверненням і API-ендпоїнт."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import llm

load_dotenv()

app = FastAPI(title="Помічник служби підтримки — ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class Question(BaseModel):
    """Звернення користувача."""

    question: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь помічника у форматі JSON."""
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Ви не надіслали звернення. Напишіть питання.")

    result = llm.ask_support_assistant(question)
    status_code = 200
    error_code = result.get("error_code")

    if error_code == "empty_input":
        status_code = 400
    elif error_code == "rate_limit":
        status_code = 429
    elif error_code in {"auth_error", "config_error"}:
        status_code = 401
    elif error_code in {"service_unavailable", "llm_error"}:
        status_code = 503
    elif error_code == "invalid_input":
        status_code = 400

    if status_code != 200:
        return JSONResponse(status_code=status_code, content=result)

    return result
