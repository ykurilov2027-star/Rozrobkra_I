"""FastAPI-вебрівень помічника служби підтримки."""

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import llm

logger = logging.getLogger(__name__)
app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class Turn(BaseModel):
    """Одна репліка розмови від клієнта або помічника."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    """Повідомлення клієнта та попередні репліки діалогу."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=12000)
    history: list[Turn] = Field(default_factory=list, max_length=100)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Повідомлення не може бути порожнім.")
        return value


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        return INDEX_PAGE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not read the chat page")
        raise HTTPException(
            status_code=500, detail="Не вдалося завантажити сторінку застосунку."
        ) from exc


@app.post("/api/chat")
def api_chat(payload: ChatRequest) -> dict:
    history = [turn.model_dump() for turn in payload.history]
    try:
        return llm.ask(history, payload.message)
    except llm.LLMError as exc:
        logger.error(
            "LLM request failed: code=%s status=%s model=%s cause=%s",
            exc.code,
            exc.status_code,
            llm.MODEL or "[unset]",
            type(exc.__cause__).__name__ if exc.__cause__ else "not available",
        )
        raise HTTPException(
            status_code=exc.status_code, detail=exc.public_message
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected failure while processing /api/chat")
        raise HTTPException(
            status_code=500,
            detail="Не вдалося обробити звернення. Спробуйте ще раз пізніше.",
        ) from exc
