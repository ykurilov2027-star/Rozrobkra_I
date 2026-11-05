"""Цикл помічника: від питання клієнта до відповіді, спертої на дані
магазину.

Рішення:

* цикл «модель → виклики → результати → модель», обмежений
  `TOOL_MAX_ROUNDS`;
* якщо ліміт вичерпано, а текстової відповіді немає — робимо **останній
  виклик без інструментів** (`tool_choice="none"`), щоб модель
  відповіла тим, що вже має; якщо й це не вдалося — контрольоване
  повідомлення;
* усі виклики з однієї відповіді виконуємо **послідовно** — просто й
  детерміновано; для `create_return` це безпечніше за паралелізм;
* повідомлення моделі з викликами повертаємо в історію **як є** —
  Gemini додає підписи, без яких наступний запит не прийметься;
* час моделі й час інструментів рахуємо окремо; токени сумуємо за всі
  звертання.
"""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


_FALLBACK_LIMIT = (
    "Вибачте, не вдалося зібрати відповідь за обмежену кількість кроків. "
    "Спробуйте, будь ласка, переформулювати питання або зверніться до "
    "оператора підтримки."
)


def answer(question: str, customer_id: str) -> Answer:
    """Відповісти клієнтові, за потреби викликаючи інструменти."""
    question = (question or "").strip()
    if not question:
        return Answer(
            text="Будь ласка, напишіть питання.",
            stopped="empty",
            elapsed={"model": 0.0, "tools": 0.0},
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    ctx = tools.Context(customer_id=customer_id)
    specs = tools.specs()
    messages = llm.build_messages(question)

    traces: list[ToolTrace] = []
    total_prompt = 0
    total_completion = 0
    model_time = 0.0
    tools_time = 0.0
    rounds = 0
    model_name: str | None = None

    for round_num in range(1, MAX_ROUNDS + 1):
        rounds = round_num
        try:
            resp = llm.chat(messages, tools=specs, tool_choice="auto")
        except llm.LLMError:
            # Збій моделі — не результат інструмента. Пробуємо один
            # раз без інструментів, щоб дати хоч якусь відповідь.
            try:
                resp = llm.chat(messages, tools=[], tool_choice="none")
            except llm.LLMError:
                raise
        model_time += resp["elapsed"]
        model_name = resp["model"]
        total_prompt += resp["usage"]["prompt_tokens"]
        total_completion += resp["usage"]["completion_tokens"]

        msg = resp["message"]
        finish = resp["finish_reason"]
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Текстова відповідь — кінець циклу
        if not tool_calls:
            text = (msg.content or "").strip()
            if not text:
                text = _FALLBACK_LIMIT
            return Answer(
                text=text,
                calls=traces,
                rounds=rounds,
                stopped="answer",
                model=model_name,
                elapsed={"model": round(model_time, 3),
                         "tools": round(tools_time, 3)},
                usage={"prompt_tokens": total_prompt,
                       "completion_tokens": total_completion},
            )

        # Додаємо повідомлення моделі як є
        messages.append(_message_to_dict(msg))

        # Виконуємо кожен виклик
        for call in tool_calls:
            t0 = time.perf_counter()
            result = tools.call(call.function.name,
                                call.function.arguments, ctx)
            elapsed = time.perf_counter() - t0
            tools_time += elapsed

            traces.append(ToolTrace(
                round=round_num,
                name=call.function.name,
                arguments=call.function.arguments or "",
                status=result.status,
                reason=result.reason,
                result=result.content,
                elapsed=round(elapsed, 4),
            ))

            # Результат — теж повідомлення, з tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result.content, ensure_ascii=False)
                           if result.content is not None else "",
            })

    # Ліміт вичерпано — останній шанс на відповідь
    try:
        final = llm.chat(messages, tools=[], tool_choice="none")
        model_time += final["elapsed"]
        total_prompt += final["usage"]["prompt_tokens"]
        total_completion += final["usage"]["completion_tokens"]
        text = (final["message"].content or "").strip() or _FALLBACK_LIMIT
    except llm.LLMError:
        text = _FALLBACK_LIMIT

    return Answer(
        text=text,
        calls=traces,
        rounds=rounds,
        stopped="limit",
        model=model_name,
        elapsed={"model": round(model_time, 3),
                 "tools": round(tools_time, 3)},
        usage={"prompt_tokens": total_prompt,
               "completion_tokens": total_completion},
    )


def _message_to_dict(msg) -> dict:
    """Перетворити повідомлення моделі в dict для історії.

    Gemini додає до `tool_calls` службові поля (`extras`,
    `thought_signature`), без яких наступний запит може не прийнятися.
    Тому серіалізуємо об'єкт як є.
    """
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    # Fallback для сумісності зі старішими версіями SDK
    out = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name,
                             "arguments": c.function.arguments},
            }
            for c in msg.tool_calls
        ]
    return out