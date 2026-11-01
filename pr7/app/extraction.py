"""Конвеєр обробки документа: від байтів файлу до рішення, що з ним робити.

Єдине місце, яке знає всі кроки послідовно: підготувати зображення
(`images.prepare`), вилучити поля (`llm.extract`), перевірити їх
правилами (`rules.check`) і вирішити долю документа. Учасники одне про
одного не знають: модуль зображень — про модель, модель — про правила,
правила — про веб-рівень.

Рішення про документ — одне з трьох:

* `auto` — дані можна передати в реєстр платежів без людини;
* `review` — дані є, але їх має переглянути людина: причини перелічено;
* `reject` — це не рахунок на оплату або з документа нічого не взяти.

Рішення ухвалює код за результатами перевірок, а не модель. Для `auto`
потрібні всі обов'язкові поля та відсутність проблем; інші результати
рахунку передаються на перевірку, а нерахунки відхиляються.
"""

import time
from dataclasses import dataclass, field

from . import images, llm, rules
from .rules import Issue


@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `decision` — `auto`, `review` або `reject`; `reasons` — чому саме так,
    словами для людини. `document` — поля, які повернула модель і які
    пройшли схему (або `None`, якщо до цього не дійшло). `issues` —
    проблеми, знайдені правилами. `image` — розміри до й після підготовки.
    `elapsed` — час за етапами, наприклад `{"prepare": 0.05,
    "extraction": 6.8, "checks": 0.001}`. `usage` — токени, якщо модель
    викликалась.
    """

    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None
    raw_response: str | None = None


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими
    проблемами. Повертає рішення і його причини."""
    document_type_status = rules.classify_document_type(document.get("document_type"))
    if document_type_status == "other":
        return "reject", ["Документ не є рахунком на оплату."]
    if document_type_status == "unknown":
        reasons = list(dict.fromkeys(
            issue.message for issue in issues if issue.field == "document_type"
        ))
        return "review", reasons or ["Тип документа відсутній або не вдалося його визначити."]

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}
    extracted_values = [
        document.get("number"),
        document.get("date"),
        supplier.get("name"),
        supplier.get("code"),
        supplier.get("iban"),
        buyer.get("name"),
        buyer.get("code"),
        document.get("total_without_vat"),
        document.get("vat"),
        document.get("total"),
    ]
    for item in document.get("items") or []:
        extracted_values.extend(item.values())
    if not any(
        value is not None and (not isinstance(value, str) or value.strip())
        for value in extracted_values
    ):
        return "reject", ["З документа не вдалося вилучити дані рахунку."]

    if not issues:
        return "auto", ["Усі обов'язкові поля та програмні перевірки пройдено."]
    reasons = list(dict.fromkeys(issue.message for issue in issues))
    return "review", reasons


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    elapsed: dict[str, float] = {}
    started = time.perf_counter()

    stage_started = time.perf_counter()
    prepared = images.prepare(content)
    elapsed["prepare"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    extracted = llm.extract(prepared)
    elapsed["extraction"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    document = extracted["document"]
    issues = rules.check(document)
    elapsed["checks"] = time.perf_counter() - stage_started

    decision, reasons = decide(document, issues)
    elapsed["total"] = time.perf_counter() - started
    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image={"original": prepared.original, "sent": prepared.sent},
        model=extracted["model"],
        elapsed=elapsed,
        usage=extracted["usage"],
        raw_response=extracted["raw_response"],
    )
