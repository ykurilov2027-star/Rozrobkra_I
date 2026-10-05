"""Pydantic-контракт структурованої відповіді помічника."""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class SupportTopic(str, Enum):
    DELIVERY = "доставка"
    RETURNS = "повернення"
    PAYMENT = "оплата"
    CATALOG = "каталог"
    OTHER = "інше"


class SupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    reply: StrictStr = Field(min_length=1, description="Відповідь клієнту українською мовою")
    topic: SupportTopic
    is_based_on_rules: StrictBool
    needs_clarification: StrictBool
    escalate_to_operator: StrictBool
    order_number: StrictStr | None = Field(
        ...,
        pattern=r"^[0-9]{6}$",
        description="Шестизначний номер замовлення, якщо клієнт його назвав; інакше null",
    )


def output_schema() -> dict[str, Any]:
    """Повернути JSON Schema для structured output API."""
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "topic": {
                "type": "string",
                "enum": ["доставка", "повернення", "оплата", "каталог", "інше"],
            },
            "is_based_on_rules": {"type": "boolean"},
            "needs_clarification": {"type": "boolean"},
            "escalate_to_operator": {"type": "boolean"},
            "order_number": {
                "anyOf": [
                    {"type": "string", "pattern": "^[0-9]{6}$"},
                    {"type": "null"},
                ]
            },
        },
        "required": [
            "reply",
            "topic",
            "is_based_on_rules",
            "needs_clarification",
            "escalate_to_operator",
            "order_number",
        ],
        "additionalProperties": False,
    }


def validate(raw_data: str | dict[str, Any] | SupportResponse) -> dict[str, Any]:
    """Розібрати й перевірити сирий JSON або dict до передачі веб-рівню."""
    if isinstance(raw_data, SupportResponse):
        response = raw_data
    elif isinstance(raw_data, str):
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Відповідь моделі не є коректним JSON: {exc.msg}") from exc
        response = SupportResponse.model_validate(data)
    else:
        response = SupportResponse.model_validate(raw_data)
    return response.model_dump(mode="json")
