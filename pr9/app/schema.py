"""Structured output contract for customer request parsing."""

import json

from jsonschema import Draft202012Validator


_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["return", "cancel", "status", "other", "multiple", "unclear"],
        },
        "order_id": {"type": ["string", "null"], "pattern": r"^\d{3,8}$"},
        "item": {"type": ["string", "null"], "maxLength": 120},
        "reason": {
            "type": ["string", "null"],
            "enum": ["not_suitable", "defect", "wrong_item", "damaged", None],
        },
        "quantity": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
    },
    "required": ["kind", "order_id", "item", "reason", "quantity"],
    "additionalProperties": False,
}


def request_schema() -> dict:
    """Return the JSON Schema sent to the model and used for validation."""
    return _REQUEST_SCHEMA


def validate(raw: str) -> dict:
    """Validate model JSON, then return normalized fields."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Відповідь моделі не є коректним JSON: {exc}") from exc

    errors = sorted(
        Draft202012Validator(_REQUEST_SCHEMA).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "результат"
        raise ValueError(f"Некоректне поле {path}: {error.message}")
    return value
