"""Pydantic-моделі відповіді мультимодального вилучення."""

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator


DecimalValue = Annotated[Decimal, Field(max_digits=30, decimal_places=6)]
_DECIMAL_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")


def _parse_decimal(value: object, field_name: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{field_name}: використовуйте десятковий рядок або число без float.")
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str):
        text = re.sub(r"[\s\u00a0\u202f]", "", value)
        text = text.replace("₴", "").replace("грн.", "").replace("грн", "")
        if "," in text and "." in text:
            decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
            grouping_separator = "." if decimal_separator == "," else ","
            text = text.replace(grouping_separator, "")
            if decimal_separator == ",":
                text = text.replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        if len(text) > 64 or not _DECIMAL_PATTERN.fullmatch(text):
            raise ValueError(f"{field_name}: очікується коректне десяткове число.")
        try:
            value = Decimal(text)
        except (InvalidOperation, ValueError, TypeError) as error:
            raise ValueError(f"{field_name}: некоректне число.") from error
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{field_name}: число має бути скінченним.")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Supplier(StrictModel):
    name: str | None = None
    code: str | None = None
    iban: str | None = None


class Buyer(StrictModel):
    name: str | None = None
    code: str | None = None


class Item(StrictModel):
    name: str | None = None
    unit: str | None = None
    quantity: DecimalValue | None = None
    price: DecimalValue | None = None
    amount: DecimalValue | None = None

    @field_validator("quantity", "price", "amount", mode="before")
    @classmethod
    def parse_decimal_fields(cls, value: object, info: ValidationInfo) -> Decimal | None:
        return _parse_decimal(value, info.field_name)


class Extraction(StrictModel):
    document_type: str | None = None
    number: str | None = None
    date: str | None = None
    valid_until: str | None = None
    supplier: Supplier = Field(default_factory=Supplier)
    buyer: Buyer = Field(default_factory=Buyer)
    items: list[Item] = Field(default_factory=list)
    total_without_vat: DecimalValue | None = None
    vat: DecimalValue | None = None
    total: DecimalValue | None = None

    @field_validator("total_without_vat", "vat", "total", mode="before")
    @classmethod
    def parse_decimal_fields(cls, value: object, info: ValidationInfo) -> Decimal | None:
        return _parse_decimal(value, info.field_name)


def output_schema() -> dict:
    """Повернути JSON Schema, сформовану з Pydantic-моделі."""
    return Extraction.model_json_schema()


def validate(raw: str) -> dict:
    """Перевірити JSON і повернути нормалізований словник."""
    try:
        parsed = json.loads(raw, parse_float=Decimal)
    except (json.JSONDecodeError, TypeError, ValueError, InvalidOperation) as error:
        raise ValueError(f"Відповідь моделі не є коректним JSON: {error}") from error
    try:
        document = Extraction.model_validate(parsed)
    except ValidationError as error:
        details = []
        for issue in error.errors()[:5]:
            path = ".".join(map(str, issue["loc"])) or "корінь"
            details.append(f"{path}: {issue['msg']}")
        raise ValueError("Відповідь моделі не відповідає схемі: " + "; ".join(details)) from error
    return document.model_dump(mode="json")
