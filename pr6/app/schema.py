"""Pydantic-контракт структурованої відповіді мовної моделі."""

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ModelAnswer(BaseModel):
  model_config = ConfigDict(extra="forbid")

  answer: str = Field(min_length=1, max_length=5000)
  found: bool
  sources: list[int] = Field(default_factory=list)


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі.

    Ця сама схема передається моделі як опис очікуваного результату
    (через `response_format` або текстом в інструкції) і використовується
    для перевірки того, що повернулося.
    """
    return ModelAnswer.model_json_schema()


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    Якщо текст не є JSON або не проходить схему — підняти помилку з
    поясненням, що саме не так.

    Схема перевіряє форму. Чи існують названі моделлю номери джерел серед
    тих, що їй показали, і чи не «знайшла» вона відповідь, не пославшись
    ні на що, — перевірка змісту, і її місце в `app/rag.py`.
    """
    try:
      value = json.loads(raw)
      return ModelAnswer.model_validate(value).model_dump()
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
      raise ValueError(f"Відповідь моделі не відповідає схемі: {exc}") from exc
