"""Побудова, зберігання та cosine-пошук по векторному індексу."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk
from .embeddings import MODEL_NAME

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Скільки результатів повертати за замовчуванням і чи відсікати слабкі.
# 0.81 відсікає перевірений позаколекційний top-1 (0.802); його варто
# повторно калібрувати, якщо зміняться модель або колекція.
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "0.81").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


class IndexLoadError(RuntimeError):
    """Індекс неможливо прочитати або використати з поточною моделлю."""


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Тут перевіряється, що векторів стільки ж, скільки фрагментів, і
    вирішується, що зробити з векторами перед пошуком (наприклад,
    нормалізувати, якщо схожість — косинусна).
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Очікується матриця векторів N × d, отримано форму {matrix.shape}.")
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Кількість векторів ({matrix.shape[0]}) не збігається "
            f"з кількістю фрагментів ({len(chunks)})."
        )
    if not model_name.strip():
        raise ValueError("Назва моделі ембедінгів не може бути порожньою.")
    if not np.isfinite(matrix).all():
        raise ValueError("Матриця векторів містить NaN або нескінченні значення.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Матриця містить нульовий вектор, який неможливо нормалізувати.")
    if len(matrix):
        matrix = matrix / norms
    return SearchIndex(chunks=list(chunks), vectors=matrix, model_name=model_name)


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти модель, вектори та фрагменти з метаданими у JSON."""
    target = path if path.suffix.lower() == ".json" else path / "index.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_name": index.model_name,
        "vectors": index.vectors.tolist(),
        "chunks": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in index.chunks
        ],
    }
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, separators=(",", ":"))
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Завантажити індекс і перевірити його відповідність налаштованій моделі."""
    target = path if path.suffix.lower() == ".json" else path / "index.json"
    if not target.is_file():
        raise IndexLoadError(
            f"Індекс не знайдено за шляхом {target}. Запустіть `python ingest.py`."
        )
    try:
        with target.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format_version") != 1:
            raise ValueError("непідтримувана версія формату")
        stored_model = payload["model_name"]
        if stored_model != MODEL_NAME:
            raise IndexLoadError(
                f"Індекс побудовано моделлю {stored_model!r}, а зараз налаштовано "
                f"{MODEL_NAME!r}. Запустіть `python ingest.py`, щоб перебудувати індекс."
            )
        chunks = [
            Chunk(
                id=item["id"],
                text=item["text"],
                source=item["source"],
                metadata=item["metadata"],
            )
            for item in payload["chunks"]
        ]
        loaded = build(chunks, np.asarray(payload["vectors"], dtype=np.float32), stored_model)
    except IndexLoadError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise IndexLoadError(
            f"Індекс {target} пошкоджений або має некоректний формат "
            f"({type(exc).__name__}: {exc}). Запустіть `python ingest.py`."
        ) from exc
    return loaded


def validate_filters(chunks: list[Chunk], filters: dict | None) -> dict[str, str]:
    """Перевірити фільтри та нормалізувати їх перед пошуком."""
    if not filters:
        return {}
    allowed_fields = {"audience", "status", "category", "product"}
    normalized: dict[str, str] = {}
    for field_name, raw_value in filters.items():
        if field_name not in allowed_fields:
            raise ValueError(
                f"Невідомий фільтр {field_name!r}. Доступні: "
                f"{', '.join(sorted(allowed_fields))}."
            )
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Для фільтра {field_name!r} потрібно вказати непорожнє значення.")
        value = raw_value.strip().casefold()
        available = {
            str(chunk.metadata.get(field_name, "")).strip().casefold()
            for chunk in chunks
            if str(chunk.metadata.get(field_name, "")).strip()
        }
        if value not in available:
            raise ValueError(
                f"Значення {raw_value!r} не знайдено для фільтра {field_name!r}."
            )
        normalized[field_name] = value
    return normalized


def _matches(chunk: Chunk, filters: dict[str, str]) -> bool:
    return all(
        str(chunk.metadata.get(key, "")).strip().casefold() == value
        for key, value in filters.items()
    )


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    `filters` — умови на метадані фрагмента, наприклад
    `{"category": "інструкція", "audience": "клієнти"}`. Як трактувати
    поле з датою і що робити з невідомим полем — ваше рішення.

    Повертає щонайбільше `top_k` влучень, упорядкованих за спаданням
    оцінки. Оцінка — та сама міра схожості, що й для ранжування, і саме
    вона показується користувачеві.
    """
    if top_k < 1:
        raise ValueError("top_k має бути не менше 1.")
    if threshold is not None and not -1 <= threshold <= 1:
        raise ValueError("Поріг схожості має бути в межах від -1 до 1.")
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.size != index.vectors.shape[1]:
        raise ValueError(
            f"Розмірність запиту ({query.size}) не збігається з індексом "
            f"({index.vectors.shape[1]})."
        )
    if not np.isfinite(query).all():
        raise ValueError("Вектор запиту містить NaN або нескінченні значення.")
    norm = float(np.linalg.norm(query))
    if norm == 0:
        raise ValueError("Нульовий вектор запиту неможливо ранжувати.")
    query /= norm

    checked_filters = validate_filters(index.chunks, filters)
    candidates = [
        position for position, chunk in enumerate(index.chunks)
        if _matches(chunk, checked_filters)
    ]
    if not candidates:
        return []
    scores = index.vectors[candidates] @ query
    order = np.argsort(-scores, kind="stable")
    hits: list[Hit] = []
    for ranked_position in order:
        score = float(scores[ranked_position])
        if threshold is not None and score < threshold:
            continue
        chunk_position = candidates[int(ranked_position)]
        hits.append(Hit(chunk=index.chunks[chunk_position], score=score))
        if len(hits) >= top_k:
            break
    return hits
