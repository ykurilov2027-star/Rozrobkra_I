"""Зберігання нормалізованих векторів і пошук cosine similarity."""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Скільки результатів повертати за замовчуванням і чи відсікати слабкі.
# Порожній поріг означає «без порога».
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


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
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Кількість рядків векторів має відповідати кількості фрагментів")
    if chunks and matrix.shape[1] == 0:
        raise ValueError("Вектори не можуть мати нульову розмірність")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    return SearchIndex(chunks=chunks, vectors=normalized, model_name=model_name)


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск.

    Формат — ваше рішення: `numpy` для векторів і JSON для фрагментів з
    метаданими — цілком достатньо. Назву моделі зберігайте обовʼязково.
    """
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", index.vectors)
    payload = {
        "model_name": index.model_name,
        "chunks": [
            {"text": chunk.text, "source": chunk.source, "metadata": chunk.metadata}
            for chunk in index.chunks
        ],
    }
    (path / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індексу немає — підняти зрозумілу помилку: веб-рівень має
    сказати користувачеві «індекс не збудовано», а не впасти з
    `FileNotFoundError` десь усередині.
    """
    vectors_path = path / "vectors.npy"
    chunks_path = path / "chunks.json"
    if not vectors_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(f"Індекс не знайдено у {path}; виконайте python ingest.py")
    try:
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload["chunks"]]
        vectors = np.load(vectors_path, allow_pickle=False)
        model_name = str(payload["model_name"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Індекс у {path} пошкоджений або має невідомий формат") from exc
    return build(chunks, vectors, model_name)


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
    if top_k <= 0 or not index.chunks:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.size != index.vectors.shape[1]:
        raise ValueError("Розмірність вектора запиту не відповідає індексу")
    norm = float(np.linalg.norm(query))
    if norm == 0:
        return []
    query = query / norm

    filters = filters or {}
    known_fields = {key for chunk in index.chunks for key in chunk.metadata}
    unknown = set(filters) - known_fields
    if unknown:
        raise ValueError(f"Невідомі поля фільтра: {', '.join(sorted(unknown))}")
    similarities = index.vectors @ query
    candidates: list[Hit] = []
    for chunk, score in zip(index.chunks, similarities):
        if any(str(chunk.metadata.get(key, "")).casefold() != str(value).casefold()
               for key, value in filters.items()):
            continue
        numeric_score = float(score)
        if threshold is not None and numeric_score < threshold:
            continue
        candidates.append(Hit(chunk=chunk, score=numeric_score))
    candidates.sort(key=lambda hit: hit.score, reverse=True)
    return candidates[:top_k]
