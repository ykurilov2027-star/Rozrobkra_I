"""Пошук BM25Okapi по тих самих фрагментах і метаданих, що й векторний."""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit, validate_filters


@dataclass
class KeywordIndex:
    """Фрагменти й токенізований корпус для BM25."""

    chunks: list[Chunk]
    extra: dict = field(default_factory=dict)
    tokens: list[list[str]] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде.
    """
    tokens: list[str] = []
    for token in re.findall(r"[^\W_]+(?:-[^\W_]+)*", text.casefold(), flags=re.UNICODE):
        tokens.append(token)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    return KeywordIndex(
        chunks=list(chunks),
        tokens=[tokenize(chunk.text) for chunk in chunks],
    )


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Формат результату — той самий `Hit`, що й у векторного пошуку, з тими
    самими фільтрами за метаданими, щоб сторінка показувала обидва
    способи поруч.
    """
    if top_k < 1:
        raise ValueError("top_k має бути не менше 1.")
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    checked_filters = validate_filters(index.chunks, filters)
    candidate_positions = [
        position for position, chunk in enumerate(index.chunks)
        if all(
            str(chunk.metadata.get(key, "")).strip().casefold() == value
            for key, value in checked_filters.items()
        )
    ]
    if not candidate_positions:
        return []

    candidate_tokens = [index.tokens[position] for position in candidate_positions]
    scorer = BM25Okapi(candidate_tokens)
    scores = scorer.get_scores(query_tokens)
    ranked = sorted(
        range(len(candidate_positions)),
        key=lambda position: (-float(scores[position]), candidate_positions[position]),
    )
    return [
        Hit(
            chunk=index.chunks[candidate_positions[position]],
            score=float(scores[position]),
        )
        for position in ranked
        if float(scores[position]) > 0
    ][:top_k]
