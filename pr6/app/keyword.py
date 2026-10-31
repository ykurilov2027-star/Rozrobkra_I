"""Індекс пошуку за словами з BM25 і фільтрами за метаданими."""

from dataclasses import dataclass, field
import re

try:
  from rank_bm25 import BM25Okapi
except ImportError:  # Keep the application usable without the optional package.
  BM25Okapi = None

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами. Що саме тут зберігати — вирішуєте ви."""

    chunks: list[Chunk]
    extra: dict = field(default_factory=dict)
    scorer: object | None = None


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде.
    """
    return re.findall(r"[\wʼ'’-]+", text.casefold(), flags=re.UNICODE)


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    corpus = [tokenize(chunk.text) for chunk in chunks]
    scorer = BM25Okapi(corpus) if BM25Okapi is not None and corpus else None
    return KeywordIndex(chunks=chunks, scorer=scorer, extra={"corpus": corpus})


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
    if top_k <= 0 or not index.chunks:
      return []
    tokens = tokenize(query)
    if not tokens:
      return []
    corpus = index.extra.get("corpus") or [tokenize(chunk.text) for chunk in index.chunks]
    scores = index.scorer.get_scores(tokens) if index.scorer is not None else [
      sum(tokens.count(token) for token in set(document)) for document in corpus
    ]
    filters = filters or {}
    known_fields = {key for chunk in index.chunks for key in chunk.metadata}
    unknown = set(filters) - known_fields
    if unknown:
      raise ValueError(f"Невідомі поля фільтра: {', '.join(sorted(unknown))}")
    hits = []
    for chunk, score in zip(index.chunks, scores):
      if score <= 0 or any(
        str(chunk.metadata.get(key, "")).casefold() != str(value).casefold()
        for key, value in filters.items()
      ):
        continue
      hits.append(Hit(chunk=chunk, score=float(score)))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]
