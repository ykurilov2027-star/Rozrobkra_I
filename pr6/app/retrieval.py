"""Відбір клієнтських фрагментів і побудова контексту в межах бюджету."""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from .documents import Chunk
from . import index as search_index
from .index import Hit, SearchIndex
from .keyword import KeywordIndex, search as KeywordIndexSearch

load_dotenv()

# Скільки фрагментів іде в контекст моделі й скільки токенів на них є.
# Значення — відправна точка; їхнє обґрунтування — результат вашої
# перевірки на наборі запитань, а не здогадка.
CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))
if CONTEXT_CHUNKS <= 0:
  raise ValueError("RAG_CONTEXT_CHUNKS має бути додатним числом")
if CONTEXT_BUDGET <= 0:
  raise ValueError("RAG_CONTEXT_BUDGET має бути додатним числом")
_threshold = os.getenv("SIMILARITY_THRESHOLD", "0.35").strip()
SIMILARITY_THRESHOLD = float(_threshold) if _threshold else 0.35
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві.

    `ref` — номер, під яким фрагмент стоїть у контексті й на який модель
    посилається у відповіді. `chunk` — сам фрагмент із метаданими;
    `score` — оцінка пошуку, за якою його відібрано.
    """

    ref: int
    chunk: Chunk
    score: float


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти, які варто показати моделі.

    Обов'язкові обмеження задаються кодом і мають пріоритет над фільтрами
    клієнта. Значення оцінки тут — косинусна схожість із векторного індексу.
    """
    from . import embeddings

    safe_filters = dict(filters or {})
    safe_filters["audience"] = "клієнти"
    safe_filters["status"] = "чинний"
    query_vector = embeddings.embed_query(query)
    search_top_k = max(search_index.DEFAULT_TOP_K, CONTEXT_CHUNKS)
    hits = search_index.search(
      index,
      query_vector,
      top_k=search_top_k,
      filters=safe_filters,
      threshold=SIMILARITY_THRESHOLD,
    )
    keyword_ranks: dict[tuple[str, int | str], int] = {}
    if keyword_index is not None and hits:
      keyword_hits = KeywordIndexSearch(
        keyword_index,
        query,
        top_k=search_top_k,
        filters=safe_filters,
      )
      keyword_ranks = {
        (
          hit.chunk.source,
          hit.chunk.metadata.get("chunk_id", hit.chunk.metadata.get("chunk_index", "")),
        ): rank
        for rank, hit in enumerate(keyword_hits)
      }

    vector_ranks = {
      (
        hit.chunk.source,
        hit.chunk.metadata.get("chunk_id", hit.chunk.metadata.get("chunk_index", "")),
      ): rank
      for rank, hit in enumerate(hits)
    }

    def fused_rank(hit: Hit) -> tuple[float, float]:
      identity = (
        hit.chunk.source,
        hit.chunk.metadata.get("chunk_id", hit.chunk.metadata.get("chunk_index", "")),
      )
      score = 1 / (60 + vector_ranks[identity])
      if identity in keyword_ranks:
        score += 1 / (60 + keyword_ranks[identity])
      return score, hit.score

    selected: list[Hit] = []
    seen_chunks: set[tuple[str, int | str]] = set()
    seen_documents: dict[str, int] = {}
    for hit in sorted(hits, key=fused_rank, reverse=True):
      if SIMILARITY_THRESHOLD is not None and hit.score < SIMILARITY_THRESHOLD:
        continue
      metadata = hit.chunk.metadata
      if str(metadata.get("audience", "")).strip().casefold() != "клієнти":
        continue
      if str(metadata.get("status", "")).strip().casefold() != "чинний":
        continue
      chunk_id = metadata.get("chunk_id", metadata.get("chunk_index", ""))
      identity = (hit.chunk.source, chunk_id)
      if identity in seen_chunks:
        continue
      # Keep the best chunk per document ahead of repeated adjacent chunks.
      if seen_documents.get(hit.chunk.source, 0) >= 2:
        continue
      seen_chunks.add(identity)
      seen_documents[hit.chunk.source] = seen_documents.get(hit.chunk.source, 0) + 1
      selected.append(hit)
      if len(selected) >= CONTEXT_CHUNKS:
        break
    return selected


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та перелік джерел.

    Найкраще влучення розміщується останнім. Бюджет оцінюється за словами,
    пунктуацією і символами; це наближення, а не токенізатор провайдера.
    """
    if budget <= 0:
      return "", []

    ranked_hits = sorted(hits, key=lambda item: item.score, reverse=True)[:CONTEXT_CHUNKS]
    fitting_hits: list[Hit] = []
    used_tokens = 0
    for hit in ranked_hits:
      metadata = hit.chunk.metadata
      title = str(metadata.get("title") or hit.chunk.source)
      heading = str(metadata.get("heading") or "Без розділу")
      updated = str(metadata.get("updated") or "дату не вказано")
      reference = max(CONTEXT_CHUNKS, len(fitting_hits) + 1)
      header = f"[{reference}] {title} > {heading} ({updated}):"
      block = f'{header}\n"{hit.chunk.text}"'
      block_tokens = max(len(TOKEN_PATTERN.findall(block)), (len(block) + 1) // 2)
      separator_tokens = 1 if fitting_hits else 0
      if used_tokens + separator_tokens + block_tokens > budget:
        continue
      fitting_hits.append(hit)
      used_tokens += separator_tokens + block_tokens

    if len(fitting_hits) > 1:
      fitting_hits = fitting_hits[1:] + fitting_hits[:1]

    blocks: list[str] = []
    sources: list[Source] = []
    for ref, hit in enumerate(fitting_hits, start=1):
      metadata = hit.chunk.metadata
      title = str(metadata.get("title") or hit.chunk.source)
      heading = str(metadata.get("heading") or "Без розділу")
      updated = str(metadata.get("updated") or "дату не вказано")
      blocks.append(f'[{ref}] {title} > {heading} ({updated}):\n"{hit.chunk.text}"')
      sources.append(Source(ref=ref, chunk=hit.chunk, score=hit.score))
    return "\n\n".join(blocks), sources
