"""Service boundary for semantic and keyword retrieval."""

import time

from . import embeddings, index, keyword


class SearchUnavailableError(RuntimeError):
    """Search cannot run because its index or embedding model is unavailable."""


class SearchService:
    """Own the indexes and retrieval implementation behind a small API."""

    def __init__(self) -> None:
        self._index = None
        self._keyword_index = None
        self._index_error: str | None = None
        try:
            self._index = index.load()
        except index.IndexLoadError as exc:
            self._index_error = str(exc)
            return
        self._keyword_index = keyword.build(self._index.chunks)

    def status(self) -> dict:
        result = {
            "ready": self._index is not None,
            "default_top_k": index.DEFAULT_TOP_K,
            "similarity_threshold": index.SIMILARITY_THRESHOLD,
        }
        if self._index is None:
            result["hint"] = self._index_error or "індекс не збудовано — виконайте python ingest.py"
            return result
        result.update({
            "chunks": len(self._index),
            "documents": len({chunk.source for chunk in self._index.chunks}),
            "model": self._index.model_name,
        })
        return result

    def search(
        self,
        query: str,
        mode: str,
        top_k: int | None,
        filters: dict[str, str] | None,
        threshold: float | None,
    ) -> dict:
        if self._index is None or self._keyword_index is None:
            raise SearchUnavailableError(
                self._index_error or "Індекс не збудовано — виконайте `python ingest.py`."
            )
        effective_top_k = top_k if top_k is not None else index.DEFAULT_TOP_K
        index.validate_filters(self._index.chunks, filters)

        result: dict = {"query": query, "semantic": None, "keyword": None, "elapsed": {}}
        if mode in ("semantic", "both"):
            started = time.perf_counter()
            try:
                vector = embeddings.embed_query(query)
            except (OSError, RuntimeError, ValueError) as exc:
                raise SearchUnavailableError(
                    f"Модель ембедінгів недоступна або не змогла обробити запит: {exc}"
                ) from exc
            hits = index.search(
                self._index,
                vector,
                top_k=effective_top_k,
                filters=filters,
                threshold=(
                    threshold
                    if threshold is not None
                    else index.SIMILARITY_THRESHOLD
                ),
            )
            result["elapsed"]["semantic"] = time.perf_counter() - started
            result["semantic"] = [self._hit_to_dict(hit) for hit in hits]

        if mode in ("keyword", "both"):
            started = time.perf_counter()
            hits = keyword.search(
                self._keyword_index,
                query,
                top_k=effective_top_k,
                filters=filters,
            )
            result["elapsed"]["keyword"] = time.perf_counter() - started
            result["keyword"] = [self._hit_to_dict(hit) for hit in hits]
        return result

    @staticmethod
    def _hit_to_dict(hit: index.Hit) -> dict:
        return {
            "score": hit.score,
            "text": hit.chunk.text,
            "source": hit.chunk.source,
            "metadata": hit.chunk.metadata,
        }
