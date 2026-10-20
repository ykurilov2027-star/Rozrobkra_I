"""Evaluate semantic and BM25 retrieval and compare E5 prefix ablation.

Run from any directory with:
    python compare/run.py
The script writes the measured per-query results and aggregate metrics to
compare/findings.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app import documents, embeddings, index, keyword

QUERY_FILE = ROOT / "compare" / "queries.json"
FINDINGS_FILE = ROOT / "compare" / "findings.md"
TOP_K = 3


def rank_of(hits: list[index.Hit], expected: str) -> int | None:
    if not expected:
        return None
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.source == expected:
            return rank
    return None


def metrics(rows: list[dict[str, Any]], key: str) -> tuple[float, float, int]:
    answerable = [row for row in rows if row["expected_document"]]
    if not answerable:
        return 0.0, 0.0, 0
    ranks = [row[key]["rank"] for row in answerable]
    return (
        sum(rank == 1 for rank in ranks) / len(answerable),
        sum(rank is not None and rank <= 3 for rank in ranks) / len(answerable),
        len(answerable),
    )


def execute(
    queries: list[dict[str, Any]],
    chunks: list[documents.Chunk],
    passage_vectors: Any,
    use_prefixes: bool,
    keyword_index: keyword.KeywordIndex,
) -> list[dict[str, Any]]:
    semantic_index = index.build(chunks, passage_vectors, embeddings.MODEL_NAME)
    rows: list[dict[str, Any]] = []
    for item in queries:
        query_text = item["query"]
        filters = item.get("filters") or None

        started = time.perf_counter()
        query_vector = embeddings.embed_query(query_text, use_prefixes=use_prefixes)
        semantic_hits = index.search(
            semantic_index, query_vector, top_k=TOP_K, filters=filters, threshold=None
        )
        semantic_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        keyword_hits = keyword.search(keyword_index, query_text, top_k=TOP_K, filters=filters)
        keyword_elapsed = time.perf_counter() - started

        rows.append({
            "type": item["type"],
            "query": query_text,
            "expected_document": item["expected_document"],
            "checks": item.get("checks", ""),
            "filters": filters,
            "semantic": {
                "rank": rank_of(semantic_hits, item["expected_document"]),
                "first": semantic_hits[0].chunk.source if semantic_hits else "немає",
                "score": semantic_hits[0].score if semantic_hits else None,
                "elapsed_ms": semantic_elapsed * 1000,
            },
            "keyword": {
                "rank": rank_of(keyword_hits, item["expected_document"]),
                "first": keyword_hits[0].chunk.source if keyword_hits else "немає",
                "score": keyword_hits[0].score if keyword_hits else None,
                "elapsed_ms": keyword_elapsed * 1000,
            },
        })
    return rows


def display_rank(rank: int | None, expected: str) -> str:
    if not expected:
        return "поза колекцією"
    return str(rank) if rank is not None else f"поза top-{TOP_K}"


def write_findings(
    queries: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    unprefixed: list[dict[str, Any]],
) -> None:
    semantic_1, semantic_3, denominator = metrics(baseline, "semantic")
    keyword_1, keyword_3, _ = metrics(baseline, "keyword")
    no_prefix_1, no_prefix_3, _ = metrics(unprefixed, "semantic")
    threshold = index.SIMILARITY_THRESHOLD
    out_of_scope = [row for row in baseline if not row["expected_document"]]
    out_scores = [
        row["semantic"]["score"] for row in out_of_scope
        if row["semantic"]["score"] is not None
    ]
    lines = [
        "# Порівняльний експеримент",
        "",
        f"Вибірка: {len(queries)} запитів, з них {denominator} мають очікуваний документ. "
        f"Top-k={TOP_K}; метрики hit@1/hit@3 рахуються тільки для запитів із відповіддю.",
        "",
        "## Основний прогін",
        "",
        "| Метод | hit@1 | hit@3 |",
        "|---|---:|---:|",
        f"| Семантичний, з префіксами E5 | {semantic_1:.1%} | {semantic_3:.1%} |",
        f"| BM25Okapi | {keyword_1:.1%} | {keyword_3:.1%} |",
        "",
        "## Одна зміна: прибрано префікси E5",
        "",
        "| Варіант | hit@1 | hit@3 |",
        "|---|---:|---:|",
        f"| `query:` / `passage:` увімкнено | {semantic_1:.1%} | {semantic_3:.1%} |",
        f"| Префікси вимкнено | {no_prefix_1:.1%} | {no_prefix_3:.1%} |",
        "",
        f"Вплив префіксів проти варіанта без них: hit@1 {(semantic_1 - no_prefix_1) * 100:+.1f} "
        f"відсоткового пункту; hit@3 {(semantic_3 - no_prefix_3) * 100:+.1f} відсоткового пункту. "
        "Префікси — єдина змінена умова; документи, розмір фрагмента, запити й top-k однакові.",
        "",
        "## Поріг схожості й запит поза колекцією",
        "",
        (
            f"Налаштований поріг: `{threshold}`. "
            if threshold is not None else
            "Поріг `SIMILARITY_THRESHOLD` не задано; семантичний пошук не відсікає слабкі збіги. "
        ),
        (
            f"Найвища оцінка top-1 для запитів поза колекцією: "
            f"{max(out_scores):.3f}. "
            if out_scores else "У наборі немає запитів поза колекцією. "
        ),
        (
            (
                f"За заданого порогу запит(и) поза колекцією "
                f"{'будуть відсічені' if all(score < threshold for score in out_scores) else 'не всі будуть відсічені'}."
            )
            if threshold is not None and out_scores else
            "За порожнього порогу навіть сторонній запит отримає найкращий наявний фрагмент; "
            "поріг слід калібрувати за відповідними й позаколекційними запитами, а не вважати оцінку "
            "косинусної схожості ймовірністю."
        ),
        "",
        "## Результати за запитами",
        "",
        "| Тип | Запит | Очікуваний документ | Семантичний ранг / top-1 / score / ms | BM25 ранг / top-1 / score / ms | Без префіксів, ранг |",
        "|---|---|---|---|---|---:|",
    ]
    for position, row in enumerate(baseline):
        no_prefix = unprefixed[position]["semantic"]
        expected = row["expected_document"] or "немає"
        semantic = row["semantic"]
        bm25 = row["keyword"]
        lines.append(
            f"| {row['type']} | {row['query']} | {expected} | "
            f"{display_rank(semantic['rank'], row['expected_document'])} / "
            f"{semantic['first']} / {semantic['score'] if semantic['score'] is not None else '—'} / "
            f"{semantic['elapsed_ms']:.1f} | "
            f"{display_rank(bm25['rank'], row['expected_document'])} / "
            f"{bm25['first']} / {bm25['score'] if bm25['score'] is not None else '—'} / "
            f"{bm25['elapsed_ms']:.1f} | "
            f"{display_rank(no_prefix['rank'], row['expected_document'])} |"
        )
    lines.extend([
        "",
        "Час семантичного пошуку включає кодування запиту; час BM25 не включає побудову індексу.",
        "Для запиту поза колекцією позиція очікуваного документа не застосовується; оцінки top-1 "
        "показують, що пошук усе одно ранжує.",
        "",
    ])
    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    data = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    queries = data["queries"]
    chunks = documents.load_chunks()
    if not chunks:
        raise RuntimeError(f"У {documents.DOCS_DIR} не знайдено документів для порівняння.")
    keyword_index = keyword.build(chunks)
    texts = [chunk.text for chunk in chunks]

    print(f"Запитів: {len(queries)}; фрагментів: {len(chunks)}; модель: {embeddings.MODEL_NAME}")
    print("Розрахунок векторів із префіксами E5…")
    baseline_vectors = embeddings.embed_passages(texts, use_prefixes=True)
    print("Розрахунок векторів без префіксів E5…")
    unprefixed_vectors = embeddings.embed_passages(texts, use_prefixes=False)
    baseline = execute(queries, chunks, baseline_vectors, True, keyword_index)
    unprefixed = execute(queries, chunks, unprefixed_vectors, False, keyword_index)
    write_findings(queries, baseline, unprefixed)

    for method in ("semantic", "keyword"):
        hit_1, hit_3, count = metrics(baseline, method)
        print(f"{method}: hit@1={hit_1:.1%}, hit@3={hit_3:.1%} ({count} запитів із відповіддю)")
    no_prefix_1, no_prefix_3, _ = metrics(unprefixed, "semantic")
    print(f"semantic без префіксів: hit@1={no_prefix_1:.1%}, hit@3={no_prefix_3:.1%}")
    print(f"Повний звіт записано: {FINDINGS_FILE}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
