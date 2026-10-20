"""Побудова індексу з колекції документів.

Запуск із папки pr5:
    python ingest.py

Читає `docs/`, ділить документи на фрагменти, рахує вектори й зберігає
індекс на диск. Виконується один раз після кожної зміни колекції, моделі
або правила поділу — застосунок при старті лише читає готовий індекс.

Сам скрипт — плумбінг: він викликає функції модулів у правильному
порядку й друкує статистику побудованого індексу. Уся логіка поділу,
кодування та пошуку лишається в модулях `app/`.
"""

import sys
import time
import argparse
from collections import Counter
from pathlib import Path

from app import documents, embeddings, index


def main() -> int:
    parser = argparse.ArgumentParser(description="Побудувати індекс бази знань.")
    parser.add_argument("--docs-dir", type=Path, default=documents.DOCS_DIR)
    parser.add_argument("--index-dir", type=Path, default=index.INDEX_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    docs = documents.load_documents(args.docs_dir)
    print(f"Документів: {len(docs)}")

    chunks = [
        chunk
        for source, metadata, body in docs
        for chunk in documents.split(body, source, metadata)
    ]
    if not chunks:
        print("Не знайдено жодного фрагмента для індексації.", file=sys.stderr)
        return 1
    per_doc = Counter(chunk.source for chunk in chunks)
    lengths = [len(chunk.text) for chunk in chunks]
    print(f"Фрагментів: {len(chunks)}")
    if chunks:
        print(f"  довжина, символів: min {min(lengths)}, "
              f"середня {sum(lengths) // len(lengths)}, max {max(lengths)}")
        for source, count in sorted(per_doc.items()):
            print(f"  {count:3d}  {source}")

    print(f"Модель: {embeddings.MODEL_NAME}")
    t_embed = time.perf_counter()
    vectors = embeddings.embed_passages([chunk.text for chunk in chunks])
    print(f"Вектори: {vectors.shape[0]} x {vectors.shape[1]} "
          f"за {time.perf_counter() - t_embed:.1f} с")

    built = index.build(chunks, vectors, embeddings.MODEL_NAME)
    index.save(built, args.index_dir)
    print(f"Індекс збережено у {args.index_dir} "
          f"({time.perf_counter() - started:.1f} с загалом)")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
