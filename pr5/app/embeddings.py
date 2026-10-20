"""Завантаження локальної моделі ембедінгів і кодування текстів."""

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Повернути готову до роботи модель.

    Локальна модель вантажиться з диска секунди й займає сотні мегабайт
    памʼяті — створюйте її один раз, а не на кожен запит. Для моделі через
    API тут створюється клієнт (ключ — із `.env`, як у ПР3).
    """
    return SentenceTransformer(MODEL_NAME, device=os.getenv("EMBEDDING_DEVICE", "cpu"))


def _encode(texts: list[str], role: str, use_prefixes: bool = True) -> np.ndarray:
    if not texts:
        model = get_model()
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = getattr(model, "get_sentence_embedding_dimension", None)
        dimension = dimension_getter() if dimension_getter is not None else None
        if dimension is None:
            raise RuntimeError(f"Не вдалося визначити розмірність моделі {MODEL_NAME}.")
        return np.empty((0, dimension), dtype=np.float32)
    prefix = f"{role}: " if use_prefixes and "e5" in MODEL_NAME.lower() else ""
    vectors = get_model().encode(
        [prefix + text for text in texts],
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        show_progress_bar=len(texts) > 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(texts):
        raise RuntimeError(f"Модель повернула неочікувану форму ембедінгів: {result.shape}.")
    if not np.isfinite(result).all():
        raise RuntimeError("Модель повернула ембедінги з нескінченними або NaN значеннями.")
    return result


def embed_passages(texts: list[str], use_prefixes: bool = True) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Викликається під час індексування — для всієї колекції одразу.
    """
    return _encode(texts, "passage", use_prefixes=use_prefixes)


def embed_query(text: str, use_prefixes: bool = True) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент, і саме тут це видно.
    """
    if not text.strip():
        raise ValueError("Пошуковий запит не може бути порожнім.")
    return _encode([text], "query", use_prefixes=use_prefixes)[0]
