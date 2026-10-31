"""Кодування запитів і фрагментів моделлю ембедінгів із налаштувань."""

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


@lru_cache(maxsize=1)
def get_model():
    """Повернути готову до роботи модель.

    Локальна модель вантажиться з диска секунди й займає сотні мегабайт
    памʼяті — створюйте її один раз, а не на кожен запит. Для моделі через
    API тут створюється клієнт (ключ — із `.env`, як у ПР3).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Встановіть sentence-transformers із requirements.txt") from exc
    return SentenceTransformer(MODEL_NAME)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Викликається під час індексування — для всієї колекції одразу.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    prepared = [f"passage: {text}" for text in texts]
    vectors = get_model().encode(prepared, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент, і саме тут це видно.
    """
    vector = get_model().encode(
        [f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True
    )
    return np.asarray(vector[0], dtype=np.float32)
