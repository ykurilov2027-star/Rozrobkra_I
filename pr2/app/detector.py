"""Робота з локальною моделлю YOLO та підготовка результату inference."""

from io import BytesIO
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25


class DetectionError(Exception):
    """Базова помилка, яку веб-рівень може перетворити на HTTP-відповідь."""


class InvalidImageError(DetectionError):
    """Файл порожній або не є зображенням."""


class ModelInferenceError(DetectionError):
    """Модель не змогла обробити коректне зображення."""


_model: YOLO | None = None


def load_model() -> YOLO:
    """Завантажити модель один раз і повернути її для наступних запитів."""
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS)
    return _model


def _read_image(image_bytes: bytes) -> Image.Image:
    """Перевірити байти й повернути незалежну від файлу RGB-копію."""
    if not image_bytes:
        raise InvalidImageError("Файл порожній")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Файл не є коректним зображенням") from exc


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
    """Виконати локальну детекцію та повернути JSON-сумісний результат."""
    if not 0 < confidence <= 1:
        raise DetectionError("Поріг confidence має бути в межах (0, 1]")

    image = _read_image(image_bytes)
    model = load_model()
    started = perf_counter()
    try:
        result = model.predict(source=image, conf=confidence, verbose=False)[0]
    except Exception as exc:
        raise ModelInferenceError("Не вдалося виконати inference моделі") from exc
    inference_time_ms = (perf_counter() - started) * 1000

    names = result.names
    detections = []
    for box in result.boxes:
        score = float(box.conf[0])
        if score < confidence:
            continue
        class_id = int(box.cls[0])
        coordinates = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "class": names[class_id],
                "class_id": class_id,
                "confidence": round(score, 4),
                "box": coordinates,
            }
        )

    return {
        "objects": detections,
        "count": len(detections),
        "inference_time_ms": round(inference_time_ms, 2),
        "confidence_threshold": confidence,
    }
