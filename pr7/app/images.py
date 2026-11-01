"""Підготовка зображення до надсилання моделі.

Файл, який надіслав користувач, — недовірені байти: це може бути не
зображення, зображення завбільшки 40 мегапікселів, фото, повернуте на
бік (орієнтація записана в EXIF, а не в пікселях), або знімок 300 × 200,
на якому цифр не розібрати. Модуль вирішує, що з цього можна віддати
моделі і в якому вигляді. Про модель він не знає: повертає підготовлені
байти й відомості про них, а як вкласти їх у запит — справа `app/llm.py`.

Функція `prepare` перевіряє та готує зображення згідно з налаштуваннями:

* що вважати неприйнятним файлом і яку помилку повернути: не
  зображення, порожній файл, завеликий файл, завеликий у пікселях;
* чи враховувати орієнтацію з EXIF (фото з телефона);
* до якого розміру зменшувати і чи зменшувати взагалі — ціна зображення
  в токенах росте з роздільністю, а дрібний шрифт зі зменшенням
  зливається;
* у якому форматі надсилати: PNG без втрат чи JPEG, який менший;
* чи відсіювати завідомо непридатні знімки ще до моделі (надто малі,
  надто розмиті) — і що тоді сказати користувачеві.
"""

import io
import os
import warnings
from dataclasses import dataclass

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))
IMAGE_MAX_PIXELS = int(os.getenv("IMAGE_MAX_PIXELS", "40000000"))


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""

    def __init__(self, message: str, status_code: int = 415):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання.

    `data` і `mime` — те, що піде моделі. `original` і `sent` — розміри
    до й після підготовки, наприклад `{"width": 1240, "height": 1754,
    "bytes": 179639}`: сторінка показує їх поруч із токенами, щоб було
    видно, скільки коштує роздільність.
    """

    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі.

    Непридатний файл — `ImageError` з поясненням, що не так.
    """
    if not content:
        raise ImageError("Файл порожній.", 400)
    if len(content) > UPLOAD_MAX_BYTES:
        limit_mb = UPLOAD_MAX_BYTES / (1024 * 1024)
        raise ImageError(f"Файл завеликий. Максимальний розмір — {limit_mb:g} МБ.", 413)
    if IMAGE_MAX_SIDE < 1 or IMAGE_MAX_PIXELS < 1:
        raise ImageError("Неправильно налаштовані обмеження зображення.", 500)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                width, height = source.size
                if width * height > IMAGE_MAX_PIXELS:
                    raise ImageError("Зображення має надто високу роздільність.", 413)
                if source.format not in {"PNG", "JPEG", "WEBP", "TIFF", "BMP"}:
                    raise ImageError("Підтримуються зображення PNG, JPEG, WEBP, TIFF і BMP.", 415)
                source.load()
                oriented = ImageOps.exif_transpose(source)
                original = {
                    "width": oriented.width,
                    "height": oriented.height,
                    "bytes": len(content),
                }
                image = oriented.convert("RGB")
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError("Зображення має надто високу роздільність.", 413) from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError):
        raise ImageError("Файл не є коректним зображенням.", 415) from None

    image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    prepared = output.getvalue()
    return PreparedImage(
        data=prepared,
        mime="image/jpeg",
        original=original,
        sent={"width": image.width, "height": image.height, "bytes": len(prepared)},
    )
