"""Веб-рівень застосунку: сторінка з формою і JSON-ендпоінт.

Цей файл не має знати про `requests`, адреси сервісів і коди їхніх
відповідей — усе це лишається в `app/weather.py`. Тут вирішується інше:
що застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr1:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import weather

app = FastAPI(title="Погода — ПР1")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку з формою вводу міста."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/weather")
def api_weather(city: str):
    """Повернути поточну погоду в місті у форматі JSON.

    Зараз виняток із модуля інтеграції не обробляється — застосунок просто
    впаде з помилкою 500. Спроєктуйте обробку самі: які збої можливі, який
    HTTP-статус відповідає кожному з них і що в такому разі отримає клієнт.
    За критеріями приймання застосунок не має аварійно завершуватися.
    """
    try:
        return weather.get_current_weather(city)
    except weather.WeatherError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
