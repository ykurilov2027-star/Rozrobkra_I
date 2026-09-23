"""Модуль інтеграції із зовнішнім API погоди.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі рішення
з розділу 4 практичної роботи:

* як передати параметри запиту, не склеюючи URL вручну;
* яке обмеження часу (timeout) поставити й що робити, коли воно спрацювало;
* чи однаково реагувати на помилку клієнта (4xx) і сервера (5xx);
* як повестися, коли міста не знайдено або у відповіді немає потрібних полів;
* що саме віддавати назовні при успіху і як позначати помилку.

Реальні відповіді обох сервісів збережено в папці `samples/` — подивіться їх
перед тим, як писати розбір відповіді.
"""

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 10


class WeatherError(Exception):
    """Помилка отримання погоди, зрозуміла веб-рівню.

    Заготовка. Вирішіть, чи достатньо одного типу помилки, чи їх варто
    розрізняти — місто не знайдено, сервіс недоступний, відповідь не та,
    якої очікували. Від цього залежить, який HTTP-статус поверне застосунок
    і що побачить користувач.
    """

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _request_json(url: str, params: dict) -> dict:
    """Виконати зовнішній запит і повернути JSON або зрозумілу помилку."""
    import requests

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise WeatherError("Сервіс Open-Meteo не відповів вчасно.", 504) from exc
    except requests.ConnectionError as exc:
        raise WeatherError("Не вдалося підключитися до сервісу Open-Meteo.", 503) from exc
    except requests.HTTPError as exc:
        status_code = response.status_code
        if 400 <= status_code < 500:
            message = "Сервіс Open-Meteo відхилив запит."
        else:
            message = "Сервіс Open-Meteo тимчасово недоступний."
        raise WeatherError(message, 502) from exc
    except requests.RequestException as exc:
        raise WeatherError("Помилка під час звернення до сервісу Open-Meteo.", 502) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise WeatherError("Сервіс Open-Meteo повернув некоректний JSON.", 502) from exc

    if not isinstance(data, dict):
        raise WeatherError("Сервіс Open-Meteo повернув неочікуваний формат даних.", 502)
    return data


def find_city(name: str) -> dict:
    """Знайти координати міста за його назвою.

    Що саме повертати — вирішіть самі: пару чисел, словник, окремий тип.
    Врахуйте випадок, коли міста з такою назвою немає.
    """
    city_name = name.strip()
    if not city_name:
        raise WeatherError("Назва міста не може бути порожньою.", 400)

    data = _request_json(
        GEOCODING_URL,
        {"name": city_name, "count": 1, "language": "uk", "format": "json"},
    )
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise WeatherError(f"Місто «{city_name}» не знайдено.", 404)

    city = results[0]
    if not isinstance(city, dict):
        raise WeatherError("Сервіс геокодування повернув неочікувані дані.", 502)
    latitude = city.get("latitude")
    longitude = city.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise WeatherError("У відповіді геокодування відсутні координати міста.", 502)
    return {"name": city.get("name", city_name), "latitude": latitude, "longitude": longitude}


def get_current_weather(city: str) -> dict:
    """Повернути поточну погоду в місті: температуру й швидкість вітру.

    Це функція, яку викликає веб-рівень. Вона поєднує геокодування і запит
    прогнозу та віддає результат у зручному для застосунку вигляді.
    """
    location = find_city(city)
    data = _request_json(
        FORECAST_URL,
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,wind_speed_10m",
        },
    )
    current = data.get("current")
    if not isinstance(current, dict):
        raise WeatherError("У відповіді прогнозу відсутні поточні дані.", 502)

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")
    if not isinstance(temperature, (int, float)) or not isinstance(wind_speed, (int, float)):
        raise WeatherError("У відповіді прогнозу відсутні потрібні показники.", 502)

    return {
        "city": location["name"],
        "temperature": temperature,
        "wind_speed": wind_speed,
    }
