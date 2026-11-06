"""Перевірка середовища для ПР10.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів,
налаштування в `.env`, зразки звернень, реалізацію модулів і — якщо
обрано справжнього провайдера — робить пробний запит у режимі
streaming: вимірює час до першого фрагмента й загальний час.

Запустіть перевірку заздалегідь: значення таймаутів у `.env` варто
брати з того, що ви тут побачите, а не навмання.

Скрипт призначений для локальної діагностики перед запуском застосунку.
"""

from importlib.util import find_spec
import json
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic")
HERE = Path(__file__).parent
STUBS = ("app/providers/openai_compat.py", "app/providers/mock.py", "app/gateway.py",
         "app/drafting.py", "app/jobs.py", "app/limits.py")

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: перенесіть ключ із .env попередніх робіт або візьміть "
                   "у Google AI Studio (aistudio.google.com)",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
}

PROBE_MESSAGES = [
    {"role": "system", "content": "Ти — оператор підтримки інтернет-магазину. Відповідай українською, "
                                  "коротко й ввічливо."},
    {"role": "user", "content": "Клієнт пише: «Де моє замовлення 10458?». Відомо: передано Новій пошті "
                                "4 жовтня, накладна 20450012345678, відділення № 12 у Києві. "
                                "Склади відповідь клієнтові з трьох речень."},
]


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def note(what: str, hint: str) -> None:
    """Зауваження, яке не є помилкою середовища."""
    print(f"[ .. ] {what} — {hint}")


def check_python() -> bool:
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    need = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, actual))
    report(ok, f"Python {have}", "" if ok else f"потрібен Python {need} або новіший")
    return ok


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            found = find_spec(name)
        except (ImportError, ValueError):
            found = None
        if found is None:
            report(False, f"пакет {name}", "не встановлено: pip install -r requirements.txt")
            ok = False
        else:
            report(True, f"пакет {name}", "")
    return ok


def check_settings() -> tuple[bool, str]:
    """Перевірити `.env`. Повертає (ok, провайдер). Значення ключа не
    друкуємо."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False, ""

    if not Path(".env").exists():
        report(False, "файл .env", "немає: скопіюйте .env.example у .env і перенесіть ключ")
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai")
    fallback = os.getenv("LLM_FALLBACK", "none")
    report(provider in ("openai", "mock"), f"провайдер {provider}",
           "" if provider in ("openai", "mock") else "очікується openai або mock")
    report(True, f"запасний варіант {fallback}", "")

    ok = True
    if provider == "openai" or fallback not in ("none", "mock"):
        for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
            value = os.getenv(name)
            if not value:
                report(False, f"налаштування {name}", HINTS[name])
                ok = False
            elif name == "LLM_API_KEY":
                report(True, f"налаштування {name}", f"задано, довжина {len(value)}")
            else:
                report(True, f"налаштування {name}", value)
    else:
        note("ключ доступу", "не потрібен, поки провайдер — mock; справжнього провайдера "
                             "перевірте, коли переключите LLM_PROVIDER=openai")
    for name in ("LLM_TIMEOUT", "LLM_IDLE_TIMEOUT", "LLM_TOTAL_TIMEOUT", "LLM_MAX_ATTEMPTS",
                 "LLM_MAX_CONCURRENT", "RATE_LIMIT_PER_MINUTE"):
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із коду")
    return ok, provider


def check_samples() -> bool:
    path = HERE / "samples" / "requests.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = [s["id"] for s in data]
        missing = [s["id"] for s in data if not s.get("facts") or not s.get("reply_example")]
    except (OSError, ValueError, KeyError) as exc:
        report(False, "зразки звернень", f"не читаються ({type(exc).__name__}): {exc} — "
                                          "перевірте, чи повністю розпаковано архів")
        return False
    report(not missing, f"зразки звернень: {len(ids)} ({', '.join(ids)})",
           "" if not missing else f"без фактів або прикладу відповіді: {', '.join(missing)}")
    return not missing


def check_stubs() -> None:
    """Перевірити, що основні модулі більше не містять заглушок."""
    pending = []
    for rel in STUBS:
        text = (HERE / rel).read_text(encoding="utf-8", errors="replace")
        if "NotImplementedError" in text:
            pending.append(rel)
    if pending:
        note("реалізація", "залишилися заглушки: " + ", ".join(pending))
    else:
        report(True, "реалізація модулів", "заглушок не знайдено")


def check_call(provider: str) -> bool:
    """Пробний запит у режимі streaming до справжнього провайдера."""
    if provider != "openai":
        note("пробний запит", "пропущено: провайдер — mock")
        return True

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model):
        report(False, "пробний запит", "перевірку пропущено: налаштування неповні")
        return False

    from app.providers.base import ProviderError, Request
    from app.providers.openai_compat import OpenAICompatProvider

    timeout = float(os.getenv("LLM_TIMEOUT", "20"))
    client = OpenAICompatProvider(base_url, api_key, model, timeout)
    started = time.perf_counter()
    first = None
    chunks = 0
    text = []
    try:
        request = Request(messages=PROBE_MESSAGES, temperature=0.3,
                          max_tokens=200, model=model)
        for chunk in client.stream(request):
            if first is None:
                first = time.perf_counter() - started
            chunks += 1
            if chunk.text:
                text.append(chunk.text)
    except ProviderError as exc:
        report(False, "пробний запит", f"збій виду {exc.kind}: {exc.message}")
        return False

    total = time.perf_counter() - started
    answer = "".join(text).strip()
    if not answer:
        report(False, "пробний запит", f"потік завершився без тексту ({chunks} фрагментів)")
        return False
    report(True, "пробний запит у режимі streaming",
           f"перший фрагмент через {first:.2f} с, разом {total:.2f} с, фрагментів {chunks}")
    print(f"       {answer[:160]}{'…' if len(answer) > 160 else ''}")
    note("таймаути", f"LLM_TIMEOUT={timeout:.0f} с на спробу; перший фрагмент прийшов за {first:.1f} с — "
                     "від цього відштовхуйтеся, обираючи LLM_IDLE_TIMEOUT і LLM_TOTAL_TIMEOUT")
    return True


def main() -> int:
    print("Перевірка середовища для ПР10\n")
    results = [check_python(), check_packages()]
    settings_ok, provider = check_settings()
    results += [settings_ok, check_samples()]
    check_stubs()
    results.append(check_call(provider))
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
