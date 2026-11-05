"""Перевірка середовища для ПР9.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, налаштування
в `.env`, те, що сервіс магазину читає свої дані, що модулі з ПР8
перенесено в `app/`, що каталог стану звернень доступний для запису, і
те, що модель повертає structured output за схемою: розбирає коротке
звернення клієнта за пробною схемою.

Запустіть перевірку заздалегідь: якщо ключ не працює або провайдер не
приймає схему, зʼясувати це краще до заняття.

Це діагностика перед роботою, а не зразок для наслідування: тут немає
ані перевірки результату за схемою, ані кроків workflow — саме це ви
проєктуєте самі.
"""

import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic")
OPTIONAL_PACKAGES = ("jsonschema",)
SETTINGS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
OPTIONAL_SETTINGS = ("LLM_TIMEOUT", "TOOL_MAX_ROUNDS", "WORKFLOW_MAX_STEPS", "WORKFLOW_TIMEOUT",
                     "WORKFLOW_TOKEN_BUDGET", "RUNS_DIR", "SHOP_TODAY")
HERE = Path(__file__).parent
FROM_PR8 = ("tools.py", "llm.py", "assistant.py")
STUB_MARK = "Заготовка з ПР8"

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: перенесіть ключ із .env попередніх робіт або візьміть "
                   "у Google AI Studio (aistudio.google.com)",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
}

# Пробна схема. Звернення навмисно без номера замовлення: подивіться, що
# модель поставить у поле, якого клієнт не назвав.
PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["return", "cancel", "status", "other"]},
        "order_id": {"type": ["string", "null"]},
        "item": {"type": ["string", "null"]},
    },
    "required": ["kind", "order_id", "item"],
    "additionalProperties": False,
}
PROBE_TEXT = "Добрий день! Навушники, які я у вас купила, мені не підійшли, хочу їх повернути."
PROBE_PROMPT = ("Визнач вид звернення клієнта магазину, номер замовлення і товар. "
                "Чого клієнт не назвав — null.\n\nЗвернення: " + PROBE_TEXT)


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
            importlib.import_module(name)
        except ImportError:
            report(False, f"пакет {name}", "не встановлено: pip install -r requirements.txt")
            ok = False
        else:
            report(True, f"пакет {name}", "")
    for name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            note(f"пакет {name}", "не встановлено; потрібен лише для перевірки за JSON Schema")
        else:
            report(True, f"пакет {name}", "")
    return ok


def check_settings() -> bool:
    """Перевірити, що .env заповнений. Значення ключа не друкуємо."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False

    if not Path(".env").exists():
        report(False, "файл .env", "немає: скопіюйте .env.example у .env і перенесіть ключ")
    load_dotenv()
    ok = True
    for name in SETTINGS:
        value = os.getenv(name)
        if not value:
            report(False, f"налаштування {name}", HINTS[name])
            ok = False
        elif name == "LLM_API_KEY":
            report(True, f"налаштування {name}", f"задано, довжина {len(value)}")
        else:
            report(True, f"налаштування {name}", value)
    for name in OPTIONAL_SETTINGS:
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із коду")
    return ok


def check_shop() -> bool:
    """Сервіс магазину читає дані й відповідає."""
    sys.path.insert(0, str(HERE))
    try:
        from shop import service
        customers = service.list_customers()
        orders = service.list_orders(customers[0]["customer_id"])
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "сервіс магазину", f"не працює ({type(exc).__name__}): {exc} — "
                                         "перевірте, чи повністю розпаковано архів")
        return False
    report(True, "сервіс магазину",
           f"клієнтів {len(customers)}, замовлень першого клієнта {len(orders)}, "
           f"сьогодні {service.today()}")
    return True


def check_pr8_modules() -> bool:
    """Модулі з ПР8 перенесено, а не лишено заготовками."""
    ok = True
    for name in FROM_PR8:
        path = HERE / "app" / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            report(False, f"app/{name}", f"не читається: {exc}")
            ok = False
            continue
        if STUB_MARK in text:
            report(False, f"app/{name}", f"ще заготовка: замініть своїм файлом з pr8/app/{name}")
            ok = False
        else:
            report(True, f"app/{name}", "перенесено з ПР8")
    llm_text = (HERE / "app" / "llm.py").read_text(encoding="utf-8", errors="replace")
    if STUB_MARK not in llm_text and "def complete" not in llm_text:
        note("app/llm.py", "немає функції complete: workflow потрібен виклик моделі без "
                           "інструментів — допишіть його (заготовка є в архіві)")
    return ok


def check_runs_dir() -> bool:
    """Каталог стану звернень існує або може бути створений, і в нього
    можна писати."""
    runs = Path(os.getenv("RUNS_DIR", "runs"))
    probe = runs / f".probe-{uuid.uuid4().hex}"
    try:
        runs.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        report(False, f"каталог стану {runs}", f"недоступний для запису: {exc}")
        return False
    count = len([p for p in runs.iterdir() if not p.name.startswith(".")])
    report(True, f"каталог стану {runs.resolve()}", f"записів у ньому: {count}")
    return True


def check_call() -> bool:
    """Попросити модель розібрати звернення за пробною схемою."""
    try:
        from openai import BadRequestError, OpenAI
    except ImportError:
        report(False, "пробний запит", "перевірку пропущено: немає пакета openai")
        return False

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model):
        report(False, "пробний запит", "перевірку пропущено: налаштування неповні")
        return False

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30)
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    started = time.perf_counter()
    try:
        answer = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=200,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "probe", "schema": PROBE_SCHEMA}},
        )
    except BadRequestError as exc:
        report(False, "пробний запит зі схемою",
               f"провайдер відхилив схему ({exc.status_code}): {exc} — для розбору звернення "
               "доведеться просити JSON текстом і перевіряти самим")
        return False
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний запит зі схемою", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    text = (answer.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        report(False, "пробний запит зі схемою", f"схему прийнято, але відповідь не JSON: {text[:80]!r}")
        return False
    report(True, "пробний запит зі схемою", f"відповідь за {elapsed:.2f} с: {data}")
    if data.get("order_id"):
        note("номер замовлення", f"клієнт його не називав, а модель поставила {data['order_id']!r} — "
                                 "саме такі випадки має ловити ваша перевірка")
    usage = getattr(answer, "usage", None)
    if usage:
        print(f"       токенів: запит {usage.prompt_tokens}, відповідь {usage.completion_tokens}")
    return True


def main() -> int:
    print("Перевірка середовища для ПР9\n")
    results = [check_python(), check_packages(), check_settings(), check_shop(),
               check_pr8_modules(), check_runs_dir(), check_call()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
