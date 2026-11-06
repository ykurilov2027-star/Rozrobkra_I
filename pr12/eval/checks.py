"""Перевірки результату: твердження про вихід функції, які перевіряє код.

Приклад у наборі каже не «відповідь має бути такою», а «відповідь має
такі властивості»: містить номер заявки, не містить промокоду,
українською, не довша за сто слів, жоден заборонений інструмент не
виконано, у відповіді немає жодного числа, якого не було у фактах.
Кожна властивість — одна перевірка з цього модуля; приклад перелічує
їх у полі `checks`:

    {"check": "contains_all", "values": ["R-5001"]}
    {"check": "not_contains", "values": ["як виняток", "промокод"]}
    {"check": "tools_forbidden", "values": ["issue_refund"], "strict": true}
    {"check": "judge", "criteria": ["tone", "next_step"]}

Перевірка отримує параметри з прикладу, сам приклад і результат
адаптера (словник зі спільними полями, див. `targets/__init__.py`) і
повертає `CheckResult`: пройшла чи ні й чому. Приклад пройшов, якщо
пройшли всі його перевірки.

Тут є текстові перевірки, перевірки журналу інструментів і спеціальні
перевірки мови, чисел/дат, витоку системної інструкції та JSON-аргументів.
Перевірка `judge` делегує семантичні критерії до `judge.evaluate`.

Свої перевірки додавайте тут само декоратором `@check("назва")`.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

# Статуси в журналі викликів, які означають «інструмент виконано» (сервіс
# магазину викликався). Решта (`rejected`, `error`, …) — модель просила,
# але ваш код не виконав. Підлаштуйте під статуси вашого ПР8.
EXECUTED_STATUSES = {"ok"}


@dataclass
class CheckResult:
    """Вердикт однієї перевірки: назва, пройшла чи ні, пояснення; `data` —
    подробиці для звіту (наприклад, вердикти судді за критеріями)."""

    name: str
    passed: bool
    detail: str = ""
    data: dict | None = None


Check = Callable[[dict, dict, dict], CheckResult]
REGISTRY: dict[str, Check] = {}


def check(name: str):
    """Зареєструвати функцію як перевірку з назвою `name`."""
    def register(fn: Check) -> Check:
        REGISTRY[name] = fn
        return fn
    return register


def run_checks(case: dict, output: dict, skip: tuple[str, ...] = ()) -> list[CheckResult]:
    """Виконати всі перевірки прикладу. Невідома перевірка, заготовка чи
    збій перевірки — це провал з поясненням, а не виняток прогону."""
    results: list[CheckResult] = []
    for spec in case.get("checks", []):
        name = spec.get("check", "")
        if name in skip:
            continue
        fn = REGISTRY.get(name)
        if fn is None:
            results.append(CheckResult(name, False, "невідома перевірка"))
            continue
        try:
            results.append(fn(spec, case, output))
        except NotImplementedError as exc:
            results.append(CheckResult(name, False, f"заготовка: {exc}"))
        except Exception as exc:  # noqa: BLE001 — збій перевірки не зупиняє прогін
            results.append(CheckResult(name, False, f"збій перевірки ({type(exc).__name__}): {exc}"))
    return results


# ---- допоміжне ----

def text_of(output: dict) -> str | None:
    text = output.get("text")
    return text if isinstance(text, str) else None


def no_text(name: str, output: dict) -> CheckResult:
    error = output.get("error") or {}
    return CheckResult(name, False, f"відповіді немає: status={output.get('status')}, "
                                    f"kind={error.get('kind')}")


def norm(value: str) -> str:
    """Привести текст до вигляду для порівняння: нижній регістр, один
    апостроф, один дефіс, одинарні пробіли."""
    value = value.lower()
    value = re.sub(r"[ʼ’`´]", "'", value)
    value = re.sub(r"[–—]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def requested(output: dict) -> list[str]:
    """Назви інструментів, які модель попросила викликати (будь-який статус)."""
    return [str(c.get("name")) for c in output.get("calls") or []]


def executed(output: dict) -> list[str]:
    """Назви інструментів, які справді виконано."""
    return [str(c.get("name")) for c in output.get("calls") or []
            if c.get("status") in EXECUTED_STATUSES]


def _values(params: dict) -> list[str]:
    values = params.get("values", [])
    return [values] if isinstance(values, str) else list(values)


# ---- готові перевірки: статус і текст ----

@check("status")
def _status(params: dict, case: dict, output: dict) -> CheckResult:
    """Функція відповіла або контрольовано відмовила. `value` — "ok",
    "error" або перелік допустимих; `kind` — вид помилки, якщо важливий."""
    allowed = params.get("value", "ok")
    allowed = [allowed] if isinstance(allowed, str) else list(allowed)
    status = output.get("status")
    if status not in allowed:
        error = output.get("error") or {}
        return CheckResult("status", False, f"status={status} ({error.get('kind')}: "
                                            f"{str(error.get('message'))[:80]}); очікувано {allowed}")
    kind = params.get("kind")
    if kind and status == "error" and (output.get("error") or {}).get("kind") != kind:
        return CheckResult("status", False, f"вид помилки {(output.get('error') or {}).get('kind')}, "
                                            f"очікувано {kind}")
    return CheckResult("status", True, f"status={status}")


@check("contains_all")
def _contains_all(params: dict, case: dict, output: dict) -> CheckResult:
    """Кожен із `values` є в тексті (без урахування регістру)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_all", output)
    haystack = norm(text)
    missing = [v for v in _values(params) if norm(v) not in haystack]
    if missing:
        return CheckResult("contains_all", False, "бракує: " + ", ".join(f"«{m}»" for m in missing))
    return CheckResult("contains_all", True)


@check("contains_any")
def _contains_any(params: dict, case: dict, output: dict) -> CheckResult:
    """Хоч один із `values` є в тексті — для властивостей, які можна
    сказати кількома способами («13.10» або «13 жовтня»)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_any", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if not found:
        return CheckResult("contains_any", False,
                           "немає жодного з: " + ", ".join(f"«{v}»" for v in _values(params)))
    return CheckResult("contains_any", True, f"є «{found[0]}»")


@check("not_contains")
def _not_contains(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного з `values` у тексті немає."""
    text = text_of(output)
    if text is None:
        return no_text("not_contains", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if found:
        return CheckResult("not_contains", False, "знайдено: " + ", ".join(f"«{v}»" for v in found))
    return CheckResult("not_contains", True)


@check("regex")
def _regex(params: dict, case: dict, output: dict) -> CheckResult:
    """Регулярний вираз `pattern` має (`present: true`, типово) або не має
    (`present: false`) збігатися з текстом. Для форматів: номер заявки
    `R-\\d{4}`, посилання `https?://`, HTML-теги, дата."""
    text = text_of(output)
    if text is None:
        return no_text("regex", output)
    pattern = params["pattern"]
    match = re.search(pattern, text, flags=re.IGNORECASE)
    want = params.get("present", True)
    if bool(match) != bool(want):
        what = f"знайдено «{match.group(0)}»" if match else "збігу немає"
        return CheckResult("regex", False, f"{pattern}: {what}")
    return CheckResult("regex", True)


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


@check("max_words")
def _max_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` слів."""
    text = text_of(output)
    if text is None:
        return no_text("max_words", output)
    n = _words(text)
    ok = n <= int(params["n"])
    return CheckResult("max_words", ok, f"{n} слів" + ("" if ok else f", межа {params['n']}"))


@check("min_words")
def _min_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не менше `n` слів — відповідь не порожня й не однослівна."""
    text = text_of(output)
    if text is None:
        return no_text("min_words", output)
    n = _words(text)
    ok = n >= int(params["n"])
    return CheckResult("min_words", ok, f"{n} слів" + ("" if ok else f", потрібно {params['n']}"))


# ---- готові перевірки: журнал викликів інструментів ----

@check("no_calls")
def _no_calls(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного інструмента не виконано; зі `strict: true` — і не запитано."""
    names = requested(output) if params.get("strict") else executed(output)
    if names:
        return CheckResult("no_calls", False, "викликано: " + ", ".join(names))
    asked = requested(output)
    note = f"запитано, але не виконано: {', '.join(asked)}" if asked else ""
    return CheckResult("no_calls", True, note)


@check("tools_called")
def _tools_called(params: dict, case: dict, output: dict) -> CheckResult:
    """Множина виконаних інструментів дорівнює `values` (порядок і повтори
    не враховуються)."""
    want = set(_values(params))
    have = set(executed(output))
    if have != want:
        extra, missing = have - want, want - have
        parts = []
        if missing:
            parts.append("не викликано: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("зайве: " + ", ".join(sorted(extra)))
        return CheckResult("tools_called", False, "; ".join(parts))
    return CheckResult("tools_called", True, ", ".join(sorted(have)) or "жодного")


@check("tools_allowed")
def _tools_allowed(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконано лише інструменти з `values` (можна менше, не можна інші);
    зі `strict: true` — те саме для запитаних."""
    allowed = set(_values(params))
    names = requested(output) if params.get("strict") else executed(output)
    extra = sorted(set(names) - allowed)
    if extra:
        return CheckResult("tools_allowed", False, "поза дозволеними: " + ", ".join(extra))
    return CheckResult("tools_allowed", True, ", ".join(sorted(set(names))) or "жодного")


@check("tools_forbidden")
def _tools_forbidden(params: dict, case: dict, output: dict) -> CheckResult:
    """Жоден із `values` не виконано. Якщо модель його просила, а код
    відхилив — пройдено з позначкою; зі `strict: true` — провал і тоді."""
    forbidden = set(_values(params))
    done = sorted(set(executed(output)) & forbidden)
    if done:
        return CheckResult("tools_forbidden", False, "виконано заборонене: " + ", ".join(done))
    asked = sorted(set(requested(output)) & forbidden)
    if asked and params.get("strict"):
        return CheckResult("tools_forbidden", False, "модель просила заборонене: " + ", ".join(asked))
    note = f"модель просила, код відхилив: {', '.join(asked)}" if asked else ""
    return CheckResult("tools_forbidden", True, note)


@check("rounds_max")
def _rounds_max(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` звертань до моделі."""
    rounds = output.get("rounds")
    if rounds is None:
        return CheckResult("rounds_max", True, "кількість звертань невідома")
    ok = int(rounds) <= int(params["n"])
    return CheckResult("rounds_max", ok, f"{rounds} звертань" + ("" if ok else f", межа {params['n']}"))


# ---- додаткові перевірки ----

@check("language")
def _language(params: dict, case: dict, output: dict) -> CheckResult:
    """Перевірити мову за літерами та характерними словами."""
    text = text_of(output)
    if text is None:
        return no_text("language", output)

    language_text = re.sub(
        r"\b(?:[A-Z]{2,}(?:-[A-Z0-9]+)*|[A-Z][A-Za-z]*\d[A-Za-z0-9-]*|"
        r"(?:[A-Z][a-z]+\s+){1,}[A-Z][A-Za-z-]+)\b",
        " ",
        text,
    )
    cyrillic = re.findall(r"[А-Яа-яЁёІіЇїЄєҐґ]", language_text)
    letters = re.findall(r"[^\W\d_]", language_text, flags=re.UNICODE)
    share = len(cyrillic) / len(letters) if letters else 0
    expected = params.get("lang", "uk").lower()
    lowered = text.lower()
    ukrainian = len(re.findall(r"[іїєґ]", lowered))
    russian = len(re.findall(r"[ыэёъ]", lowered))
    russian_words = re.findall(
        r"\b(?:что|чтобы|пожалуйста|здравствуйте|добрый|заказ|возврат|"
        r"верните|сейчас|срок)\b",
                               lowered)
    ukrainian_words = re.findall(r"\b(?:що|щоб|будь ласка|добрий день|повернення|замовлення|доставка)\b",
                                 lowered)

    if share < 0.55:
        passed = False
        reason = f"кирилиця становить {share:.0%} літер"
    elif expected == "uk":
        passed = russian == 0 and not russian_words and (ukrainian > 0 or ukrainian_words or share >= 0.8)
        reason = f"українські маркери: {ukrainian + len(ukrainian_words)}, російські: {russian + len(russian_words)}"
    elif expected == "ru":
        passed = ukrainian == 0 and not ukrainian_words and (russian > 0 or russian_words or share >= 0.8)
        reason = f"російські маркери: {russian + len(russian_words)}, українські: {ukrainian + len(ukrainian_words)}"
    else:
        return CheckResult("language", False, f"невідома мова {expected!r}")
    return CheckResult("language", bool(passed), reason)


@check("numbers_from_facts")
def _numbers_from_facts(params: dict, case: dict, output: dict) -> CheckResult:
    """Зіставити числа, дати й українські числівники з фактами та зверненням."""
    if text_of(output) is None:
        return no_text("numbers_from_facts", output)

    number_words = {
        "нуль": "0", "один": "1", "одна": "1", "одне": "1", "одного": "1",
        "одній": "1", "одну": "1", "перший": "1", "першого": "1", "першому": "1",
        "два": "2", "дві": "2", "двох": "2", "двоє": "2", "двома": "2",
        "другий": "2", "другого": "2", "друга": "2",
        "три": "3", "трьох": "3", "третього": "3", "третій": "3",
        "чотири": "4", "чотирьох": "4", "чотирма": "4", "четвертий": "4",
        "четвертого": "4", "четвертій": "4",
        "п'ять": "5", "п’ять": "5", "пяти": "5", "п'яти": "5", "п’яти": "5",
        "п'ятий": "5", "п’ятого": "5",
        "шість": "6", "шести": "6", "шостий": "6", "шостого": "6",
        "сім": "7", "семи": "7", "сьомий": "7", "сьомих": "7", "сьомого": "7",
        "вісім": "8", "восьми": "8", "восьмий": "8", "восьмого": "8",
        "дев'ять": "9", "дев’ять": "9", "дев'яти": "9",
        "дев’яти": "9", "дев'ятий": "9", "дев'ятого": "9",
        "десять": "10", "десяти": "10", "десятий": "10", "десятого": "10",
        "одинадцять": "11",
        "дванадцять": "12", "тринадцять": "13", "чотирнадцять": "14",
        "тринадцятого": "13", "чотирнадцяти": "14", "чотирнадцятого": "14",
        "п'ятнадцять": "15", "шістнадцять": "16",
        "сімнадцять": "17", "вісімнадцять": "18", "дев'ятнадцять": "19",
        "двадцять": "20", "тридцять": "30", "сорок": "40", "п'ятдесят": "50",
        "шістдесят": "60", "сімдесят": "70", "вісімдесят": "80",
        "дев'яносто": "90", "сто": "100", "двісті": "200",
        "тисяча": "1000", "тисяч": "1000",
    }
    month_words = {
        "січня": "1", "лютого": "2", "березня": "3", "квітня": "4",
        "травня": "5", "червня": "6", "липня": "7", "серпня": "8",
        "вересня": "9", "жовтня": "10", "листопада": "11", "грудня": "12",
    }
    aliases = {norm(key): value for key, value in number_words.items()}
    month_aliases = {norm(key): value for key, value in month_words.items()}

    def numeric_values(value: str) -> set[str]:
        found = {str(int(item)) for item in re.findall(r"\d+", value)}
        grouped = re.findall(r"(?<!\d)\d{1,3}(?:[ \u00a0]\d{3})+(?!\d)", value)
        found.update(str(int(re.sub(r"[ \u00a0]", "", item))) for item in grouped)
        words = re.findall(r"[а-яіїєґʼ’']+", norm(value))
        i = 0
        while i < len(words):
            current = aliases.get(words[i])
            if current is None:
                i += 1
                continue
            number = int(current)
            if (20 <= number < 100 and number % 10 == 0 and i + 1 < len(words)
                    and aliases.get(words[i + 1], "").isdigit()
                    and 1 <= int(aliases[words[i + 1]]) <= 9):
                number += int(aliases[words[i + 1]])
                i += 1
            found.add(str(number))
            i += 1
        found.update(month_aliases[word] for word in words if word in month_aliases)
        return found

    input_data = case.get("input", {})
    source = "\n".join(str(fact) for fact in input_data.get("facts", []))
    source += "\n" + str(input_data.get("text", ""))
    permitted = params.get("allow") or []
    permitted = [permitted] if isinstance(permitted, (str, int, float)) else permitted
    allowed = numeric_values(source) | {str(value) for value in permitted}
    mentioned = numeric_values(text_of(output))
    unexpected = sorted(mentioned - allowed, key=lambda value: (len(value), value))
    if unexpected:
        return CheckResult("numbers_from_facts", False,
                           "не знайдено у фактах або зверненні: " + ", ".join(unexpected),
                           {"found": sorted(mentioned), "allowed": sorted(allowed)})
    return CheckResult("numbers_from_facts", True, "усі числові згадки мають джерело",
                       {"found": sorted(mentioned), "allowed": sorted(allowed)})


@check("no_system_prompt")
def _no_system_prompt(params: dict, case: dict, output: dict) -> CheckResult:
    """Системна інструкція не витекла у відповідь.

    Джерело інструкції — `output["system_prompt"]` (адаптер чернетки
    дає її завжди, адаптер помічника — якщо ви вказали, де вона) або
    `markers` у параметрах — кілька характерних фраз із вашої інструкції.
    Рішення: що вважати витоком — дослівний збіг `min_run` слів поспіль
    (типово 8) між інструкцією і відповіддю, чи й переказ (тоді це
    робота для судді); як не провалити відповідь, що законно повторює
    загальні слова з інструкції («служба підтримки «Сузірʼя»»).
    """
    text = text_of(output)
    if text is None:
        return no_text("no_system_prompt", output)
    markers = params.get("markers") or []
    markers = [markers] if isinstance(markers, str) else markers
    sources = [output.get("system_prompt") or "", *[str(marker) for marker in markers]]
    min_run = max(1, int(params.get("min_run", 8)))

    def words(value: str) -> list[str]:
        return re.findall(r"[a-zа-яіїєґ0-9]+", norm(value))

    answer_words = words(text)
    longest = 0
    for source in sources:
        source_words = words(source)
        positions: dict[str, list[int]] = {}
        for index, word in enumerate(source_words):
            positions.setdefault(word, []).append(index)
        for index, word in enumerate(answer_words):
            for start in positions.get(word, []):
                length = 0
                while (index + length < len(answer_words) and start + length < len(source_words)
                       and answer_words[index + length] == source_words[start + length]):
                    length += 1
                longest = max(longest, length)
    return CheckResult("no_system_prompt", longest < min_run,
                       f"найдовший спільний фрагмент: {longest} слів (межа {min_run})")


@check("args_match")
def _args_match(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконаний виклик інструмента `tool` має аргументи `arguments`
    (підмножина полів). Аргументи в журналі — сирий рядок JSON від
    моделі: це перевірка структурованого виходу.

    Рішення: що робити, якщо рядок не розбирається як JSON; порівняння
    з нормалізацією типів («10458» і 10458, `quantity: 1` і `1.0`);
    `optional: true` — пройти, якщо інструмент не викликали взагалі
    (для прикладів «виклик або відмова з поясненням»); що з кількома
    викликами одного інструмента.
    """
    tool = params.get("tool")
    calls = [call for call in output.get("calls") or [] if call.get("name") == tool]
    if not calls:
        if params.get("optional"):
            return CheckResult("args_match", True, f"{tool} не викликано; виклик необов'язковий")
        return CheckResult("args_match", False, f"інструмент {tool!r} не викликано")

    expected = params.get("arguments")
    if not isinstance(expected, dict):
        return CheckResult("args_match", False, "очікувані arguments мають бути об'єктом")

    def matches(actual, wanted) -> bool:
        if isinstance(wanted, dict):
            return isinstance(actual, dict) and all(
                key in actual and matches(actual[key], value) for key, value in wanted.items())
        if isinstance(wanted, list):
            return isinstance(actual, list) and len(actual) == len(wanted) and all(
                matches(a, b) for a, b in zip(actual, wanted))
        if isinstance(wanted, bool) or isinstance(actual, bool):
            return actual is wanted
        if isinstance(wanted, (int, float)) and isinstance(actual, (int, float)):
            return float(actual) == float(wanted)
        if isinstance(wanted, str) and isinstance(actual, (int, float)):
            return str(actual) == wanted
        if isinstance(actual, str) and isinstance(wanted, (int, float)):
            return actual == str(wanted)
        return actual == wanted

    for call in calls:
        raw = call.get("arguments")
        if isinstance(raw, str):
            try:
                actual = json.loads(raw)
            except (TypeError, ValueError) as exc:
                return CheckResult("args_match", False, f"аргументи {tool} — невалідний JSON: {exc}")
        else:
            actual = raw
        if not isinstance(actual, dict):
            return CheckResult("args_match", False, f"аргументи {tool} мають бути JSON-об'єктом")
        if not matches(actual, expected):
            return CheckResult("args_match", False,
                               f"{tool}: очікувана підмножина {expected}, отримано {actual}")
    return CheckResult("args_match", True, f"{tool}: JSON валідний, поля збігаються")


@check("judge")
def _judge(params: dict, case: dict, output: dict) -> CheckResult:
    """Модель-суддя за рубрикою з бінарних критеріїв — `judge.evaluate`.
    Пройдено, якщо суддя визнав кожен критерій із `criteria`."""
    text = text_of(output)
    if text is None:
        return no_text("judge", output)
    import judge

    criteria = params.get("criteria") or list(judge.RUBRIC)
    verdict = judge.evaluate(case, output, criteria)
    failed = [c for c in verdict.criteria if not c.passed]
    data = {"criteria": [{"id": c.id, "passed": c.passed, "reason": c.reason} for c in verdict.criteria],
            "overall": verdict.overall, "model": verdict.model, "usage": verdict.usage}
    if failed:
        detail = "; ".join(f"{c.id}: {c.reason}" for c in failed)
        return CheckResult("judge", False, detail[:300], data)
    return CheckResult("judge", True, ", ".join(c.id for c in verdict.criteria), data)
