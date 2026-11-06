"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:

    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow

Читання файлу, відбір за датою і допоміжні функції готові. Розділи
звіту — заготовки: поля, за якими вони працюють, — ваші, бо назви
подій і полів ви обрали, коли писали записи в шлюзі й фонових задачах.
`summary` працює на будь-якому журналі й показує, що в ньому взагалі є.

Орієнтир для кожного розділу — питання, на яке він відповідає; воно в
докстрінгу. Відповідь — числа й кілька рядків журналу, а не переказ.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from statistics import median
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines. Рядки, які не розбираються, рахуються й
    пропускаються; `since` — дата або час у форматі ISO, від якого брати
    записи (порівнюється з полем `ts` як рядок)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для
    порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    """Згрупувати записи за значенням поля."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі: скільки записів, за який час, яких подій і
    рівнів скільки. Працює на будь-якому журналі."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    """Які виклики найповільніші й що таке «повільно» в цих числах?

    Розподіл тривалості звертань до провайдера (медіана, p95, максимум),
    а не середнє; десятка найдовших із ідентифікаторами, провайдером,
    моделлю, спробою; де пройшов час — у черзі, до першого фрагмента чи
    в генерації; розріз за провайдером і годиною.
    """
    calls = [r for r in records if r.get("event") == "llm.attempt"
             and isinstance(r.get("duration_ms"), (int, float))]
    if not calls:
        print("викликів моделі немає")
        return
    durations = [float(r["duration_ms"]) for r in calls]
    print(f"виклики: {len(calls)}; медіана {median(durations):.1f} ms; "
          f"p95 {percentile(durations, .95):.1f} ms; максимум {max(durations):.1f} ms")
    print("час за етапами (медіана / p95, ms):")
    for field in ("queue_ms", "first_token_ms", "generation_ms"):
        values = [float(r[field]) for r in calls if isinstance(r.get(field), (int, float))]
        print(f"  {field}: {median(values):.1f} / {percentile(values, .95):.1f}" if values
              else f"  {field}: даних немає")
    print("найповільніші виклики:")
    for row in sorted(calls, key=lambda r: r["duration_ms"], reverse=True)[:10]:
        print(f"  {row.get('duration_ms', 0):>8.1f} ms draft={row.get('draft_id')} "
              f"provider={row.get('provider')} model={row.get('model')} attempt={row.get('attempt')} "
              f"queue={row.get('queue_ms')} first={row.get('first_token_ms')} "
              f"generation={row.get('generation_ms')}")


def failed(records: list[dict]) -> None:
    """Що й чому не вдалося?

    Невдалі спроби за видом збою; остаточні збої чернеток; частка
    чернеток із невдалою першою спробою, які все ж завершилися успіхом
    (чи окупилися повтори); вартість спроб, які нічого не дали.
    """
    calls = [r for r in records if r.get("event") == "llm.attempt"]
    failures = [r for r in calls if r.get("status") in {"failed", "rejected"}]
    if not calls:
        print("викликів моделі немає")
        return
    kinds = Counter(r.get("kind") or "unknown" for r in failures)
    print(f"невдалі/відхилені спроби: {len(failures)} з {len(calls)}")
    print("за видом збою:", dict(kinds))
    drafts: dict[str, list[dict]] = defaultdict(list)
    for row in calls:
        drafts[str(row.get("draft_id") or row.get("request_id"))].append(row)
    retried = 0
    rescued_second = 0
    rescued_later = 0
    final_failed = 0
    for attempts in drafts.values():
        attempts.sort(key=lambda r: r.get("attempt", 0))
        if len(attempts) > 1:
            retried += 1
        if attempts and attempts[0].get("status") in {"failed", "rejected"}:
            if len(attempts) > 1 and attempts[1].get("status") == "success":
                rescued_second += 1
            elif any(r.get("status") == "success" for r in attempts[2:]):
                rescued_later += 1
        if attempts and attempts[-1].get("status") != "success":
            final_failed += 1
    print(f"чернеток із повторами: {retried}; урятувала друга спроба: {rescued_second}; "
          f"урятували пізніші спроби: {rescued_later}; "
          f"остаточно невдалих: {final_failed}")
    failed_cost = sum(float(r.get("cost") or 0) for r in failures)
    print(f"вартість невдалих/відхилених спроб: ${failed_cost:.6f}")


def repeated(records: list[dict]) -> None:
    """Що ми робили двічі?

    Той самий ключ кешу кілька разів за короткий час (кандидати на кеш
    і питання «чому»); чернетки з кількома спробами (повтори шлюзу);
    клієнти з серіями замовлень за хвилину.
    """
    calls = [r for r in records if r.get("event") == "llm.attempt"]
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        key = row.get("cache_key")
        if key:
            by_key[str(key)].append(row)
    repeated_keys = [(key, rows) for key, rows in by_key.items() if len(rows) > 1]
    print(f"ключів кешу, що зустрічалися кілька разів: {len(repeated_keys)}")
    for key, rows in sorted(repeated_keys, key=lambda item: len(item[1]), reverse=True)[:10]:
        print(f"  {key[:12]}: {len(rows)} подій")
    by_draft: dict[str, list[dict]] = defaultdict(list)
    for row in calls:
        if row.get("draft_id"):
            by_draft[str(row["draft_id"])].append(row)
    retried = [(draft, rows) for draft, rows in by_draft.items() if len(rows) > 1]
    print(f"чернеток із кількома спробами: {len(retried)}")
    for draft, rows in sorted(retried, key=lambda item: len(item[1]), reverse=True)[:10]:
        print(f"  {draft}: {len(rows)} спроб")


def spend(records: list[dict]) -> None:
    """Скільки це коштувало?

    Вартість і токени за день, за модель, за клієнта; середня вартість
    чернетки; частка вартості, яка припала на спроби зі збоєм і на
    запасний варіант; скільки записів без вартості або з оцінкою.
    """
    calls = [r for r in records if r.get("event") == "llm.attempt"]
    if not calls:
        print("викликів моделі немає")
        return
    by_day: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    by_day_model: dict[tuple[str, str], float] = defaultdict(float)
    estimated = missing = 0
    for row in calls:
        cost = row.get("cost")
        if isinstance(cost, (int, float)):
            day = str(row.get("ts", ""))[:10]
            model = str(row.get("model") or "unknown")
            by_day[day] += float(cost)
            by_model[model] += float(cost)
            by_day_model[(day, model)] += float(cost)
        else:
            missing += 1
        estimated += bool(row.get("cost_estimated"))
    print("витрати за днем:")
    for day, amount in sorted(by_day.items()):
        print(f"  {day}: ${amount:.6f}")
    print("витрати за моделлю:")
    for model, amount in sorted(by_model.items(), key=lambda item: item[1], reverse=True):
        print(f"  {model}: ${amount:.6f}")
    print("витрати за днем і моделлю:")
    for (day, model), amount in sorted(by_day_model.items()):
        print(f"  {day} / {model}: ${amount:.6f}")
    print(f"спроб із оціненою вартістю: {estimated}; без вартості: {missing}")
    print(f"невдалі спроби включено; їхня сума: "
          f"${sum(float(r.get('cost') or 0) for r in calls if r.get('status') != 'success'):.6f}")


def cache_report(records: list[dict]) -> None:
    """Що дав кеш?

    Влучання, промахи, обходи (`fresh`) і частка влучань; скільки
    звертань до провайдера й грошей заощаджено; які звернення
    повторювалися найчастіше.
    """
    events = [r for r in records if r.get("event") in {
        "llm.cache_hit", "llm.cache_miss", "llm.cache_bypassed"}]
    counts = Counter(r["event"] for r in events)
    hits = counts["llm.cache_hit"]
    misses = counts["llm.cache_miss"]
    bypassed = counts["llm.cache_bypassed"]
    total = hits + misses
    print(f"влучань: {hits}; промахів: {misses}; обходів fresh: {bypassed}; "
          f"hit rate: {hits / total * 100:.1f} %" if total else
          f"влучань: {hits}; промахів: {misses}; обходів fresh: {bypassed}; hit rate: даних немає")
    keys: Counter = Counter(r.get("cache_key") for r in events if r.get("cache_key"))
    repeated_keys = [(key, count) for key, count in keys.items() if count > 1]
    print(f"повторних ключів: {len(repeated_keys)}")
    for key, count in sorted(repeated_keys, key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {str(key)[:12]}: {count} звернень")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        SECTIONS[name](records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
