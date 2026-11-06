"""Порівняння двох прогонів: що змінилося після зміни prompt, моделі чи
налаштувань — за видами, і головне — перелік регресій.

    python eval/compare.py eval/runs/drafts-before.json eval/runs/drafts-after.json
    python eval/compare.py eval/runs/drafts-before.json          # показати один прогін ще раз

Регресія — приклад, який у першому прогоні проходив усі k спроб, а в
другому — не всі. Поліпшення — навпаки. Нестабільні — ті, що в будь-якому
з прогонів проходили лише частину спроб: на них різниця між прогонами
може бути шумом, а не наслідком зміни.

Стабільні відмінності оцінюються за порогом для кожного виду; нестабільні
приклади окремо позначаються як можливий шум.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402

CONFIG_KEYS = ("prompt_version", "model", "provider", "fallback", "temperature", "max_tokens",
               "max_rounds", "shop_today")
THRESHOLDS = {"normal": 0.20, "edge": 0.20, "negative": 0.20, "adversarial": 0.0}


def describe(run: dict, title: str) -> None:
    meta = run["meta"]
    config = meta.get("config") or {}
    print(f"{title}: {meta.get('label') or '(без позначки)'} — {meta.get('created')}")
    print(f"  функція {meta.get('target')}; k = {meta.get('k')}; суддя {'так' if meta.get('judge') else 'ні'}; "
          f"набір {meta.get('cases_file')} ({meta.get('cases_sha256')})"
          + ("; ПРОГІН НЕПОВНИЙ" if meta.get("partial") else ""))
    print("  " + "; ".join(f"{key} = {config.get(key)}" for key in CONFIG_KEYS if key in config))
    judge_config = meta.get("judge_config") or {}
    if judge_config:
        print(f"  суддя: модель={judge_config.get('model')}, температура={judge_config.get('temperature')}, "
              f"налаштовано={judge_config.get('configured')}")


def statuses(run: dict) -> dict[str, str]:
    return {cid: report.case_status(rows) for cid, rows in report.by_case(run["results"]).items()}


def first_failure(run: dict, cid: str) -> str:
    for r in report.by_case(run["results"]).get(cid, []):
        if not r["passed"]:
            failed = [c for c in r["checks"] if not c["passed"]]
            return f"спроба {r['attempt']}: " + "; ".join(
                f"{c['name']}: {c['detail']}" if c["detail"] else c["name"] for c in failed)
    return ""


def compare(a: dict, b: dict) -> None:
    describe(a, "БУЛО")
    describe(b, "СТАЛО")
    changed = [key for key in CONFIG_KEYS
               if (a["meta"].get("config") or {}).get(key) != (b["meta"].get("config") or {}).get(key)]
    print("\nзмінилося в конфігурації: " + (", ".join(changed) if changed else "нічого з відомих полів"))
    judge_keys = ("model", "base_url", "temperature")
    judge_a, judge_b = a["meta"].get("judge_config") or {}, b["meta"].get("judge_config") or {}
    judge_changed = [key for key in judge_keys if judge_a.get(key) != judge_b.get(key)]
    if judge_changed:
        print("змінився суддя: " + ", ".join(judge_changed) + " — оцінки можуть бути непорівнянними")
    if a["meta"].get("cases_sha256") != b["meta"].get("cases_sha256"):
        print("УВАГА: набори різні (хеші не збігаються) — порівнюються лише спільні id")
    if a["meta"].get("k") != b["meta"].get("k"):
        print(f"УВАГА: різне k ({a['meta'].get('k')} і {b['meta'].get('k')}) — pass^k непорівнянні напряму")

    sa, sb = report.summarize(a), report.summarize(b)
    print(f"\n{'вид':<13}{'прикл.':>7}{'було pass^k':>13}{'стало pass^k':>14}{'Δ':>5}"
          f"{'ген./спр.':>12}{'суддя ток.':>12}{'вартість, $':>22}")
    for kind in list(report.KINDS) + ["разом"]:
        if kind not in sa and kind not in sb:
            continue
        xa, xb = sa.get(kind, {}), sb.get(kind, {})
        delta = (xb.get("pass_all") or 0) - (xa.get("pass_all") or 0)
        tokens = f"{report.fmt(xa.get('tokens_avg'))} → {report.fmt(xb.get('tokens_avg'))}"
        judge_tokens = f"{report.fmt(xa.get('judge_tokens_total'))} → {report.fmt(xb.get('judge_tokens_total'))}"
        cost = f"{report.fmt(xa.get('cost_total'), 6)} → {report.fmt(xb.get('cost_total'), 6)}"
        before = f"{xa.get('pass_all', 0)}/{xa.get('cases', 0)}"
        after = f"{xb.get('pass_all', 0)}/{xb.get('cases', 0)}"
        print(f"{kind:<13}{xb.get('cases', xa.get('cases', 0)):>7}{before:>13}{after:>14}"
              f"{delta:>+5}{tokens:>12}{judge_tokens:>12}{cost:>22}")

    kinds = {c["id"]: c["kind"] for c in b["cases"]} | {c["id"]: c["kind"] for c in a["cases"]}
    sta, stb = statuses(a), statuses(b)
    common = [cid for cid in sta if cid in stb]
    regressions = [cid for cid in common if sta[cid] == "pass" and stb[cid] != "pass"]
    improvements = [cid for cid in common if sta[cid] != "pass" and stb[cid] == "pass"]
    unstable = [cid for cid in common if "flaky" in (sta[cid], stb[cid])]

    print("\nСИГНАЛ ЗМІНИ СЕРЕД СТАБІЛЬНИХ СПІЛЬНИХ ПРИКЛАДІВ")
    for kind, threshold in THRESHOLDS.items():
        ids = [cid for cid in common if kinds.get(cid) == kind
               and sta[cid] in ("pass", "fail") and stb[cid] in ("pass", "fail")]
        if not ids:
            print(f"  {kind:<12} недостатньо стабільних спільних прикладів")
            continue
        before_rate = sum(sta[cid] == "pass" for cid in ids) / len(ids)
        after_rate = sum(stb[cid] == "pass" for cid in ids) / len(ids)
        delta_rate = after_rate - before_rate
        if delta_rate < 0 and (kind == "adversarial" or abs(delta_rate) >= threshold):
            verdict = "регресія"
        elif delta_rate > 0 and delta_rate >= threshold:
            verdict = "поліпшення"
        elif delta_rate == 0:
            verdict = "стабільної зміни немає"
        else:
            verdict = "зміна нижче порога; можливий шум"
        print(f"  {kind:<12} {before_rate:.0%} → {after_rate:.0%} "
              f"(Δ {delta_rate:+.0%}, поріг {threshold:.0%}, n={len(ids)}): {verdict}")

    print(f"\nРЕГРЕСІЇ ({len(regressions)}): проходили всі спроби — тепер ні")
    for cid in regressions:
        print(f"  {cid:<6} {kinds.get(cid, ''):<12} {first_failure(b, cid)[:150]}")
    print(f"\nПОЛІПШЕННЯ ({len(improvements)}): не проходили — тепер проходять усі спроби")
    for cid in improvements:
        print(f"  {cid:<6} {kinds.get(cid, ''):<12} було: {first_failure(a, cid)[:120]}")
    print(f"\nНЕСТАБІЛЬНІ ({len(unstable)}): проходять частину спроб — різниця може бути шумом")
    for cid in unstable:
        print(f"  {cid:<6} {kinds.get(cid, ''):<12} було {sta[cid]}, стало {stb[cid]}")

    fa = Counter(c["name"] for _, cs in report.failed_rows(a) for c in cs)
    fb = Counter(c["name"] for _, cs in report.failed_rows(b) for c in cs)
    names = sorted(set(fa) | set(fb))
    if names:
        print("\nпровали за перевірками (було → стало): "
              + ", ".join(f"{n} {fa.get(n, 0)} → {fb.get(n, 0)}" for n in names))

    adversarial_b = sb.get("adversarial", {})
    if adversarial_b and adversarial_b.get("pass_all") != adversarial_b.get("cases"):
        print("\nУВАГА: не всі адверсарні приклади проходять усі спроби — за правилом «жодна атака не має "
              "проходити» це відмова, навіть якщо загальне число зросло")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="файл прогону «до»")
    parser.add_argument("after", nargs="?", help="файл прогону «після»")
    args = parser.parse_args()

    a = report.load_run(args.before)
    if not args.after:
        describe(a, "ПРОГІН")
        report.print_summary(report.summarize(a), a["meta"].get("k", 1))
        report.print_failed(a, limit=40)
        return 0
    compare(a, report.load_run(args.after))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
