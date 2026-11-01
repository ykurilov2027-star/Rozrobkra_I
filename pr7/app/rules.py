"""Програмні правила: перевірка змісту вилучених даних.

Схема (`app/schema.py`) каже, що IBAN — рядок, а сума — число. Чи це
справжній IBAN і чи сходиться сума з позиціями, схема не знає. Це знає
документ: у рахунку багато надлишковості — контрольні розряди в кодах,
підсумки, що мають дорівнювати сумі рядків, ПДВ, що має дорівнювати
п'ятій частині. Модель, яка прочитала 8 замість 3, цю надлишковість
порушує, і код це бачить. Правила тут не знають ні про модель, ні про
веб-рівень: на вході — перевірений за схемою словник, на виході — перелік
знайдених проблем.

Набір правил задано (розділ 2 практичної роботи):

* IBAN: формат і контрольне число за ISO 13616 (залишок mod 97);
* код постачальника й покупця: 8 цифр ЄДРПОУ з контрольним розрядом
  або 10 цифр РНОКПП;
* арифметика: кількість × ціна = сума в кожному рядку; сума рядків =
  разом без ПДВ; ПДВ = 20 % (або нуль для неплатника); разом + ПДВ =
  до сплати — з допуском на округлення, який ви оберете й обґрунтуєте;
* дати: існують, не з майбутнього, строк дії не раніше дати рахунку;
* покупець — ваша компанія (`reference/company.json`);
* постачальник відомий, і IBAN збігається з довідником
  (`reference/suppliers.json`);
* обовʼязкові поля присутні.

Рішення — ваші: які правила дають помилку, а які — попередження; чи
перевіряти правило, якщо потрібного поля немає; як рахувати гроші, щоб
0,1 + 0,2 не дало 0,30000000000000004.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BI": 27, "BR": 29, "BY": 28, "CH": 21, "CR": 22,
    "CY": 28, "CZ": 24, "DE": 22, "DJ": 27, "DK": 18, "DO": 28, "EE": 20,
    "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22,
    "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22,
    "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "LY": 25,
    "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MN": 20, "MR": 27, "MT": 31,
    "MU": 30, "NL": 18, "NO": 15, "OM": 23, "PK": 24, "PL": 28, "PS": 29,
    "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SC": 31, "SD": 18,
    "SE": 24, "SI": 19, "SK": 24, "SM": 27, "SO": 23, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}
INVOICE_TYPES = {
    "рахунок",
    "рахунок на оплату",
    "рахунок-фактура",
    "рахунок фактура",
    "інвойс",
    "invoice",
}
NON_INVOICE_TYPES = {
    "видаткова накладна",
    "накладна",
    "акт",
    "договір",
    "квитанція",
    "чек",
    "товарний чек",
    "платіжне доручення",
}


class ReferenceDataError(Exception):
    """Довідники компанії чи постачальників не вдалося прочитати."""


@dataclass
class Issue:
    """Проблема, знайдена правилом.

    `field` — шлях до поля, наприклад `supplier.iban` чи `items.1.amount`
    (другий рядок: індекс — з нуля); сторінка підсвічує поле з таким
    шляхом. `rule` — коротка назва
    правила. `message` — пояснення для людини, яка перевірятиме документ.
    `severity` — `error` або `warning`; що з цього випливає для рішення,
    вирішує `app/extraction.py`.
    """

    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити своєї компанії й довідник постачальників."""
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


def classify_document_type(value: object) -> str:
    document_type = unicodedata.normalize("NFKC", str(value or "")).casefold()
    document_type = re.sub(r"[\s\u00a0]+", " ", document_type).strip()
    document_type = re.sub(r"\s*[-‐‑‒–—]\s*", "-", document_type)
    if document_type in INVOICE_TYPES:
        return "invoice"
    if document_type in NON_INVOICE_TYPES:
        return "other"
    return "unknown"


def is_invoice_document_type(value: object) -> bool:
    return classify_document_type(value) == "invoice"


def check(document: dict) -> list[Issue]:
    """Застосувати правила до вилученого документа.

    Порожній перелік означає, що правила нічого не знайшли, — а не що
    документ прочитано правильно. Різниця між цими двома речами і є
    предметом перевірки на наборі зразків.
    """
    issues: list[Issue] = []

    def add(field: str, rule: str, message: str, severity: str = "error") -> None:
        issues.append(Issue(field, rule, message, severity))

    def blank(value: object) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    def amount(value: object) -> Decimal | None:
        if blank(value):
            return None
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None
        return result if result.is_finite() else None

    def parse_amount(field: str, value: object) -> Decimal | None:
        if blank(value):
            return None
        parsed = amount(value)
        if parsed is None:
            add(field, "amount_invalid", "Значення має бути коректним скінченним десятковим числом.")
        return parsed

    def normalize_code(value: object) -> str:
        return re.sub(r"\D", "", str(value or ""))

    def valid_edrpou(code: str) -> bool:
        if not re.fullmatch(r"\d{8}", code):
            return False
        digits = [int(char) for char in code]
        weights = (
            [1, 2, 3, 4, 5, 6, 7] if digits[0] <= 2 else
            [7, 1, 2, 3, 4, 5, 6] if digits[0] <= 4 else
            [3, 4, 5, 6, 7, 1, 2] if digits[0] <= 6 else
            [9, 1, 2, 3, 4, 5, 6]
        )
        check_digit = sum(a * b for a, b in zip(digits[:7], weights)) % 11
        if check_digit == 10:
            check_digit = sum(
                a * b for a, b in zip(digits[:7], [3, 4, 5, 6, 7, 8, 9])
            ) % 11
        return check_digit == digits[7]

    def valid_rnokpp(code: str) -> bool:
        if not re.fullmatch(r"\d{10}", code):
            return False
        digits = [int(char) for char in code]
        weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
        check_digit = sum(a * b for a, b in zip(digits[:9], weights)) % 11
        if check_digit == 10:
            check_digit = sum(
                a * b for a, b in zip(digits[:9], [3, 4, 5, 6, 8, 9, 10, 11, 12])
            ) % 11
        return check_digit == digits[9]

    def valid_iban(value: object) -> bool:
        iban = re.sub(r"\s+", "", str(value or "")).upper()
        country = iban[:2]
        if (
            not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban)
            or IBAN_LENGTHS.get(country) != len(iban)
        ):
            return False
        remainder = 0
        for char in iban[4:] + iban[:4]:
            expanded = str(ord(char) - 55) if char.isalpha() else char
            for digit in expanded:
                remainder = (remainder * 10 + int(digit)) % 97
        return remainder == 1

    def normalize_name(value: object) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold()
        text = text.translate(str.maketrans({
            "a": "а", "b": "в", "c": "с", "e": "е", "h": "н",
            "i": "і", "k": "к", "m": "м", "o": "о", "p": "р",
            "t": "т", "x": "х", "y": "у",
        }))
        text = re.sub(r"[^а-яіїєґ0-9]", "", text)
        return re.sub(r"^(тов|фоп|пп|ат)", "", text)

    def check_code(field: str, value: object) -> None:
        if blank(value):
            return
        code = normalize_code(value)
        valid = valid_edrpou(code) if len(code) == 8 else valid_rnokpp(code)
        if not valid:
            add(field, "code_checksum", "Контрольний розряд коду ЄДРПОУ/РНОКПП некоректний.")

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}
    required = (
        ("document_type", document.get("document_type")),
        ("number", document.get("number")),
        ("date", document.get("date")),
        ("supplier.name", supplier.get("name")),
        ("supplier.code", supplier.get("code")),
        ("supplier.iban", supplier.get("iban")),
        ("buyer.name", buyer.get("name")),
        ("buyer.code", buyer.get("code")),
        ("total_without_vat", document.get("total_without_vat")),
        ("vat", document.get("vat")),
        ("total", document.get("total")),
    )
    for field, value in required:
        if blank(value):
            add(field, "missing_required", f"Обов'язкове поле «{field}» відсутнє або не читається.")
    if not document.get("items"):
        add("items", "missing_required", "У рахунку не знайдено жодної позиції.")

    document_type_status = classify_document_type(document.get("document_type"))
    if document_type_status == "other":
        add("document_type", "document_type", "Документ не є рахунком на оплату.")
    elif document_type_status == "unknown":
        add("document_type", "document_type_unknown", "Тип документа відсутній або не вдалося його визначити.")

    invoice_date = None
    for field, raw_date in (("date", document.get("date")), ("valid_until", document.get("valid_until"))):
        if blank(raw_date):
            continue
        try:
            parsed = date.fromisoformat(str(raw_date).strip())
        except ValueError:
            add(field, "date_invalid", "Дата має бути коректною датою у форматі YYYY-MM-DD.")
            continue
        if field == "date" and parsed > date.today():
            add(field, "date_future", "Дата не може бути в майбутньому.")
        if field == "date":
            invoice_date = parsed
        elif invoice_date and parsed < invoice_date:
            add(field, "date_order", "Строк дії рахунку раніше за дату його складання.")

    check_code("supplier.code", supplier.get("code"))
    check_code("buyer.code", buyer.get("code"))
    if not blank(supplier.get("iban")) and not valid_iban(supplier["iban"]):
        add("supplier.iban", "iban_checksum", "IBAN має неправильний формат або контрольну суму.")

    try:
        company, suppliers = load_reference()
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceDataError(f"Не вдалося прочитати довідники reference/: {error}") from error

    buyer_code = normalize_code(buyer.get("code"))
    company_code = normalize_code(company.get("code"))
    if buyer_code and buyer_code != company_code:
        add("buyer.code", "buyer", "Код покупця не збігається з кодом вашої компанії.")
    elif buyer_code and normalize_name(buyer.get("name")) != normalize_name(company.get("name")):
        add("buyer.name", "buyer", "Назва покупця не збігається з довідником компанії.", "warning")

    supplier_code = normalize_code(supplier.get("code"))
    known_supplier = next(
        (entry for entry in suppliers if normalize_code(entry.get("code")) == supplier_code),
        None,
    )
    if not known_supplier:
        add("supplier.code", "supplier_registry", "Постачальника немає у довіднику.")
    else:
        if normalize_name(supplier.get("name")) != normalize_name(known_supplier.get("name")):
            add("supplier.name", "supplier_registry", "Назва постачальника не збігається з довідником.")
        expected_iban = re.sub(r"\s+", "", str(known_supplier.get("iban", ""))).upper()
        actual_iban = re.sub(r"\s+", "", str(supplier.get("iban") or "")).upper()
        if actual_iban and actual_iban != expected_iban:
            add("supplier.iban", "iban_registry", "IBAN не збігається з рахунком постачальника у довіднику.")

    tolerance = Decimal("0.01")
    line_amounts: list[Decimal] = []
    for index, item in enumerate(document.get("items") or []):
        for key in ("name", "unit", "quantity", "price", "amount"):
            if blank(item.get(key)):
                add(f"items.{index}.{key}", "missing_required", "Поле позиції відсутнє або не читається.")
        quantity = parse_amount(f"items.{index}.quantity", item.get("quantity"))
        price = parse_amount(f"items.{index}.price", item.get("price"))
        line_total = parse_amount(f"items.{index}.amount", item.get("amount"))
        if line_total is not None:
            line_amounts.append(line_total)
        if quantity is not None and price is not None and line_total is not None:
            try:
                calculated = (quantity * price).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP,
                )
            except (InvalidOperation, ValueError, TypeError):
                add(
                    f"items.{index}.amount",
                    "arithmetic_invalid",
                    "Не вдалося безпечно перевірити арифметику позиції.",
                )
                continue
            if abs(calculated - line_total) > tolerance:
                add(
                    f"items.{index}.amount",
                    "arithmetic_line",
                    f"Кількість × ціна дорівнює {calculated}, а вказано {line_total}.",
                )

    subtotal = parse_amount("total_without_vat", document.get("total_without_vat"))
    vat = parse_amount("vat", document.get("vat"))
    total = parse_amount("total", document.get("total"))
    if line_amounts and subtotal is not None:
        try:
            if abs(sum(line_amounts) - subtotal) > tolerance:
                add("total_without_vat", "arithmetic_subtotal", "Сума позицій не збігається з сумою без ПДВ.")
        except (InvalidOperation, ValueError, TypeError):
            add("total_without_vat", "arithmetic_invalid", "Не вдалося безпечно перевірити суму позицій.")
    if subtotal is not None and vat is not None and total is not None:
        try:
            if abs(subtotal + vat - total) > tolerance:
                add("total", "arithmetic_total", "Сума без ПДВ плюс ПДВ не збігається з підсумком.")
        except (InvalidOperation, ValueError, TypeError):
            add("total", "arithmetic_invalid", "Не вдалося безпечно перевірити загальну суму.")
    if known_supplier and subtotal is not None and vat is not None:
        try:
            expected_vat = (
                (subtotal * Decimal("0.20")).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP,
                )
                if known_supplier.get("vat_payer")
                else Decimal("0.00")
            )
            if abs(vat - expected_vat) > tolerance:
                add("vat", "vat_calculation", f"ПДВ має дорівнювати {expected_vat} за довідником постачальників.")
        except (InvalidOperation, ValueError, TypeError):
            add("vat", "arithmetic_invalid", "Не вдалося безпечно перевірити суму ПДВ.")

    return issues
