"""Інструменти помічника: що модель може попросити виконати і як це
перевіряється.

Рішення:

* набір із **6 дозволених** інструментів + `create_return`:
  `list_orders`, `get_order`, `search_products`, `get_product`,
  `get_stock`, `delivery_quote`, `create_return`;
* **адміністративні операції не описані** — модель про них не знає:
  `issue_refund`, `set_order_status`, `update_price`, `add_bonus`;
* **скасування не даємо** моделі — хай робить у кабінеті;
* `customer_id` **не аргумент** жодного інструмента — береться з
  `Context` (сеанс), модель його не бачить;
* після схеми перевіряємо **зміст**: чи замовлення існує, чи належить
  клієнтові, чи товар у замовленні; сервісні правила (14 днів,
  `returnable`, наявність на складі) — за сервісом;
* у результаті моделі — **лише потрібні поля**, без `internal_note`,
  `card_last4`, чужих `customer_id`, повних контактів;
* помилки сервісу (`ShopError`) — теж результат, з полем `error.code`
  і зрозумілим `message`, щоб модель переказала клієнтові;
* жоден виняток не виходить назовні: будь-який результат — `ToolResult`.
"""

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from jsonschema import Draft202012Validator, ValidationError

from shop import service
from shop.service import (
    ShopError, NotFound, InvalidRequest, PolicyViolation, Unavailable,
)

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Те, що відомо про сеанс незалежно від моделі."""
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


# ---------------------------------------------------------------------------
# Хелпери
# ---------------------------------------------------------------------------
def _strip_money(value: Any) -> Any:
    """Гроші — рядками з двома знаками."""
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    return value


def _pick_order_fields(order: dict) -> dict:
    """Лише те, що потрібно клієнтові. Без internal_note, customer_id,
    payment.card_last4 і решти службових полів."""
    return {
        "order_id": order["order_id"],
        "created_at": order["created_at"],
        "status": order["status"],
        "status_label": order.get("status_label", order["status"]),
        "items": [
            {
                "sku": it["sku"],
                "name": it["name"],
                "quantity": it["quantity"],
                "price": it["price"],
            }
            for it in order.get("items", [])
        ],
        "total": order["total"],
        "delivery": {
            "method": order.get("delivery", {}).get("method"),
            "city": order.get("delivery", {}).get("city"),
            "point": order.get("delivery", {}).get("point"),
            "tracking": order.get("delivery", {}).get("tracking"),
            "delivered_at": order.get("delivery", {}).get("delivered_at"),
            "cost": order.get("delivery", {}).get("cost"),
        },
    }


def _pick_list_row(o: dict) -> dict:
    return {
        "order_id": o["order_id"],
        "created_at": o["created_at"],
        "status": o["status"],
        "status_label": o["status_label"],
        "total": o["total"],
        "items_count": o["items_count"],
    }


def _pick_product(p: dict) -> dict:
    """Картка товару без внутрішніх полів. description очищаємо від
    тексту, адресованого «AI-асистенту» — це дані, але моделі вони не
    потрібні."""
    desc = p.get("description", "")
    # Прибираємо підозрілі «примітки для AI»
    for marker in ("Примітка для AI", "AI-асистент", "assistant:"):
        idx = desc.find(marker)
        if idx != -1:
            desc = desc[:idx].strip()
    return {
        "sku": p["sku"],
        "name": p["name"],
        "category": p["category"],
        "price": p["price"],
        "weight_kg": p["weight_kg"],
        "warranty_months": p["warranty_months"],
        "returnable": p["returnable"],
        "bulky": p["bulky"],
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Опис інструментів
# ---------------------------------------------------------------------------
def specs() -> List[Dict[str, Any]]:
    """Повернути описи інструментів у форматі OpenAI tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_orders",
                "description": (
                    "Повертає список замовлень поточного клієнта (номер, "
                    "дата, статус, сума). Використовуй, коли клієнт питає "
                    "про свої замовлення, не називаючи номера. Не викликай, "
                    "якщо клієнт назвав конкретний номер — виклич get_order."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": (
                    "Повертає деталі замовлення клієнта за номером. "
                    "Використовуй, коли клієнт назвав номер замовлення "
                    "і питає про його стан, склад або доставку. Номер — "
                    "послідовність цифр, без пробілів і префіксів."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "pattern": r"^\d{3,8}$",
                            "description": (
                                "Номер замовлення, лише цифри, "
                                "без пробілів (наприклад, 10458)."
                            ),
                        },
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": (
                    "Шукає товари в каталозі за словами в назві, категорії "
                    "чи описі. Використовуй, коли клієнт питає про товар, "
                    "але не назвав артикул. Повертає короткі картки: "
                    "артикул, назва, ціна. Якщо треба деталі — виклич "
                    "get_product за отриманим артикулом."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Пошукові слова (наприклад, «роутер "
                                "Альтаїр»). Порожній рядок — усі товари."
                            ),
                        },
                        "category": {
                            "type": "string",
                            "description": "Категорія, якщо відома.",
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Максимальна ціна в гривнях.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1, "maximum": 20,
                            "description": "Скільки товарів повернути.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product",
                "description": (
                    "Повертає картку товару за артикулом: ціна, вага, "
                    "гарантія, чи можна повернути товар належної якості, "
                    "опис. Використовуй після search_products, коли "
                    "потрібні деталі."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару (наприклад, ALT-AX3).",
                        },
                    },
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock",
                "description": (
                    "Повертає наявність товару на складі за артикулом. "
                    "Якщо available=0, у полі expected_restock може бути "
                    "дата очікуваної поставки. Використовуй, коли клієнт "
                    "питає, чи є товар."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару.",
                        },
                    },
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delivery_quote",
                "description": (
                    "Повертає вартість і строк доставки набору товарів у "
                    "місто. items — список {sku, quantity}. Методи: "
                    "branch — у відділення перевізника (безкоштовно від "
                    "2000 грн), courier — курʼєром, pickup — самовивіз зі "
                    "складу. Магазин доставляє лише по Україні."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Місто доставки.",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["branch", "courier", "pickup"],
                            "description": "Спосіб доставки.",
                        },
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {"type": "string"},
                                    "quantity": {"type": "integer", "minimum": 1},
                                },
                                "required": ["sku", "quantity"],
                            },
                            "description": "Товари для розрахунку.",
                        },
                    },
                    "required": ["city", "method", "items"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_return",
                "description": (
                    "Створює заявку на повернення товару із замовлення. "
                    "Використовуй, коли клієнт хоче повернути товар і "
                    "назвав причину. Причини: not_suitable — належна "
                    "якість, не підійшло; defect — виробничий дефект; "
                    "wrong_item — не той товар; damaged — пошкоджено при "
                    "доставці. Якщо клієнт не сказав причину — спитай."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "pattern": r"^\d{3,8}$",
                            "description": "Номер замовлення, лише цифри.",
                        },
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару із замовлення.",
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["not_suitable", "defect",
                                     "wrong_item", "damaged"],
                            "description": "Причина повернення.",
                        },
                        "quantity": {
                            "type": "integer", "minimum": 1,
                            "description": "Кількість, за замовчуванням 1.",
                        },
                        "comment": {
                            "type": "string",
                            "description": "Необов'язковий коментар.",
                        },
                    },
                    "required": ["order_id", "sku", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Перевірка й виконання викликів
# ---------------------------------------------------------------------------
_NAME_TO_FUNC: Dict[str, Callable] = {
    "list_orders": lambda ctx: service.list_orders(ctx.customer_id),
    "get_order": lambda ctx, order_id: service.get_order(order_id),
    "search_products": lambda ctx, query, category=None, max_price=None, limit=10:
        service.search_products(query, category, max_price, limit),
    "get_product": lambda ctx, sku: service.get_product(sku),
    "get_stock": lambda ctx, sku: service.get_stock(sku),
    "delivery_quote": lambda ctx, city, method, items:
        service.delivery_quote(city, method, items),
    "create_return": lambda ctx, order_id, sku, reason, quantity=1, comment="":
        service.create_return(order_id, sku, reason, quantity, comment),
}


def _build_validators() -> Dict[str, Draft202012Validator]:
    out = {}
    for spec in specs():
        f = spec["function"]
        out[f["name"]] = Draft202012Validator(f["parameters"])
    return out


_VALIDATORS: Optional[Dict[str, Draft202012Validator]] = None


def _validators() -> Dict[str, Draft202012Validator]:
    global _VALIDATORS
    if _VALIDATORS is None:
        _VALIDATORS = _build_validators()
    return _VALIDATORS


def _ensure_own_order(order_id: str, ctx: Context) -> dict:
    """Перевірити, що замовлення існує й належить клієнтові.

    Сервіс не перевіряє власника — це робить код помічника. Інакше
    клієнт C-1001 міг би попросити показати чуже замовлення.
    """
    order = service.get_order(order_id)
    if order.get("customer_id") != ctx.customer_id:
        raise NotFound(f"замовлення {order_id} не знайдено")
    return order


def _check_content(name: str, args: dict, ctx: Context) -> None:
    """Перевірки, які схема не покриває: власник, товар у замовленні."""
    if name == "get_order":
        _ensure_own_order(args["order_id"], ctx)

    elif name == "create_return":
        order = _ensure_own_order(args["order_id"], ctx)
        sku = args["sku"]
        if not any(it["sku"] == sku for it in order.get("items", [])):
            raise InvalidRequest(
                f"товару {sku} немає в замовленні {args['order_id']}"
            )

    elif name == "list_orders":
        # дозволено без параметрів
        pass


def _sanitize_result(name: str, raw: Any) -> Any:
    """Залишити лише потрібні поля. Прибирає internal_note, customer_id,
    card_last4, контакти й усе, чого клієнтові бачити не можна."""
    if name == "list_orders":
        return [_pick_list_row(o) for o in raw]

    if name == "get_order":
        return _pick_order_fields(raw)

    if name == "search_products":
        return [
            {"sku": p["sku"], "name": p["name"],
             "category": p["category"], "price": p["price"]}
            for p in raw
        ]

    if name == "get_product":
        return _pick_product(raw)

    if name == "get_stock":
        return {
            "sku": raw["sku"],
            "available": raw["available"],
            "expected_restock": raw.get("expected_restock"),
        }

    if name == "delivery_quote":
        return {
            "city": raw["city"],
            "method": raw["method"],
            "cost": raw["cost"],
            "days": raw["days"],
            "note": raw.get("note", ""),
        }

    if name == "create_return":
        return {
            "return_id": raw["return_id"],
            "order_id": raw["order_id"],
            "sku": raw["sku"],
            "reason": raw["reason"],
            "status": raw["status"],
            "next_steps": raw["next_steps"],
        }

    return raw


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    """Виконати виклик інструмента, який запропонувала модель.

    Шари перевірки:
    1. назва — чи є такий інструмент серед дозволених;
    2. JSON — чи рядок аргументів розбирається;
    3. схема — типи, обов'язкові поля, enum, pattern;
    4. зміст — власник замовлення, наявність товару.

    Помилки сервісу (`ShopError`) — теж результат із полем `error`.
    Жоден виняток не виходить назовні.
    """
    # 1. Назва
    if name not in _NAME_TO_FUNC:
        return ToolResult(
            status="rejected",
            reason=f"інструмент {name!r} невідомий",
            content={"error": "unknown_tool",
                     "message": f"Інструмент {name!r} не підтримується."},
        )

    # 2. JSON
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except (json.JSONDecodeError, TypeError) as e:
        return ToolResult(
            status="rejected",
            reason=f"аргументи не є JSON: {e}",
            content={"error": "bad_json",
                     "message": "Аргументи виклику не є коректним JSON."},
        )

    if not isinstance(args, dict):
        return ToolResult(
            status="rejected",
            reason="аргументи не є JSON-об'єктом",
            content={"error": "bad_json",
                     "message": "Аргументи мають бути JSON-об'єктом."},
        )

    # 3. Схема
    validator = _validators()[name]
    try:
        validator.validate(args)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "?"
        return ToolResult(
            status="rejected",
            reason=f"аргументи не за схемою: {path}: {e.message}",
            arguments=args,
            content={"error": "bad_schema",
                     "message": f"Аргумент «{path}» некоректний: {e.message}"},
        )

    # 4. Зміст
    try:
        _check_content(name, args, ctx)
    except NotFound as e:
        # чуже або неіснуюче замовлення — маскуємо під «не знайдено»
        return ToolResult(
            status="rejected",
            reason=str(e),
            arguments=args,
            content={"error": "not_found", "message": str(e)},
        )
    except ShopError as e:
        return ToolResult(
            status="rejected",
            reason=str(e),
            arguments=args,
            content={"error": e.code, "message": str(e)},
        )

    # Виконання
    func = _NAME_TO_FUNC[name]
    try:
        raw = func(ctx, **args)
    except NotFound as e:
        return ToolResult(
            status="error",
            reason=str(e),
            arguments=args,
            content={"error": "not_found", "message": str(e)},
        )
    except InvalidRequest as e:
        return ToolResult(
            status="error",
            reason=str(e),
            arguments=args,
            content={"error": "invalid_request", "message": str(e)},
        )
    except PolicyViolation as e:
        return ToolResult(
            status="error",
            reason=str(e),
            arguments=args,
            content={"error": "policy", "message": str(e)},
        )
    except Unavailable as e:
        return ToolResult(
            status="error",
            reason=str(e),
            arguments=args,
            content={"error": "unavailable",
                     "message": "Сервіс магазину тимчасово недоступний. "
                                "Спробуйте пізніше."},
        )
    except ShopError as e:
        logger.warning("ShopError у %s: %s", name, e)
        return ToolResult(
            status="error",
            reason=str(e),
            arguments=args,
            content={"error": e.code, "message": str(e)},
        )
    except Exception as e:
        logger.exception("Непередбачена помилка в інструменті %s: %s", name, e)
        return ToolResult(
            status="error",
            reason=f"{type(e).__name__}: {e}",
            arguments=args,
            content={"error": "internal",
                     "message": "Внутрішня помилка інструмента."},
        )

    # Очищення результату
    try:
        safe = _sanitize_result(name, raw)
    except Exception as e:
        logger.exception("Помилка санітайзу результату %s: %s", name, e)
        safe = raw

    return ToolResult(
        status="ok",
        content=safe,
        arguments=args,
    )