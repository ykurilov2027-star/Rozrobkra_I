"""Колекція документів: читання, метадані, поділ на фрагменти.

Файли читаються з `docs/`, діляться за заголовками й абзацами, а
метадані документа копіюються до кожного фрагмента.
"""

import hashlib
import os
import re
from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу — відправна точка, а не рекомендація. У чому їх
# рахувати (символи, слова, токени моделі) і як застосовувати, коли
# ділите за заголовками, — ваше рішення.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    id: str = ""


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    closing = next(
        (position for position, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise ValueError("Front matter починається з `---`, але не має закривального `---`.")
    parsed = yaml.safe_load("\n".join(lines[1:closing])) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Front matter має містити YAML-словник метаданих.")
    metadata = {
        str(key): (
            value.isoformat() if isinstance(value, (date, datetime))
            else "" if value is None
            else str(value)
        )
        for key, value in parsed.items()
    }
    body = "\n".join(lines[closing + 1:]).lstrip("\n")
    return metadata, body


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.relative_to(docs_dir).as_posix(), metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити Markdown на структуровані фрагменти зі збереженням контексту."""
    if CHUNK_SIZE < 1 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_SIZE має бути > 0, а CHUNK_OVERLAP — у межах [0, CHUNK_SIZE).")

    title = str(metadata.get("title") or Path(source).stem)
    sections: list[tuple[str, list[str]]] = []
    heading = str(metadata.get("title") or "")
    section_lines: list[str] = []

    def save_section() -> None:
        if section_lines:
            sections.append((heading, section_lines.copy()))
            section_lines.clear()

    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            save_section()
            heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()
        else:
            section_lines.append(line)
    save_section()

    def markdown_blocks(lines: list[str]) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        for line in lines:
            if not line.strip():
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            else:
                current.append(line.rstrip())
        if current:
            blocks.append("\n".join(current).strip())
        return blocks

    def split_oversized(block: str, budget: int) -> list[str]:
        if len(block) <= budget:
            return [block]
        # Keep list/table/quote lines together wherever possible.
        lines = block.splitlines()
        if len(lines) > 1:
            parts: list[str] = []
            current: list[str] = []
            length = 0
            for line in lines:
                if len(line) > budget:
                    if current:
                        parts.append("\n".join(current))
                        current, length = [], 0
                    parts.extend(split_oversized(line, budget))
                    continue
                extra = len(line) + (1 if current else 0)
                if current and length + extra > budget:
                    parts.append("\n".join(current))
                    current, length = [], 0
                current.append(line)
                length += len(line) + (1 if len(current) > 1 else 0)
            if current:
                parts.append("\n".join(current))
            return parts

        pieces: list[str] = []
        remaining = block
        while len(remaining) > budget:
            boundary = remaining.rfind(" ", 0, budget + 1)
            if boundary < budget // 2:
                boundary = budget
            pieces.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    chunks: list[Chunk] = []
    ordinal = 0
    for section_heading, lines in sections:
        blocks = markdown_blocks(lines)
        if not blocks:
            continue
        prefix = f"[Документ: {title}"
        if section_heading:
            prefix += f" | Розділ: {section_heading}"
        prefix += "]\n"
        budget = max(1, CHUNK_SIZE - len(prefix))
        expanded = [piece for block in blocks for piece in split_oversized(block, budget)]

        pending: list[str] = []
        pending_length = 0
        position = 0
        while position < len(expanded):
            block = expanded[position]
            extra = len(block) + (2 if pending else 0)
            if pending and pending_length + extra > budget:
                body = "\n\n".join(pending)
                ordinal += 1
                chunk_metadata = dict(metadata)
                chunk_metadata["heading"] = section_heading
                chunk_metadata["chunk_index"] = ordinal
                chunk_id = hashlib.sha256(
                    f"{source}\0{ordinal}\0{body}".encode("utf-8")
                ).hexdigest()[:20]
                chunks.append(Chunk(
                    text=prefix + body,
                    source=source,
                    metadata=chunk_metadata,
                    id=chunk_id,
                ))
                overlap: list[str] = []
                overlap_length = 0
                for previous in reversed(pending):
                    cost = len(previous) + (2 if overlap else 0)
                    if overlap and overlap_length + cost > CHUNK_OVERLAP:
                        break
                    if not overlap and len(previous) > CHUNK_OVERLAP:
                        break
                    overlap.insert(0, previous)
                    overlap_length += cost
                if (
                    overlap
                    and position < len(expanded)
                    and overlap_length + 2 + len(expanded[position]) > budget
                ):
                    overlap = []
                    overlap_length = 0
                pending = overlap
                pending_length = overlap_length
                continue
            pending.append(block)
            pending_length += extra
            position += 1

        if pending:
            body = "\n\n".join(pending)
            ordinal += 1
            chunk_metadata = dict(metadata)
            chunk_metadata["heading"] = section_heading
            chunk_metadata["chunk_index"] = ordinal
            chunk_id = hashlib.sha256(
                f"{source}\0{ordinal}\0{body}".encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(Chunk(
                text=prefix + body,
                source=source,
                metadata=chunk_metadata,
                id=chunk_id,
            ))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
