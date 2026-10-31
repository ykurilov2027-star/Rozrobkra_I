"""Читання документів, front matter та поділ за заголовками на фрагменти."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Межі розміру й перекриття вимірюються символами.
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
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити текст документа на фрагменти.

    Розбиває текст за заголовками й абзацами, не втрачаючи прив'язки до
    розділу. Розмір і перекриття вимірюються символами.
    """
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_SIZE має бути додатним, а CHUNK_OVERLAP меншим за нього")

    sections: list[tuple[str, str]] = []
    heading = "Без розділу"
    body_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            section_text = "\n".join(body_lines).strip()
            if section_text:
                sections.append((heading, section_text))
            heading = line[3:].strip()
            body_lines = []
        else:
            body_lines.append(line)
    section_text = "\n".join(body_lines).strip()
    if section_text:
        sections.append((heading, section_text))

    chunks: list[Chunk] = []
    chunk_index = 0
    for section_heading, section_text in sections:
        section_chunks: list[str] = []
        start = 0
        while start < len(section_text):
            end = min(start + CHUNK_SIZE, len(section_text))
            if end < len(section_text):
                boundary = section_text.rfind("\n", start + int(CHUNK_SIZE * 0.6), end)
                if boundary > start:
                    end = boundary
            chunk_text = section_text[start:end].strip()
            if chunk_text:
                section_chunks.append(chunk_text)
            if end >= len(section_text):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)

        for chunk_text in section_chunks:
            chunk_metadata = dict(metadata)
            chunk_metadata.update({"heading": section_heading, "chunk_index": chunk_index})
            chunks.append(Chunk(text=chunk_text, source=source, metadata=chunk_metadata))
            chunk_index += 1
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
