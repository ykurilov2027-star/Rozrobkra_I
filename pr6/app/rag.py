"""RAG-конвеєр: пошук, бюджетування контексту, генерація та валідація."""

from dataclasses import dataclass, field
import re
import time

from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source

NO_INFORMATION = "У базі знань немає інформації для відповіді."
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")


@dataclass
class Answer:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `text` — відповідь для клієнта. `found` — чи відповідь спирається на
    базу знань (після вашої перевірки, а не зі слів моделі). `sources` —
    джерела, які показуються під відповіддю. `retrieved` — фрагменти,
    які бачила модель, для налагодження: за ними видно, чия це помилка —
    пошуку чи генерації. `elapsed` — час за етапами, наприклад
    `{"retrieval": 0.04, "generation": 1.9}`. `usage` — токени запиту й
    відповіді, якщо модель викликалась.
    """

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Відповісти на питання за базою знань.

    `retrieved` містить лише фрагменти, що фактично увійшли до контексту.
    Модельні цитати перевіряються проти присвоєних цим фрагментам номерів.
    """
    from . import llm, retrieval

    clean_question = question.strip()
    if not clean_question:
      raise ValueError("Введіть питання.")
    if index is None:
      raise FileNotFoundError("Індекс не збудовано — виконайте python ingest.py")

    retrieval_started = time.perf_counter()
    hits = retrieval.retrieve(clean_question, index, keyword_index, filters)
    context, context_sources = retrieval.build_context(hits)
    retrieval_elapsed = time.perf_counter() - retrieval_started
    actual_hits = [Hit(chunk=source.chunk, score=source.score) for source in context_sources]
    if not context_sources:
      return Answer(
        text=NO_INFORMATION,
        found=False,
        retrieved=[],
        model=llm.MODEL,
        elapsed={"retrieval": retrieval_elapsed, "generation": 0.0},
        usage=None,
      )

    generation_started = time.perf_counter()
    try:
      generated = llm.ask(clean_question, context)
    except llm.LLMError as exc:
      exc.set_pipeline_trace(
        actual_hits,
        {
          "retrieval": retrieval_elapsed,
          "generation": time.perf_counter() - generation_started,
        },
      )
      raise

    def invalid_generation(message: str) -> llm.LLMError:
      error = llm.LLMError(message)
      error.set_pipeline_trace(
        actual_hits,
        {"retrieval": retrieval_elapsed, "generation": generated["elapsed"]},
      )
      return error

    valid_refs = {source.ref for source in context_sources}
    cited_refs = generated["sources"]
    if len(cited_refs) != len(set(cited_refs)):
      raise invalid_generation("Модель повернула повторні номери джерел.")
    if any(ref not in valid_refs for ref in cited_refs):
      raise invalid_generation("Модель послалася на джерело, якого не було в її контексті.")

    if not generated["found"]:
      return Answer(
        text=NO_INFORMATION,
        found=False,
        sources=[],
        retrieved=actual_hits,
        model=generated["model"],
        elapsed={"retrieval": retrieval_elapsed, "generation": generated["elapsed"]},
        usage=generated["usage"],
      )
    if not cited_refs:
      return Answer(
        text=NO_INFORMATION,
        found=False,
        sources=[],
        retrieved=actual_hits,
        model=generated["model"],
        elapsed={"retrieval": retrieval_elapsed, "generation": generated["elapsed"]},
        usage=generated["usage"],
      )

    cited = set(cited_refs)
    cited_sources = [source for source in context_sources if source.ref in cited]
    answer_numbers = set(NUMBER_PATTERN.findall(
      re.sub(r"\[(\d+)\]", "", generated["answer"])
    ))
    cited_content = " ".join(
      " ".join(
        str(source.chunk.metadata.get(field, ""))
        for field in ("title", "heading", "updated")
      ) + " " + source.chunk.text
      for source in cited_sources
    )
    unsupported_numbers = answer_numbers - set(NUMBER_PATTERN.findall(cited_content))
    if unsupported_numbers:
      raise invalid_generation(
        "Відповідь містить числа, яких немає у процитованих фрагментах: "
        + ", ".join(sorted(unsupported_numbers))
      )
    return Answer(
      text=generated["answer"],
      found=True,
      sources=cited_sources,
      retrieved=actual_hits,
      model=generated["model"],
      elapsed={"retrieval": retrieval_elapsed, "generation": generated["elapsed"]},
      usage=generated["usage"],
    )
