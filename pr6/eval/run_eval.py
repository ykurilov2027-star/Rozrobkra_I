"""Run the RAG assistant against a question set and save a detailed report."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import index, keyword, llm, rag, retrieval  # noqa: E402

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
EXAMPLE_QUESTIONS_PATH = Path(__file__).with_name("questions.example.json")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
INTERNAL_DOCUMENT = "внутрішня-інструкція-оператора.md"
ARCHIVED_DOCUMENT = "правила-доставка-2025.md"


def _contains_all_facts(answer: str, expected_facts: str) -> tuple[bool, list[dict]]:
    facts = [fact.strip() for fact in expected_facts.split(";") if fact.strip()]
    checks = [
        {"fact": fact, "matched": fact.casefold() in answer.casefold()}
        for fact in facts
    ]
    return all(item["matched"] for item in checks), checks


def _record_result(number: int, item: dict, result: rag.Answer) -> dict:
    expected_docs = item.get("очікувані_документи", [])
    expected_context_docs = item.get("очікувані_документи_в_контексті", expected_docs)
    expected_found = item.get("очікується_відповідь", True)
    context_docs = [hit.chunk.source for hit in result.retrieved]
    positions = {
        document: [
            position for position, source in enumerate(context_docs, start=1)
            if source == document
        ]
        for document in expected_context_docs
    }
    missing_context_docs = [
        document for document, document_positions in positions.items()
        if not document_positions
    ]
    restricted_docs = [
        source for source in context_docs
        if source in {INTERNAL_DOCUMENT, ARCHIVED_DOCUMENT}
    ]
    retrieval_errors = []
    if missing_context_docs:
        retrieval_errors.append("необхідного документа немає в контексті")
    if restricted_docs:
        retrieval_errors.append("у контекст потрапив внутрішній або архівний документ")
    if not expected_context_docs and context_docs:
        retrieval_errors.append("контекст містить нерелевантні документи")

    facts_match, fact_checks = _contains_all_facts(
        result.text, item.get("очікувана_відповідь", "")
    )
    refusal_correct = result.found == expected_found
    generation_errors = []
    if not refusal_correct:
        generation_errors.append(
            "безпідставна відмова" if expected_found else "відповідь замість відмови"
        )
    if expected_found and not facts_match:
        generation_errors.append("не збігаються очікувані факти")
    if result.found and not result.sources:
        generation_errors.append("відповідь без джерел")

    retrieved_by_ref = {
        position: hit for position, hit in enumerate(result.retrieved, start=1)
    }
    invalid_citations = [
        source.ref for source in result.sources
        if source.ref not in retrieved_by_ref
        or retrieved_by_ref[source.ref].chunk.source != source.chunk.source
    ]
    if invalid_citations:
        generation_errors.append("посилання не відповідає фрагментам контексту")

    cited_text = " ".join(
        f"{source.chunk.text} {source.chunk.metadata.get('updated', '')}"
        for source in result.sources
    )
    answer_text_without_refs = re.sub(r"\[(\d+)\]", "", result.text)
    answer_numbers = set(NUMBER_PATTERN.findall(answer_text_without_refs))
    cited_numbers = set(NUMBER_PATTERN.findall(cited_text))
    unsupported_numbers = sorted(answer_numbers - cited_numbers)
    if result.found and unsupported_numbers:
        generation_errors.append("числа відповіді відсутні в процитованих фрагментах")

    if not expected_context_docs and not expected_docs:
        expected_docs_found = True
    else:
        expected_docs_found = not missing_context_docs

    error_classes = []
    if retrieval_errors:
        error_classes.append("Помилка пошуку")
    if generation_errors:
        error_classes.append("Помилка генерації")

    return {
        "number": number,
        "type": item.get("вид"),
        "question": item["питання"],
        "expected_documents": expected_docs,
        "expected_context_documents": expected_context_docs,
        "expected_document_positions": positions,
        "expected_documents_found": expected_docs_found,
        "documents_in_context": context_docs,
        "restricted_documents_in_context": restricted_docs,
        "expected_answer_facts": item.get("очікувана_відповідь"),
        "fact_checks": fact_checks,
        "answer_facts_match": facts_match if expected_found else None,
        "expected_found": expected_found,
        "found": result.found,
        "refusal_correct": refusal_correct,
        "answer": result.text,
        "citations": [
            {
                "ref": source.ref,
                "document": source.chunk.source,
                "heading": source.chunk.metadata.get("heading"),
                "updated": source.chunk.metadata.get("updated"),
            }
            for source in result.sources
        ],
        "invalid_citations": invalid_citations,
        "unsupported_answer_numbers": unsupported_numbers,
        "context": [
            {
                "document": hit.chunk.source,
                "metadata": hit.chunk.metadata,
                "score": hit.score,
                "text": hit.chunk.text,
            }
            for hit in result.retrieved
        ],
        "elapsed": result.elapsed,
        "usage": result.usage,
        "retrieval_completed": True,
        "retrieval_errors": retrieval_errors,
        "generation_errors": generation_errors,
        "error_class": error_classes,
    }


def _summary(results: list[dict]) -> dict:
    expected_context_cases = [
        record for record in results
        if record.get("retrieval_completed") and record["expected_context_documents"]
    ]
    answer_cases = [
        record for record in results
        if record["expected_found"] and record.get("found") is not None
    ]
    refusal_cases = [
        record for record in results
        if not record["expected_found"] and record.get("found") is not None
    ]
    completed_cases = [record for record in results if record.get("found") is not None]
    retrieval_times = [
        record["elapsed"]["retrieval"]
        for record in results
        if record.get("elapsed", {}).get("retrieval") is not None
    ]
    return {
        "question_count": len(results),
        "generation_completed_cases": len(completed_cases),
        "mean_context_chunks": (
            sum(len(record.get("documents_in_context", [])) for record in results)
            / len(results) if results else 0
        ),
        "max_context_chunks": max(
            (len(record.get("documents_in_context", [])) for record in results),
            default=0,
        ),
        "mean_retrieval_seconds": (
            sum(retrieval_times) / len(retrieval_times) if retrieval_times else None
        ),
        "expected_document_context_recall": (
            sum(record["expected_documents_found"] for record in expected_context_cases)
            / len(expected_context_cases) if expected_context_cases else None
        ),
        "answer_fact_match_rate": (
            sum(record["answer_facts_match"] is True for record in answer_cases)
            / len(answer_cases) if answer_cases else None
        ),
        "correct_refusal_rate": (
            sum(record["refusal_correct"] for record in refusal_cases) / len(refusal_cases)
            if refusal_cases else None
        ),
        "answers_when_refusal_expected": (
            sum(record["found"] for record in refusal_cases) if refusal_cases else None
        ),
        "false_refusal_count": (
            sum(not record["found"] for record in answer_cases) if answer_cases else None
        ),
        "invalid_citation_cases": (
            sum(bool(record["invalid_citations"]) for record in completed_cases)
            if completed_cases else None
        ),
        "unsupported_number_cases": (
            sum(bool(record["unsupported_answer_numbers"]) for record in completed_cases)
            if completed_cases else None
        ),
        "retrieval_error_cases": sum(
            "Помилка пошуку" in record["error_class"] for record in results
        ),
        "generation_error_cases": sum(
            "Помилка генерації" in record["error_class"] for record in results
        ),
        "fact_matching_method": "case-insensitive substring; review report manually",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="eval/report.json",
        help="шлях звіту від кореня проєкту; використовуйте різні файли для прогонів",
    )
    parser.add_argument("--label", default="baseline", help="позначка конфігурації цього прогону")
    parser.add_argument(
        "--context-chunks",
        type=int,
        help="перевизначити кількість фрагментів контексту для контрольованого експерименту",
    )
    args = parser.parse_args()
    if args.context_chunks is not None:
        if args.context_chunks <= 0:
            parser.error("--context-chunks має бути додатним числом")
        retrieval.CONTEXT_CHUNKS = args.context_chunks

    questions_path = QUESTIONS_PATH if QUESTIONS_PATH.exists() else EXAMPLE_QUESTIONS_PATH
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["питання"]
    if not 12 <= len(questions) <= 15:
        raise ValueError(f"Eval має містити 12–15 питань; знайдено {len(questions)}")

    vector_index = index.load()
    keyword_index = keyword.build(vector_index.chunks)
    results = []

    for number, item in enumerate(questions, start=1):
        started = time.perf_counter()
        try:
            answer = rag.answer(
                item["питання"],
                vector_index,
                keyword_index,
                filters=item.get("фільтри") or None,
            )
            record = _record_result(number, item, answer)
        except Exception as exc:
            if isinstance(exc, llm.LLMError):
                error_class = "Помилка генерації"
            elif isinstance(exc, (ValueError, FileNotFoundError, OSError, RuntimeError)):
                error_class = "Помилка пошуку"
            else:
                error_class = "Помилка виконання"
            retrieved = getattr(exc, "retrieved", [])
            expected_docs = item.get("очікувані_документи", [])
            expected_context_docs = item.get(
                "очікувані_документи_в_контексті", expected_docs
            )
            context_docs = [hit.chunk.source for hit in retrieved]
            positions = {
                document: [
                    position for position, source in enumerate(context_docs, start=1)
                    if source == document
                ]
                for document in expected_context_docs
            }
            missing_context_docs = [
                document for document, document_positions in positions.items()
                if not document_positions
            ]
            restricted_docs = [
                source for source in context_docs
                if source in {INTERNAL_DOCUMENT, ARCHIVED_DOCUMENT}
            ]
            retrieval_errors = []
            if getattr(exc, "pipeline_trace_available", False):
                if missing_context_docs:
                    retrieval_errors.append("необхідного документа немає в контексті")
                if restricted_docs:
                    retrieval_errors.append(
                        "у контекст потрапив внутрішній або архівний документ"
                    )
                if not expected_context_docs and context_docs:
                    retrieval_errors.append("контекст містить нерелевантні документи")
            generation_errors = [str(exc)] if error_class == "Помилка генерації" else []
            error_classes = []
            if retrieval_errors:
                error_classes.append("Помилка пошуку")
            if generation_errors:
                error_classes.append("Помилка генерації")
            if not error_classes:
                error_classes.append(error_class)
            record = {
                "number": number,
                "type": item.get("вид"),
                "question": item["питання"],
                "expected_documents": expected_docs,
                "expected_context_documents": expected_context_docs,
                "expected_document_positions": positions,
                "expected_documents_found": not missing_context_docs,
                "documents_in_context": context_docs,
                "restricted_documents_in_context": restricted_docs,
                "expected_found": item.get("очікується_відповідь", True),
                "found": None,
                "answer_facts_match": None,
                "refusal_correct": None,
                "citations": [],
                "invalid_citations": [],
                "unsupported_answer_numbers": [],
                "context": [
                    {
                        "document": hit.chunk.source,
                        "metadata": hit.chunk.metadata,
                        "score": hit.score,
                        "text": hit.chunk.text,
                    }
                    for hit in retrieved
                ],
                "elapsed": getattr(exc, "elapsed", {}),
                "usage": None,
                "retrieval_completed": getattr(exc, "pipeline_trace_available", False),
                "retrieval_errors": retrieval_errors,
                "generation_errors": generation_errors,
                "error_class": error_classes,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_total": time.perf_counter() - started,
            }
        results.append(record)
        state = ", ".join(record["error_class"]) or "OK"
        print(f"[{number:02d}/{len(questions):02d}] {state}: {item['питання']}")

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "questions_file": questions_path.name,
        "label": args.label,
        "configuration": {
            "rag_context_chunks": retrieval.CONTEXT_CHUNKS,
            "rag_context_budget": retrieval.CONTEXT_BUDGET,
            "similarity_threshold": retrieval.SIMILARITY_THRESHOLD,
            "search_top_k": index.DEFAULT_TOP_K,
        },
        "question_count": len(results),
        "summary": _summary(results),
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Звіт: {report_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
