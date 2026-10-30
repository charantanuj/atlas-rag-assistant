from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.guardrails.models import Citation


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    precision: float
    completeness: float
    exact_source_accuracy: float


@dataclass(frozen=True, slots=True)
class ExpectedCitation:
    chunk_id: str
    document_id: str
    location_type: str
    location: int


def evaluate_citations(
    actual: tuple[Citation, ...],
    expected: tuple[ExpectedCitation, ...],
) -> CitationMetrics:
    expected_by_chunk = {item.chunk_id: item for item in expected}
    correct_chunks = sum(item.chunk_id in expected_by_chunk for item in actual)
    exact = 0
    for citation in actual:
        target = expected_by_chunk.get(citation.chunk_id)
        if target and (
            citation.document_id == target.document_id
            and citation.location_type == target.location_type
            and citation.location == target.location
        ):
            exact += 1
    precision = correct_chunks / len(actual) if actual else (1.0 if not expected else 0.0)
    completeness = correct_chunks / len(expected) if expected else 1.0
    exact_accuracy = exact / len(actual) if actual else (1.0 if not expected else 0.0)
    return CitationMetrics(precision, completeness, exact_accuracy)

