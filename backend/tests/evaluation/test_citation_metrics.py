from rag_assistant.evaluation.citation_metrics import (
    ExpectedCitation,
    evaluate_citations,
)
from rag_assistant.guardrails.models import Citation


def citation(chunk_id="chk_1", location=2):
    return Citation(1, chunk_id, "doc_1", "source.pdf", "page", location, "text", 0.9)


def test_citation_metrics_distinguish_chunk_and_exact_location() -> None:
    expected = (ExpectedCitation("chk_1", "doc_1", "page", 2),)

    exact = evaluate_citations((citation(),), expected)
    wrong_page = evaluate_citations((citation(location=3),), expected)

    assert exact.precision == 1.0
    assert exact.completeness == 1.0
    assert exact.exact_source_accuracy == 1.0
    assert wrong_page.precision == 1.0
    assert wrong_page.exact_source_accuracy == 0.0

