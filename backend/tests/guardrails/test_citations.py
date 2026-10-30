from rag_assistant.guardrails.citation_validation import CitationValidator
from rag_assistant.guardrails.models import Citation
from rag_assistant.processing.reranking import RankedEvidence


def evidence() -> RankedEvidence:
    return RankedEvidence(
        chunk_id="chk_1",
        text="Transfers above $50,000 require two approvers.",
        metadata={
            "document_id": "doc_1",
            "source_name": "policy.pdf",
            "location_type": "page",
            "location": 7,
        },
        vector_score=0.8,
        reranker_score=0.9,
        final_score=0.88,
    )


def citation(**changes) -> Citation:
    values = {
        "id": 1,
        "chunk_id": "chk_1",
        "document_id": "doc_1",
        "document_name": "policy.pdf",
        "location_type": "page",
        "location": 7,
        "excerpt": "Transfers above $50,000",
        "relevance": 0.88,
    }
    values.update(changes)
    return Citation(**values)


def test_valid_citation_resolves_to_retrieved_evidence() -> None:
    result = CitationValidator().validate(
        "Two approvers are required [1].", (citation(),), (evidence(),)
    )

    assert result.valid is True
    assert result.referenced_ids == (1,)


def test_invented_reference_is_rejected() -> None:
    result = CitationValidator().validate(
        "Two approvers are required [2].", (citation(),), (evidence(),)
    )

    assert result.valid is False
    assert "UNKNOWN_CITATION_REFERENCE" in result.reason_codes


def test_wrong_page_is_rejected() -> None:
    result = CitationValidator().validate(
        "Two approvers are required [1].", (citation(location=8),), (evidence(),)
    )

    assert result.valid is False
    assert "CITATION_LOCATION_MISMATCH" in result.reason_codes

