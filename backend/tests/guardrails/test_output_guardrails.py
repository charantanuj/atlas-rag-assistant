from rag_assistant.guardrails.models import Citation, GuardrailDecision
from rag_assistant.guardrails.output_guardrails import OutputGuardrails
from rag_assistant.processing.reranking import RankedEvidence


def evidence_and_citation():
    evidence = RankedEvidence(
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
    citation = Citation(
        id=1,
        chunk_id="chk_1",
        document_id="doc_1",
        document_name="policy.pdf",
        location_type="page",
        location=7,
        excerpt="Transfers above $50,000",
        relevance=0.88,
    )
    return evidence, citation


def test_grounded_cited_answer_passes() -> None:
    evidence, citation = evidence_and_citation()

    result = OutputGuardrails().evaluate(
        answer="Transfers above $50,000 require two approvers [1].",
        citations=(citation,),
        evidence=(evidence,),
    )

    assert result.decision == GuardrailDecision.PASS


def test_uncited_factual_sentence_abstains() -> None:
    evidence, citation = evidence_and_citation()

    result = OutputGuardrails().evaluate(
        answer="Transfers above $50,000 require two approvers.",
        citations=(citation,),
        evidence=(evidence,),
    )

    assert result.decision == GuardrailDecision.BLOCK
    assert "CITATIONS_NOT_REFERENCED" in result.reason_codes


def test_secret_leakage_is_blocked() -> None:
    evidence, citation = evidence_and_citation()

    result = OutputGuardrails().evaluate(
        answer="The api key = abcdefghijklmnop [1].",
        citations=(citation,),
        evidence=(evidence,),
    )

    assert result.decision == GuardrailDecision.BLOCK
    assert result.reason_codes == ("SENSITIVE_OUTPUT",)


def test_clean_insufficient_evidence_response_passes_without_citations() -> None:
    result = OutputGuardrails().evaluate(
        answer="I could not find sufficient evidence in the uploaded documents.",
        citations=(),
        evidence=(),
        insufficient_evidence=True,
    )

    assert result.decision == GuardrailDecision.PASS


def test_abstention_with_citation_is_blocked() -> None:
    evidence, citation = evidence_and_citation()

    result = OutputGuardrails().evaluate(
        answer="I could not find enough evidence [1].",
        citations=(citation,),
        evidence=(evidence,),
        insufficient_evidence=True,
    )

    assert result.decision == GuardrailDecision.BLOCK

