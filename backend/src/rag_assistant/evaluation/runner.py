from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.agents.models import AgentResponse
from rag_assistant.evaluation.citation_metrics import (
    CitationMetrics,
    ExpectedCitation,
    evaluate_citations,
)
from rag_assistant.evaluation.generation_metrics import (
    GenerationMetrics,
    evaluate_generation,
)
from rag_assistant.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    evaluate_retrieval,
)


@dataclass(frozen=True, slots=True)
class OfflineEvaluationCase:
    id: str
    question: str
    relevant_chunk_ids: frozenset[str]
    expected_citations: tuple[ExpectedCitation, ...]
    should_abstain: bool


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    id: str
    retrieval: RetrievalMetrics
    citations: CitationMetrics
    generation: GenerationMetrics


def evaluate_case(
    case: OfflineEvaluationCase,
    *,
    retrieved_chunk_ids: tuple[str, ...],
    response: AgentResponse,
    k: int,
) -> CaseEvaluation:
    return CaseEvaluation(
        id=case.id,
        retrieval=evaluate_retrieval(
            retrieved_chunk_ids, case.relevant_chunk_ids, k=k
        ),
        citations=evaluate_citations(response.citations, case.expected_citations),
        generation=evaluate_generation(
            response, should_abstain=case.should_abstain
        ),
    )

