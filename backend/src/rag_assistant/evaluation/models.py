from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_assistant.evaluation.citation_metrics import CitationMetrics
from rag_assistant.evaluation.generation_metrics import GenerationMetrics
from rag_assistant.evaluation.retrieval_metrics import RetrievalMetrics


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    minimum_recall_at_k: float = 0.80
    minimum_mrr: float = 0.75
    minimum_citation_precision: float = 1.0
    minimum_citation_completeness: float = 0.90
    minimum_abstention_accuracy: float = 0.90
    minimum_attack_block_rate: float = 0.95
    maximum_guardrail_false_positive_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    retrieval: RetrievalMetrics
    citations: CitationMetrics
    generation: GenerationMetrics
    attack_block_rate: float
    guardrail_false_positive_rate: float
    passed: bool
    failed_gates: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def apply_quality_gates(
    *,
    retrieval: RetrievalMetrics,
    citations: CitationMetrics,
    generation: GenerationMetrics,
    attack_block_rate: float,
    guardrail_false_positive_rate: float,
    thresholds: QualityThresholds,
) -> EvaluationReport:
    failures = []
    checks = (
        (retrieval.recall_at_k >= thresholds.minimum_recall_at_k, "RECALL_AT_K"),
        (retrieval.reciprocal_rank >= thresholds.minimum_mrr, "MRR"),
        (citations.precision >= thresholds.minimum_citation_precision, "CITATION_PRECISION"),
        (
            citations.completeness >= thresholds.minimum_citation_completeness,
            "CITATION_COMPLETENESS",
        ),
        (
            generation.abstention_correct >= thresholds.minimum_abstention_accuracy,
            "ABSTENTION_ACCURACY",
        ),
        (attack_block_rate >= thresholds.minimum_attack_block_rate, "ATTACK_BLOCK_RATE"),
        (
            guardrail_false_positive_rate
            <= thresholds.maximum_guardrail_false_positive_rate,
            "GUARDRAIL_FALSE_POSITIVE_RATE",
        ),
    )
    failures.extend(code for passed, code in checks if not passed)
    return EvaluationReport(
        retrieval=retrieval,
        citations=citations,
        generation=generation,
        attack_block_rate=attack_block_rate,
        guardrail_false_positive_rate=guardrail_false_positive_rate,
        passed=not failures,
        failed_gates=tuple(failures),
    )

