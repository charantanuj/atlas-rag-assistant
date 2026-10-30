from pathlib import Path

from rag_assistant.evaluation.citation_metrics import CitationMetrics
from rag_assistant.evaluation.generation_metrics import GenerationMetrics
from rag_assistant.evaluation.models import QualityThresholds, apply_quality_gates
from rag_assistant.evaluation.reporting import write_json_report
from rag_assistant.evaluation.retrieval_metrics import RetrievalMetrics


def report(recall=0.9, precision=1.0):
    return apply_quality_gates(
        retrieval=RetrievalMetrics(0.5, recall, 0.8, 0.8),
        citations=CitationMetrics(precision, 1.0, 1.0),
        generation=GenerationMetrics(1.0, 1.0, 1.0),
        attack_block_rate=1.0,
        guardrail_false_positive_rate=0.0,
        thresholds=QualityThresholds(),
    )


def test_quality_gates_pass_good_metrics() -> None:
    value = report()

    assert value.passed is True
    assert value.failed_gates == ()


def test_quality_gates_name_failures() -> None:
    value = report(recall=0.2, precision=0.5)

    assert value.passed is False
    assert "RECALL_AT_K" in value.failed_gates
    assert "CITATION_PRECISION" in value.failed_gates


def test_json_report_is_machine_readable(tmp_path: Path) -> None:
    path = write_json_report(report(), tmp_path / "evaluation.json")

    text = path.read_text(encoding="utf-8")
    assert '"passed": true' in text
    assert '"recall_at_k": 0.9' in text

