import pytest

from rag_assistant.evaluation.retrieval_metrics import evaluate_retrieval


def test_retrieval_metrics_use_rank_and_relevance() -> None:
    metrics = evaluate_retrieval(
        ["irrelevant", "relevant-a", "relevant-b"],
        {"relevant-a", "relevant-b"},
        k=3,
    )

    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert 0 < metrics.ndcg_at_k < 1


def test_no_relevant_documents_has_perfect_empty_recall() -> None:
    metrics = evaluate_retrieval([], set(), k=5)

    assert metrics.recall_at_k == 1.0
    assert metrics.ndcg_at_k == 1.0

