from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def evaluate_retrieval(
    retrieved_ids: list[str] | tuple[str, ...],
    relevant_ids: set[str] | frozenset[str],
    *,
    k: int,
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(retrieved_ids[:k])
    hits = [1 if chunk_id in relevant_ids else 0 for chunk_id in ranked]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 1.0
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0
    )
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 1.0
    return RetrievalMetrics(precision, recall, reciprocal_rank, ndcg)


def mean_retrieval_metrics(values: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not values:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0)
    count = len(values)
    return RetrievalMetrics(
        precision_at_k=sum(item.precision_at_k for item in values) / count,
        recall_at_k=sum(item.recall_at_k for item in values) / count,
        reciprocal_rank=sum(item.reciprocal_rank for item in values) / count,
        ndcg_at_k=sum(item.ndcg_at_k for item in values) / count,
    )

