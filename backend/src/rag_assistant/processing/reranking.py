"""Local and cross-encoder reranking implementations."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_assistant.vectorstores.models import VectorSearchResult

_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RankedEvidence:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]
    vector_score: float
    reranker_score: float
    final_score: float


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[VectorSearchResult]
    ) -> list[RankedEvidence]:
        """Return all candidates ordered from most to least relevant."""


class LexicalVectorReranker(Reranker):
    """Dependency-free baseline combining vector, term, and title relevance."""

    def __init__(
        self,
        *,
        vector_weight: float = 0.65,
        lexical_weight: float = 0.30,
        title_weight: float = 0.05,
        noise_penalty: float = 0.25,
    ) -> None:
        if not math.isclose(vector_weight + lexical_weight + title_weight, 1.0):
            raise ValueError("reranking weights must sum to 1.0")
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.title_weight = title_weight
        self.noise_penalty = noise_penalty

    def rerank(
        self, query: str, candidates: list[VectorSearchResult]
    ) -> list[RankedEvidence]:
        query_terms = self._terms(query)
        ranked = []
        for candidate in candidates:
            lexical = self._overlap(query_terms, self._terms(candidate.text))
            source_name = str(candidate.metadata.get("source_name", ""))
            title = self._overlap(query_terms, self._terms(source_name))
            reranker_score = self.lexical_weight * lexical + self.title_weight * title
            score = self.vector_weight * candidate.relevance + reranker_score
            quality_flags = str(candidate.metadata.get("quality_flags", ""))
            if "synthetic_noise" in quality_flags.split("|"):
                score *= 1.0 - self.noise_penalty
            ranked.append(
                RankedEvidence(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    vector_score=candidate.relevance,
                    reranker_score=reranker_score,
                    final_score=max(0.0, min(1.0, score)),
                )
            )
        return sorted(ranked, key=lambda item: (-item.final_score, item.chunk_id))

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term.lower() for term in _TOKEN.findall(text) if len(term) > 1}

    @staticmethod
    def _overlap(query_terms: set[str], candidate_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        return len(query_terms & candidate_terms) / len(query_terms)


class FlashRankReranker(Reranker):
    """Optional local cross-encoder adapter, loaded only when selected."""

    def __init__(self, ranker: Any | None = None, model_name: str | None = None) -> None:
        if ranker is None:
            try:
                from flashrank import Ranker
            except ImportError as exc:
                raise RuntimeError(
                    "FlashRank is not installed. Install backend/requirements.txt."
                ) from exc
            kwargs = {"model_name": model_name} if model_name else {}
            ranker = Ranker(**kwargs)
        self.ranker = ranker

    def rerank(
        self, query: str, candidates: list[VectorSearchResult]
    ) -> list[RankedEvidence]:
        if not candidates:
            return []
        passages = [
            {"id": candidate.chunk_id, "text": candidate.text, "meta": candidate.metadata}
            for candidate in candidates
        ]
        try:
            from flashrank import RerankRequest

            request: Any = RerankRequest(query=query, passages=passages)
        except ImportError:
            request = {"query": query, "passages": passages}
        raw_results = self.ranker.rerank(request)
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ranked = []
        for result in raw_results:
            chunk_id = str(result["id"])
            candidate = by_id[chunk_id]
            cross_score = max(0.0, min(1.0, float(result["score"])))
            ranked.append(
                RankedEvidence(
                    chunk_id=chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    vector_score=candidate.relevance,
                    reranker_score=cross_score,
                    final_score=0.25 * candidate.relevance + 0.75 * cross_score,
                )
            )
        return ranked

