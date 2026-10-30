from __future__ import annotations

from typing import Protocol

from rag_assistant.core.cache import TTLCache
from rag_assistant.retrieval.bm25 import reciprocal_rank_fusion
from rag_assistant.embeddings.base import EmbeddingProvider
from rag_assistant.processing.reranking import Reranker
from rag_assistant.retrieval.models import EvidenceStatus, RetrievalResult
from rag_assistant.vectorstores.models import VectorSearchResult


class QueryVectorStore(Protocol):
    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict | None = None,
    ) -> list[VectorSearchResult]: ...


class RetrievalService:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: QueryVectorStore,
        reranker: Reranker,
        retrieval_top_k: int = 12,
        rerank_top_n: int = 5,
        minimum_score: float = 0.45,
        query_cache: TTLCache[list[float]] | None = None,
        keyword_index=None,
    ) -> None:
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        if rerank_top_n < 1 or rerank_top_n > retrieval_top_k:
            raise ValueError("rerank_top_n must be between 1 and retrieval_top_k")
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n
        self.minimum_score = minimum_score
        self.query_cache = query_cache if query_cache is not None else TTLCache[list[float]]()
        self.keyword_index = keyword_index

    def retrieve(
        self,
        query: str,
        *,
        document_ids: tuple[str, ...] = (),
        allow_synthetic_noise: bool = False,
    ) -> RetrievalResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        where = None
        if document_ids:
            where = {"document_id": {"$in": list(document_ids)}}
        cache_key = query.strip().lower()
        query_vector = self.query_cache.get(cache_key)
        if query_vector is None:
            query_vector = self.embeddings.embed_query(query)
            self.query_cache.set(cache_key, query_vector)
        candidates = self.vector_store.query(
            query_vector,
            top_k=self.retrieval_top_k,
            where=where,
        )
        if self.keyword_index is not None:
            lexical = self.keyword_index.search(query, top_k=self.retrieval_top_k, document_ids=document_ids)
            candidates = reciprocal_rank_fusion(candidates, lexical)
        filtered = [
            candidate
            for candidate in candidates
            if allow_synthetic_noise or not self._is_synthetic(candidate)
        ]
        ranked = self.reranker.rerank(query, filtered)
        accepted = tuple(
            item
            for item in ranked[: self.rerank_top_n]
            if item.final_score >= self.minimum_score
        )
        return RetrievalResult(
            status=(
                EvidenceStatus.SUFFICIENT if accepted else EvidenceStatus.INSUFFICIENT
            ),
            evidence=accepted,
            candidates_retrieved=len(candidates),
            candidates_after_filtering=len(filtered),
            minimum_score=self.minimum_score,
        )

    @staticmethod
    def _is_synthetic(candidate: VectorSearchResult) -> bool:
        flags = str(candidate.metadata.get("quality_flags", "")).split("|")
        return "synthetic_noise" in flags

