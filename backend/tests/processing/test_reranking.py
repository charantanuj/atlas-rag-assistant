from rag_assistant.processing.reranking import (
    FlashRankReranker,
    LexicalVectorReranker,
)
from rag_assistant.vectorstores.models import VectorSearchResult


def candidate(chunk_id: str, text: str, relevance: float, source_name: str = "doc.pdf"):
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_name": source_name, "quality_flags": ""},
        distance=1 - relevance,
        relevance=relevance,
    )


def test_lexical_reranker_promotes_direct_query_evidence() -> None:
    candidates = [
        candidate("high-vector", "unrelated material", 0.70),
        candidate("direct", "cache coherence maintains a consistent cache view", 0.60),
    ]

    ranked = LexicalVectorReranker().rerank("cache coherence", candidates)

    assert ranked[0].chunk_id == "direct"
    assert ranked[0].final_score > ranked[1].final_score


def test_lexical_reranker_penalizes_synthetic_noise() -> None:
    clean = candidate("clean", "cache coherence", 0.6)
    noisy = VectorSearchResult(
        chunk_id="noise",
        text="cache coherence",
        metadata={"source_name": "noise.txt", "quality_flags": "synthetic_noise"},
        distance=0.4,
        relevance=0.6,
    )

    ranked = LexicalVectorReranker().rerank("cache coherence", [noisy, clean])

    assert ranked[0].chunk_id == "clean"


class FakeRanker:
    def rerank(self, request):
        return [{"id": "b", "score": 0.9}, {"id": "a", "score": 0.2}]


def test_flashrank_adapter_maps_cross_encoder_scores() -> None:
    reranker = FlashRankReranker(ranker=FakeRanker())
    candidates = [candidate("a", "first", 0.9), candidate("b", "second", 0.5)]

    ranked = reranker.rerank("query", candidates)

    assert [item.chunk_id for item in ranked] == ["b", "a"]
    assert ranked[0].final_score > ranked[1].final_score

