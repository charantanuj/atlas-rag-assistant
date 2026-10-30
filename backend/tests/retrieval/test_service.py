from rag_assistant.embeddings.deterministic import DeterministicEmbeddingProvider
from rag_assistant.processing.reranking import LexicalVectorReranker
from rag_assistant.retrieval.models import EvidenceStatus
from rag_assistant.retrieval.service import RetrievalService
from rag_assistant.vectorstores.models import VectorSearchResult


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def query(self, query_embedding, *, top_k, where=None):
        self.calls.append({"vector": query_embedding, "top_k": top_k, "where": where})
        return self.results[:top_k]


def result(chunk_id, text, relevance, *, flags="", document_id="doc_1"):
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "document_id": document_id,
            "source_name": "source.pdf",
            "location_type": "page",
            "location": 1,
            "quality_flags": flags,
        },
        distance=1 - relevance,
        relevance=relevance,
    )


def service(results, minimum_score=0.45):
    store = FakeVectorStore(results)
    return (
        RetrievalService(
            embeddings=DeterministicEmbeddingProvider(dimension=16),
            vector_store=store,
            reranker=LexicalVectorReranker(),
            retrieval_top_k=5,
            rerank_top_n=3,
            minimum_score=minimum_score,
        ),
        store,
    )


def test_retrieval_returns_only_supported_reranked_evidence() -> None:
    retrieval, _ = service(
        [
            result("weak", "unrelated text", 0.4),
            result("strong", "wire transfer approval threshold", 0.75),
        ]
    )

    response = retrieval.retrieve("wire transfer approval threshold")

    assert response.status == EvidenceStatus.SUFFICIENT
    assert [item.chunk_id for item in response.evidence] == ["strong"]
    assert response.candidates_retrieved == 2


def test_retrieval_filters_synthetic_noise_by_default() -> None:
    retrieval, _ = service(
        [result("noise", "wire transfer approval", 0.95, flags="synthetic_noise")]
    )

    response = retrieval.retrieve("wire transfer approval")

    assert response.status == EvidenceStatus.INSUFFICIENT
    assert response.candidates_after_filtering == 0


def test_retrieval_can_include_noise_for_evaluation() -> None:
    retrieval, _ = service(
        [result("noise", "wire transfer approval", 0.95, flags="synthetic_noise")]
    )

    response = retrieval.retrieve("wire transfer approval", allow_synthetic_noise=True)

    assert response.status == EvidenceStatus.SUFFICIENT
    assert response.evidence[0].chunk_id == "noise"


def test_retrieval_returns_insufficient_when_threshold_is_not_met() -> None:
    retrieval, _ = service([result("weak", "some overlap", 0.2)], minimum_score=0.8)

    response = retrieval.retrieve("some overlap")

    assert response.status == EvidenceStatus.INSUFFICIENT
    assert response.evidence == ()


def test_retrieval_builds_document_filter() -> None:
    retrieval, store = service([])

    retrieval.retrieve("question", document_ids=("doc_a", "doc_b"))

    assert store.calls[0]["where"] == {
        "document_id": {"$in": ["doc_a", "doc_b"]}
    }

