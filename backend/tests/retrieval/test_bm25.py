from rag_assistant.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from rag_assistant.vectorstores.models import VectorSearchResult


class Collection:
    def count(self):
        return 3

    def get(self, **kwargs):
        return {"ids": ["a", "b", "c"], "documents": ["AVX-512 vector sorting", "memory paging", "vector sorting"],
                "metadatas": [{"document_id": "one"}, {"document_id": "two"}, {"document_id": "two"}]}


def test_keyword_search_uses_whole_collection_and_scope():
    index = BM25Index(Collection())
    assert index.search("AVX-512", top_k=2)[0].chunk_id == "a"
    assert index.search("AVX-512", top_k=2, document_ids=("two",)) == []


def test_rank_fusion_deduplicates_and_rewards_agreement():
    def result(chunk):
        return VectorSearchResult(chunk, chunk, {}, 0.1, 0.9)
    result_ids = [r.chunk_id for r in reciprocal_rank_fusion([result("a"), result("b")], [result("b"), result("c")])]
    assert result_ids[0] == "b"
    assert set(result_ids) == {"a", "b", "c"}
