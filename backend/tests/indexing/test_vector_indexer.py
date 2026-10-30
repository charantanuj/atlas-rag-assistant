from dataclasses import replace

from rag_assistant.embeddings.deterministic import DeterministicEmbeddingProvider
from rag_assistant.ingestion.base import content_hash
from rag_assistant.ingestion.models import LocationType, NormalizedSegment
from rag_assistant.indexing.vector_indexer import VectorIndexer
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.processing.splitting import RecursiveTextSplitter


class RecordingStore:
    def __init__(self) -> None:
        self.calls = []

    def upsert(self, chunks, embeddings) -> None:
        self.calls.append((chunks, embeddings))


def chunks(count: int):
    base = NormalizedSegment(
        document_id="doc_test",
        source_name="test.txt",
        media_type="text/plain",
        location_type=LocationType.LINE_RANGE,
        location=1,
        text="base",
        content_hash=content_hash("base"),
    )
    chunker = DocumentChunker(RecursiveTextSplitter())
    return [
        chunker.chunk_segments(
            [replace(base, location=index + 1, text=f"chunk text {index}")]
        )[0]
        for index in range(count)
    ]


def test_vector_indexer_batches_without_dropping_chunks() -> None:
    store = RecordingStore()
    indexer = VectorIndexer(
        embeddings=DeterministicEmbeddingProvider(dimension=16),
        vector_store=store,
        batch_size=2,
    )

    result = indexer.index(chunks(5))

    assert result.chunks_indexed == 5
    assert result.batches == 3
    assert [len(call[0]) for call in store.calls] == [2, 2, 1]
    assert all(len(vector) == 16 for call in store.calls for vector in call[1])

