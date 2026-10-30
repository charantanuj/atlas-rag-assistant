from pathlib import Path

import pytest

from rag_assistant.ingestion.base import content_hash
from rag_assistant.ingestion.models import LocationType, NormalizedSegment
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.processing.splitting import RecursiveTextSplitter
from rag_assistant.vectorstores.chroma import ChromaVectorStore, CollectionVersionError


class FakeCollection:
    def __init__(self, metadata):
        self.metadata = metadata
        self.upsert_call = None

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def query(self, **kwargs):
        return {
            "ids": [["chk_1"]],
            "documents": [["supporting evidence"]],
            "metadatas": [[{"document_id": "doc_1", "location": 2}]],
            "distances": [[0.2]],
        }


class FakeClient:
    def __init__(self, existing_metadata=None):
        self.existing_metadata = existing_metadata
        self.collection = None

    def get_or_create_collection(self, name, metadata):
        self.collection = FakeCollection(self.existing_metadata or metadata)
        return self.collection


def one_chunk():
    text = "citation-ready evidence"
    segment = NormalizedSegment(
        document_id="doc_1",
        source_name="source.pdf",
        media_type="application/pdf",
        location_type=LocationType.PAGE,
        location=2,
        text=text,
        content_hash=content_hash(text),
    )
    return DocumentChunker(RecursiveTextSplitter()).chunk_segments([segment])[0]


def test_chroma_adapter_serializes_metadata_and_scores_results(tmp_path: Path) -> None:
    client = FakeClient()
    store = ChromaVectorStore(
        persist_path=tmp_path,
        collection_name="documents-v1",
        embedding_model="model-v1",
        dimension=3,
        client=client,
    )
    chunk = one_chunk()

    store.upsert([chunk], [[0.1, 0.2, 0.3]])
    results = store.query([0.1, 0.2, 0.3], top_k=5)

    assert client.collection.upsert_call["ids"] == [chunk.id]
    assert client.collection.upsert_call["metadatas"][0]["location"] == 2
    assert results[0].chunk_id == "chk_1"
    assert results[0].relevance == pytest.approx(0.8)


def test_chroma_adapter_rejects_collection_model_mismatch(tmp_path: Path) -> None:
    client = FakeClient({"embedding_model": "old-model", "dimension": 3})

    with pytest.raises(CollectionVersionError):
        ChromaVectorStore(
            persist_path=tmp_path,
            collection_name="documents-v1",
            embedding_model="new-model",
            dimension=3,
            client=client,
        )


def test_chroma_adapter_rejects_wrong_vector_dimension(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        persist_path=tmp_path,
        collection_name="documents-v1",
        embedding_model="model-v1",
        dimension=3,
        client=FakeClient(),
    )

    with pytest.raises(ValueError):
        store.upsert([one_chunk()], [[0.1, 0.2]])

