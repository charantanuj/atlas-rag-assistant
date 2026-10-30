from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_assistant.embeddings.base import EmbeddingProvider
from rag_assistant.processing.chunking import DocumentChunk


class ChunkVectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None: ...


@dataclass(frozen=True, slots=True)
class IndexingResult:
    chunks_indexed: int
    batches: int
    embedding_model: str


class VectorIndexer:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: ChunkVectorStore,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.batch_size = batch_size

    def index(self, chunks: list[DocumentChunk]) -> IndexingResult:
        batches = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self.embeddings.embed_documents([chunk.text for chunk in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding provider returned an unexpected batch size")
            self.vector_store.upsert(batch, vectors)
            batches += 1
        return IndexingResult(
            chunks_indexed=len(chunks),
            batches=batches,
            embedding_model=self.embeddings.model_id,
        )

