from __future__ import annotations

from rag_assistant.ingestion.models import LocationType, NormalizedSegment, QualityFlag
from rag_assistant.indexing.vector_indexer import IndexingResult, VectorIndexer
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.storage.normalized_store import NormalizedDocumentStore


class DocumentIndexingService:
    """Converts persisted normalized segments into citation-ready vectors."""

    def __init__(
        self,
        *,
        normalized_store: NormalizedDocumentStore,
        chunker: DocumentChunker,
        vector_indexer: VectorIndexer,
    ) -> None:
        self.normalized_store = normalized_store
        self.chunker = chunker
        self.vector_indexer = vector_indexer

    def index_document(self, document_id: str) -> IndexingResult:
        payload = self.normalized_store.read(document_id)
        segments = [self._segment(item) for item in payload["segments"]]
        chunks = self.chunker.chunk_segments(segments)
        return self.vector_indexer.index(chunks)

    @staticmethod
    def _segment(item: dict) -> NormalizedSegment:
        return NormalizedSegment(
            document_id=item["document_id"],
            source_name=item["source_name"],
            media_type=item["media_type"],
            location_type=LocationType(item["location_type"]),
            location=int(item["location"]),
            text=item["text"],
            content_hash=item["content_hash"],
            heading=item.get("heading"),
            quality_flags=tuple(QualityFlag(value) for value in item.get("quality_flags", [])),
        )
