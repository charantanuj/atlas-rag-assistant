from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.ingestion.base import file_hash
from rag_assistant.ingestion.errors import IngestionError
from rag_assistant.ingestion.models import SourceFile
from rag_assistant.ingestion.quality import QualityClassifier
from rag_assistant.ingestion.registry import LoaderRegistry
from rag_assistant.storage.metadata_repository import MetadataRepository
from rag_assistant.storage.models import DocumentRecord
from rag_assistant.storage.normalized_store import NormalizedDocumentStore


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    document: DocumentRecord
    duplicate: bool


class IngestionPipeline:
    def __init__(
        self,
        *,
        registry: LoaderRegistry,
        metadata: MetadataRepository,
        normalized_store: NormalizedDocumentStore,
        quality_classifier: QualityClassifier | None = None,
    ) -> None:
        self.registry = registry
        self.metadata = metadata
        self.normalized_store = normalized_store
        self.quality_classifier = quality_classifier or QualityClassifier()

    def ingest(self, path: Path, *, display_name: str | None = None) -> IngestionOutcome:
        digest = file_hash(path)
        existing = self.metadata.find_by_hash(digest)
        if existing is not None:
            if existing.status.value != "failed":
                return IngestionOutcome(document=existing, duplicate=True)
            self.metadata.delete(existing.id)

        extension = path.suffix.lower().lstrip(".")
        original_name = display_name or path.name
        media_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        document_id = f"doc_{digest[:24]}"
        record = self.metadata.begin(
            document_id=document_id,
            original_name=original_name,
            media_type=media_type,
            extension=extension,
            sha256=digest,
            size_bytes=path.stat().st_size,
        )
        source = SourceFile(
            path=path,
            document_id=document_id,
            display_name=original_name,
            media_type=media_type,
            sha256=digest,
        )
        try:
            segments = self.registry.for_path(path).load(source)
            segments = self.quality_classifier.classify(segments)
            normalized_path = self.normalized_store.write(document_id, segments)
            flags = tuple(
                sorted({flag.value for segment in segments for flag in segment.quality_flags})
            )
            record = self.metadata.mark_ready(
                document_id,
                segment_count=len(segments),
                character_count=sum(len(segment.text) for segment in segments),
                normalized_path=normalized_path,
                quality_flags=flags,
            )
            return IngestionOutcome(document=record, duplicate=False)
        except IngestionError as exc:
            self.metadata.mark_failed(document_id, exc.code)
            raise

