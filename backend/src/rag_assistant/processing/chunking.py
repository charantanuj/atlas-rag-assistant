"""Metadata-preserving, deterministic chunk construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from rag_assistant.ingestion.models import LocationType, NormalizedSegment, QualityFlag
from rag_assistant.processing.splitting import RecursiveTextSplitter


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    document_id: str
    source_name: str
    media_type: str
    location_type: LocationType
    location: int
    heading: str | None
    chunk_index: int
    text: str
    text_hash: str
    source_content_hash: str
    start_char: int
    end_char: int
    quality_flags: tuple[QualityFlag, ...]


class DocumentChunker:
    def __init__(self, splitter: RecursiveTextSplitter) -> None:
        self.splitter = splitter

    def chunk_segments(self, segments: list[NormalizedSegment]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for segment in segments:
            for split in self.splitter.split(segment.text):
                text_hash = sha256(split.text.encode("utf-8")).hexdigest()
                identity = "|".join(
                    (
                        segment.document_id,
                        segment.location_type.value,
                        str(segment.location),
                        str(split.index),
                        text_hash,
                    )
                )
                chunk_id = f"chk_{sha256(identity.encode('utf-8')).hexdigest()[:32]}"
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=segment.document_id,
                        source_name=segment.source_name,
                        media_type=segment.media_type,
                        location_type=segment.location_type,
                        location=segment.location,
                        heading=segment.heading,
                        chunk_index=split.index,
                        text=split.text,
                        text_hash=text_hash,
                        source_content_hash=segment.content_hash,
                        start_char=split.start_char,
                        end_char=split.end_char,
                        quality_flags=segment.quality_flags,
                    )
                )
        return chunks

