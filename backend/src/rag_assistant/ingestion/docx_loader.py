from __future__ import annotations

from rag_assistant.ingestion.base import DocumentLoader, content_hash, normalize_text
from rag_assistant.ingestion.models import LocationType, NormalizedSegment, SourceFile
from rag_assistant.ingestion.ooxml import extract_xml_text, open_office_archive


class DocxLoader(DocumentLoader):
    extensions = frozenset({"docx"})
    media_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )

    def load(self, source: SourceFile) -> list[NormalizedSegment]:
        with open_office_archive(source.path) as archive:
            text = normalize_text(extract_xml_text(archive, "word/document.xml"))
        if not text:
            return []
        return [
            NormalizedSegment(
                document_id=source.document_id,
                source_name=source.display_name,
                media_type=source.media_type,
                location_type=LocationType.PARAGRAPH,
                location=1,
                text=text,
                content_hash=content_hash(text),
            )
        ]

