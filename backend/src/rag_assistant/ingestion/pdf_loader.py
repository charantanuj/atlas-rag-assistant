from __future__ import annotations

from rag_assistant.ingestion.base import DocumentLoader, content_hash, normalize_text
from rag_assistant.ingestion.errors import (
    EncryptedDocumentError,
    InvalidDocumentError,
    MissingParserDependencyError,
)
from rag_assistant.ingestion.models import LocationType, NormalizedSegment, SourceFile


class PdfLoader(DocumentLoader):
    extensions = frozenset({"pdf"})
    media_types = frozenset({"application/pdf"})

    def load(self, source: SourceFile) -> list[NormalizedSegment]:
        if source.path.read_bytes()[:5] != b"%PDF-":
            raise InvalidDocumentError("File extension is PDF but its signature is invalid.")
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:
            raise MissingParserDependencyError(
                "Install the backend project dependencies to parse PDFs."
            ) from exc

        try:
            reader = PdfReader(source.path)
            if reader.is_encrypted:
                raise EncryptedDocumentError("Encrypted PDFs are not ingested.")
            segments = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = normalize_text(page.extract_text() or "")
                if not text:
                    continue
                segments.append(
                    NormalizedSegment(
                        document_id=source.document_id,
                        source_name=source.display_name,
                        media_type=source.media_type,
                        location_type=LocationType.PAGE,
                        location=page_number,
                        text=text,
                        content_hash=content_hash(text),
                    )
                )
            return segments
        except EncryptedDocumentError:
            raise
        except PdfReadError as exc:
            raise InvalidDocumentError("PDF could not be parsed.") from exc

