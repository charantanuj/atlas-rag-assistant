from __future__ import annotations

from rag_assistant.ingestion.base import DocumentLoader, content_hash, normalize_text
from rag_assistant.ingestion.errors import InvalidDocumentError
from rag_assistant.ingestion.models import LocationType, NormalizedSegment, SourceFile


class TxtLoader(DocumentLoader):
    extensions = frozenset({"txt"})
    media_types = frozenset({"text/plain"})

    def load(self, source: SourceFile) -> list[NormalizedSegment]:
        raw = source.path.read_bytes()
        if b"\x00" in raw:
            raise InvalidDocumentError("TXT file appears to contain binary data.")

        text = self._decode(raw)
        lines = text.splitlines()
        segments: list[NormalizedSegment] = []
        for start in range(0, len(lines), 100):
            normalized = normalize_text("\n".join(lines[start : start + 100]))
            if not normalized:
                continue
            segments.append(
                NormalizedSegment(
                    document_id=source.document_id,
                    source_name=source.display_name,
                    media_type=source.media_type,
                    location_type=LocationType.LINE_RANGE,
                    location=start + 1,
                    text=normalized,
                    content_hash=content_hash(normalized),
                )
            )
        return segments

    @staticmethod
    def _decode(raw: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise InvalidDocumentError("TXT encoding is not supported.")

