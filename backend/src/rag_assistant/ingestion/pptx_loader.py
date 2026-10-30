from __future__ import annotations

import re

from rag_assistant.ingestion.base import DocumentLoader, content_hash, normalize_text
from rag_assistant.ingestion.models import LocationType, NormalizedSegment, SourceFile
from rag_assistant.ingestion.ooxml import extract_xml_text, open_office_archive

_SLIDE_MEMBER = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


class PptxLoader(DocumentLoader):
    extensions = frozenset({"pptx"})
    media_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    )

    def load(self, source: SourceFile) -> list[NormalizedSegment]:
        segments: list[NormalizedSegment] = []
        with open_office_archive(source.path) as archive:
            members = []
            for name in archive.namelist():
                match = _SLIDE_MEMBER.match(name)
                if match:
                    members.append((int(match.group(1)), name))
            for slide_number, member in sorted(members):
                text = normalize_text(extract_xml_text(archive, member))
                if not text:
                    continue
                segments.append(
                    NormalizedSegment(
                        document_id=source.document_id,
                        source_name=source.display_name,
                        media_type=source.media_type,
                        location_type=LocationType.SLIDE,
                        location=slide_number,
                        text=text,
                        content_hash=content_hash(text),
                    )
                )
        return segments

