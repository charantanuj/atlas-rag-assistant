from __future__ import annotations

from html.parser import HTMLParser

from rag_assistant.ingestion.base import DocumentLoader, content_hash, normalize_text
from rag_assistant.ingestion.models import LocationType, NormalizedSegment, SourceFile


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


class HtmlLoader(DocumentLoader):
    extensions = frozenset({"html", "htm"})
    media_types = frozenset({"text/html"})

    def load(self, source: SourceFile) -> list[NormalizedSegment]:
        parser = _VisibleTextParser()
        parser.feed(source.path.read_text(encoding="utf-8", errors="replace"))
        text = normalize_text("\n".join(parser.parts))
        if not text:
            return []
        return [
            NormalizedSegment(
                document_id=source.document_id,
                source_name=source.display_name,
                media_type=source.media_type,
                location_type=LocationType.SECTION,
                location=1,
                text=text,
                content_hash=content_hash(text),
            )
        ]

