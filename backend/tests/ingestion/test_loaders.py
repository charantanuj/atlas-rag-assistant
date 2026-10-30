from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from rag_assistant.ingestion.docx_loader import DocxLoader
from rag_assistant.ingestion.errors import InvalidDocumentError, UnsupportedFormatError
from rag_assistant.ingestion.html_loader import HtmlLoader
from rag_assistant.ingestion.models import LocationType, SourceFile
from rag_assistant.ingestion.pptx_loader import PptxLoader
from rag_assistant.ingestion.registry import default_loader_registry
from rag_assistant.ingestion.txt_loader import TxtLoader


def source(path: Path, media_type: str) -> SourceFile:
    return SourceFile(
        path=path,
        document_id="doc_test",
        display_name=path.name,
        media_type=media_type,
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def test_registry_routes_each_required_format(tmp_path: Path) -> None:
    registry = default_loader_registry()

    assert registry.supported_extensions == {
        "pdf", "txt", "html", "htm", "docx", "pptx"
    }
    with pytest.raises(UnsupportedFormatError):
        registry.for_path(tmp_path / "document.csv")


def test_txt_loader_preserves_line_location(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("first line\nsecond line", encoding="utf-8")

    segments = TxtLoader().load(source(path, "text/plain"))

    assert len(segments) == 1
    assert segments[0].location_type == LocationType.LINE_RANGE
    assert segments[0].location == 1
    assert segments[0].text == "first line\nsecond line"


def test_txt_loader_rejects_binary_content(tmp_path: Path) -> None:
    path = tmp_path / "fake.txt"
    path.write_bytes(b"text\x00binary")

    with pytest.raises(InvalidDocumentError):
        TxtLoader().load(source(path, "text/plain"))


def test_html_loader_removes_script_and_style(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<h1>Title</h1><script>ignore()</script><p>Visible text</p><style>x{}</style>",
        encoding="utf-8",
    )

    segment = HtmlLoader().load(source(path, "text/html"))[0]

    assert segment.text == "Title\nVisible text"
    assert "ignore" not in segment.text


def test_docx_loader_reads_document_xml(tmp_path: Path) -> None:
    path = tmp_path / "report.docx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>Policy text</w:t></w:r></w:p></w:body></w:document>',
        )

    segment = DocxLoader().load(source(path, DocxLoader.media_types.__iter__().__next__()))[0]

    assert segment.text == "Policy text"
    assert segment.location_type == LocationType.PARAGRAPH


def test_pptx_loader_returns_one_segment_per_nonempty_slide(tmp_path: Path) -> None:
    path = tmp_path / "slides.pptx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide2.xml",
            '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>Second</a:t></p:sld>',
        )
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>First</a:t></p:sld>',
        )

    segments = PptxLoader().load(source(path, next(iter(PptxLoader.media_types))))

    assert [(item.location, item.text) for item in segments] == [(1, "First"), (2, "Second")]

