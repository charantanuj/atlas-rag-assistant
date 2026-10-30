from pathlib import Path
import sys

import pytest

from rag_assistant.ingestion.errors import (
    InvalidDocumentError,
    MissingParserDependencyError,
)
from rag_assistant.ingestion.models import SourceFile
from rag_assistant.ingestion.pdf_loader import PdfLoader


def pdf_source(path: Path) -> SourceFile:
    return SourceFile(path, "doc_pdf", path.name, "application/pdf", "hash")


def test_pdf_loader_rejects_spoofed_extension(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_text("not actually a pdf", encoding="utf-8")

    with pytest.raises(InvalidDocumentError):
        PdfLoader().load(pdf_source(path))


def test_pdf_loader_reports_missing_parser_dependency(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "minimal.pdf"
    path.write_bytes(b"%PDF-placeholder")
    # Simulate an optional dependency being unavailable even when the test venv
    # has the normal production parser installed.
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(MissingParserDependencyError):
        PdfLoader().load(pdf_source(path))

