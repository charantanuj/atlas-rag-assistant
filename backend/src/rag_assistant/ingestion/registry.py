from __future__ import annotations

from pathlib import Path

from rag_assistant.ingestion.base import DocumentLoader
from rag_assistant.ingestion.docx_loader import DocxLoader
from rag_assistant.ingestion.errors import UnsupportedFormatError
from rag_assistant.ingestion.html_loader import HtmlLoader
from rag_assistant.ingestion.pdf_loader import PdfLoader
from rag_assistant.ingestion.pptx_loader import PptxLoader
from rag_assistant.ingestion.txt_loader import TxtLoader


class LoaderRegistry:
    def __init__(self, loaders: list[DocumentLoader]) -> None:
        self._by_extension = {
            extension: loader for loader in loaders for extension in loader.extensions
        }

    def for_path(self, path: Path) -> DocumentLoader:
        extension = path.suffix.lower().lstrip(".")
        try:
            return self._by_extension[extension]
        except KeyError as exc:
            raise UnsupportedFormatError(
                f"No document loader is registered for '.{extension or '<none>'}'."
            ) from exc

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)


def default_loader_registry() -> LoaderRegistry:
    return LoaderRegistry([PdfLoader(), TxtLoader(), HtmlLoader(), DocxLoader(), PptxLoader()])

