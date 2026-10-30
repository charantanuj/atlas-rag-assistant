from __future__ import annotations

from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from rag_assistant.ingestion.errors import InvalidDocumentError


def extract_xml_text(archive: ZipFile, member: str) -> str:
    try:
        root = ElementTree.fromstring(archive.read(member))
    except (KeyError, ElementTree.ParseError) as exc:
        raise InvalidDocumentError(f"Office document member is invalid: {member}") from exc
    return "\n".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))


def open_office_archive(path):
    try:
        return ZipFile(path)
    except BadZipFile as exc:
        raise InvalidDocumentError("Office document is not a valid OOXML package.") from exc

