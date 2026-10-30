from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path

from rag_assistant.ingestion.models import NormalizedSegment, SourceFile


class DocumentLoader(ABC):
    extensions: frozenset[str]
    media_types: frozenset[str]

    @abstractmethod
    def load(self, source: SourceFile) -> list[NormalizedSegment]:
        """Extract ordered, location-aware segments from one source file."""


def content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.replace("\x00", "").splitlines())
    return "\n".join(line for line in lines if line).strip()

