from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    original_name: str
    media_type: str
    extension: str
    sha256: str
    size_bytes: int
    status: DocumentStatus
    segment_count: int
    character_count: int
    normalized_path: Path | None
    quality_flags: tuple[str, ...]
    error_code: str | None
    created_at: str
    updated_at: str

