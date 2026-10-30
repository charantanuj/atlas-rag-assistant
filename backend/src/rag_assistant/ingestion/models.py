from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class LocationType(StrEnum):
    PAGE = "page"
    SLIDE = "slide"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    LINE_RANGE = "line_range"


class QualityFlag(StrEnum):
    EMPTY = "empty"
    ENCRYPTED = "encrypted"
    SYNTHETIC_NOISE = "synthetic_noise"
    POSSIBLE_DUPLICATE = "possible_duplicate"


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    document_id: str
    display_name: str
    media_type: str
    sha256: str


@dataclass(frozen=True, slots=True)
class NormalizedSegment:
    document_id: str
    source_name: str
    media_type: str
    location_type: LocationType
    location: int
    text: str
    content_hash: str
    heading: str | None = None
    quality_flags: tuple[QualityFlag, ...] = field(default_factory=tuple)

