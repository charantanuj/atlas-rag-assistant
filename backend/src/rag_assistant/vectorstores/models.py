from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]
    distance: float
    relevance: float

