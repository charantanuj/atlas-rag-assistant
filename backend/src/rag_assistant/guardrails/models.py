from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GuardrailDecision(StrEnum):
    PASS = "pass"
    BLOCK = "block"
    SANITIZE = "sanitize"
    ABSTAIN = "abstain"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    decision: GuardrailDecision
    reason_codes: tuple[str, ...] = ()
    safe_text: str | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    id: int
    chunk_id: str
    document_id: str
    document_name: str
    location_type: str
    location: int
    excerpt: str
    relevance: float

