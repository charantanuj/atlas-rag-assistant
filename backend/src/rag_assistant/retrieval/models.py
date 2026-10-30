from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rag_assistant.processing.reranking import RankedEvidence


class EvidenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    status: EvidenceStatus
    evidence: tuple[RankedEvidence, ...]
    candidates_retrieved: int
    candidates_after_filtering: int
    minimum_score: float

