from __future__ import annotations

import re
from dataclasses import dataclass

from rag_assistant.guardrails.models import Citation
from rag_assistant.processing.reranking import RankedEvidence

_REFERENCE = re.compile(r"\[(\d+)]")


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    valid: bool
    reason_codes: tuple[str, ...]
    referenced_ids: tuple[int, ...]


class CitationValidator:
    def validate(
        self,
        answer: str,
        citations: tuple[Citation, ...],
        evidence: tuple[RankedEvidence, ...],
    ) -> CitationValidationResult:
        reasons: set[str] = set()
        referenced_ids = tuple(dict.fromkeys(int(value) for value in _REFERENCE.findall(answer)))
        citation_by_id = {citation.id: citation for citation in citations}
        evidence_by_chunk = {item.chunk_id: item for item in evidence}

        if len(citation_by_id) != len(citations):
            reasons.add("DUPLICATE_CITATION_ID")
        for reference in referenced_ids:
            if reference not in citation_by_id:
                reasons.add("UNKNOWN_CITATION_REFERENCE")
        for citation in citations:
            item = evidence_by_chunk.get(citation.chunk_id)
            if item is None:
                reasons.add("CITATION_NOT_RETRIEVED")
                continue
            if citation.document_id != str(item.metadata.get("document_id", "")):
                reasons.add("CITATION_DOCUMENT_MISMATCH")
            if citation.document_name != str(item.metadata.get("source_name", "")):
                reasons.add("CITATION_NAME_MISMATCH")
            if citation.location_type != str(item.metadata.get("location_type", "")):
                reasons.add("CITATION_LOCATION_TYPE_MISMATCH")
            if citation.location != int(item.metadata.get("location", -1)):
                reasons.add("CITATION_LOCATION_MISMATCH")
            normalized_evidence = " ".join(item.text.split())
            normalized_excerpt = " ".join(citation.excerpt.rstrip("…").split())
            if normalized_excerpt not in normalized_evidence:
                reasons.add("CITATION_EXCERPT_MISMATCH")
        if citations and not referenced_ids:
            reasons.add("CITATIONS_NOT_REFERENCED")
        return CitationValidationResult(
            valid=not reasons,
            reason_codes=tuple(sorted(reasons)),
            referenced_ids=referenced_ids,
        )

