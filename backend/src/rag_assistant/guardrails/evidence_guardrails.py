from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.guardrails.prompt_injection import PromptInjectionDetector
from rag_assistant.processing.reranking import RankedEvidence


@dataclass(frozen=True, slots=True)
class EvidenceGuardrailResult:
    safe_evidence: tuple[RankedEvidence, ...]
    rejected_chunk_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


class EvidenceGuardrails:
    """Quarantines chunks that attempt to instruct the model."""

    def __init__(self, detector: PromptInjectionDetector | None = None) -> None:
        self.detector = detector or PromptInjectionDetector()

    def evaluate(
        self, evidence: tuple[RankedEvidence, ...]
    ) -> EvidenceGuardrailResult:
        safe = []
        rejected = []
        reasons = set()
        for item in evidence:
            findings = self.detector.detect(item.text)
            if findings:
                rejected.append(item.chunk_id)
                reasons.update(f"EVIDENCE_{finding.code}" for finding in findings)
            else:
                safe.append(item)
        return EvidenceGuardrailResult(
            safe_evidence=tuple(safe),
            rejected_chunk_ids=tuple(rejected),
            reason_codes=tuple(sorted(reasons)),
        )
