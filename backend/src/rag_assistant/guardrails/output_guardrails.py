from __future__ import annotations

import re

from rag_assistant.guardrails.citation_validation import CitationValidator
from rag_assistant.guardrails.models import Citation, GuardrailDecision, GuardrailResult
from rag_assistant.guardrails.pii_detection import SensitiveDataDetector
from rag_assistant.processing.reranking import RankedEvidence

_SENTENCE_BREAK = re.compile(r"(?<!\d)[.!?](?=\s|$)|(?<=\d)[.!?](?=\s|$)|\n+")
_REFERENCE = re.compile(r"\[\d+]" )


class OutputGuardrails:
    def __init__(
        self,
        *,
        citation_validator: CitationValidator | None = None,
        sensitive_data_detector: SensitiveDataDetector | None = None,
    ) -> None:
        self.citation_validator = citation_validator or CitationValidator()
        self.sensitive_data_detector = sensitive_data_detector or SensitiveDataDetector()

    def evaluate(
        self,
        *,
        answer: str,
        citations: tuple[Citation, ...],
        evidence: tuple[RankedEvidence, ...],
        insufficient_evidence: bool = False,
    ) -> GuardrailResult:
        normalized = answer.strip()
        if not normalized:
            return GuardrailResult(GuardrailDecision.ABSTAIN, ("EMPTY_OUTPUT",))
        sensitive = self.sensitive_data_detector.detect(normalized)
        if any(finding.code == "API_KEY" for finding in sensitive):
            return GuardrailResult(GuardrailDecision.BLOCK, ("SENSITIVE_OUTPUT",))
        if insufficient_evidence:
            if citations or _REFERENCE.search(normalized):
                return GuardrailResult(
                    GuardrailDecision.BLOCK,
                    ("ABSTENTION_WITH_CITATIONS",),
                )
            return GuardrailResult(GuardrailDecision.PASS, safe_text=normalized)
        if not evidence:
            return GuardrailResult(GuardrailDecision.ABSTAIN, ("NO_EVIDENCE",))

        citation_result = self.citation_validator.validate(
            normalized, citations, evidence
        )
        if not citation_result.valid:
            return GuardrailResult(
                GuardrailDecision.BLOCK, citation_result.reason_codes
            )
        if not _REFERENCE.search(normalized):
            return GuardrailResult(GuardrailDecision.ABSTAIN, ("MISSING_CITATION",))
        factual_sentences = [sentence.strip() for sentence in _SENTENCE_BREAK.split(normalized) if len(sentence.strip()) >= 20]
        if factual_sentences and not all(_REFERENCE.search(sentence) for sentence in factual_sentences):
            return GuardrailResult(
                GuardrailDecision.ABSTAIN,
                ("UNSUPPORTED_SENTENCE",),
            )
        return GuardrailResult(GuardrailDecision.PASS, safe_text=normalized)

