"""Responder node: generates, cites, and validates grounded answers."""

from __future__ import annotations

import re
from dataclasses import replace

from rag_assistant.agents.models import AgentResponse, AnswerStatus
from rag_assistant.generation.base import AnswerGenerator
from rag_assistant.guardrails.evidence_guardrails import EvidenceGuardrails
from rag_assistant.guardrails.models import Citation, GuardrailDecision
from rag_assistant.guardrails.output_guardrails import OutputGuardrails
from rag_assistant.retrieval.models import EvidenceStatus, RetrievalResult

_REFERENCE = re.compile(r"\[(\d+)]")
_GROUPED_REFERENCE = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")
_INSUFFICIENT_MESSAGE = (
    "I could not find sufficient evidence in the uploaded documents to answer that question."
)


class CitationAssembler:
    def assemble(self, answer: str, retrieval: RetrievalResult) -> tuple[Citation, ...]:
        referenced = tuple(dict.fromkeys(int(value) for value in _REFERENCE.findall(answer)))
        citations = []
        for citation_id in referenced:
            if citation_id < 1 or citation_id > len(retrieval.evidence):
                continue
            item = retrieval.evidence[citation_id - 1]
            citations.append(
                Citation(
                    id=citation_id,
                    chunk_id=item.chunk_id,
                    document_id=str(item.metadata.get("document_id", "")),
                    document_name=str(item.metadata.get("source_name", "")),
                    location_type=str(item.metadata.get("location_type", "")),
                    location=int(item.metadata.get("location", -1)),
                    excerpt=self._excerpt(item.text),
                    relevance=item.final_score,
                )
            )
        return tuple(citations)

    @staticmethod
    def _excerpt(text: str, maximum: int = 320) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= maximum else compact[: maximum - 1].rstrip() + "…"


class ResponderNode:
    def __init__(
        self,
        *,
        generator: AnswerGenerator,
        output_guardrails: OutputGuardrails,
        citation_assembler: CitationAssembler | None = None,
        evidence_guardrails: EvidenceGuardrails | None = None,
    ) -> None:
        self.generator = generator
        self.output_guardrails = output_guardrails
        self.citation_assembler = citation_assembler or CitationAssembler()
        self.evidence_guardrails = evidence_guardrails or EvidenceGuardrails()

    def respond(self, question: str, retrieval: RetrievalResult) -> AgentResponse:
        if retrieval.status == EvidenceStatus.INSUFFICIENT:
            guardrail = self.output_guardrails.evaluate(
                answer=_INSUFFICIENT_MESSAGE,
                citations=(),
                evidence=(),
                insufficient_evidence=True,
            )
            return AgentResponse(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=guardrail.safe_text or _INSUFFICIENT_MESSAGE,
                reason_codes=guardrail.reason_codes,
                candidates_retrieved=retrieval.candidates_retrieved,
            )

        evidence_check = self.evidence_guardrails.evaluate(retrieval.evidence)
        if not evidence_check.safe_evidence:
            return AgentResponse(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=_INSUFFICIENT_MESSAGE,
                reason_codes=evidence_check.reason_codes,
                candidates_retrieved=retrieval.candidates_retrieved,
            )
        safe_retrieval = replace(retrieval, evidence=evidence_check.safe_evidence)
        answer = self.generator.generate(question, safe_retrieval.evidence)
        answer = _GROUPED_REFERENCE.sub(lambda match: " ".join(f"[{number.strip()}]" for number in match.group(1).split(",")), answer)
        if answer == "INSUFFICIENT_EVIDENCE":
            return AgentResponse(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=_INSUFFICIENT_MESSAGE,
                candidates_retrieved=retrieval.candidates_retrieved,
            )
        citations = self.citation_assembler.assemble(answer, safe_retrieval)
        guardrail = self.output_guardrails.evaluate(
            answer=answer,
            citations=citations,
            evidence=safe_retrieval.evidence,
        )
        if guardrail.decision != GuardrailDecision.PASS:
            status = (
                AnswerStatus.INSUFFICIENT_EVIDENCE
                if guardrail.decision == GuardrailDecision.ABSTAIN
                else AnswerStatus.BLOCKED
            )
            return AgentResponse(
                status=status,
                answer=(
                    _INSUFFICIENT_MESSAGE
                    if status == AnswerStatus.INSUFFICIENT_EVIDENCE
                    else "The generated answer did not pass safety and grounding checks."
                ),
                reason_codes=guardrail.reason_codes,
                candidates_retrieved=retrieval.candidates_retrieved,
                evidence_used=len(safe_retrieval.evidence),
            )
        return AgentResponse(
            status=AnswerStatus.ANSWERED,
            answer=guardrail.safe_text or answer,
            citations=citations,
            candidates_retrieved=retrieval.candidates_retrieved,
            evidence_used=len(safe_retrieval.evidence),
        )



