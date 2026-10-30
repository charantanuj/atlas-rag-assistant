from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rag_assistant.guardrails.models import Citation, GuardrailDecision


class Intent(StrEnum):
    DOCUMENT_QA = "document_qa"
    CONVERSATIONAL = "conversational"
    BLOCKED = "blocked"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BLOCKED = "blocked"
    CONVERSATIONAL = "conversational"


@dataclass(frozen=True, slots=True)
class Plan:
    intent: Intent
    safe_question: str | None
    guardrail_decision: GuardrailDecision
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentResponse:
    status: AnswerStatus
    answer: str
    citations: tuple[Citation, ...] = ()
    reason_codes: tuple[str, ...] = ()
    candidates_retrieved: int = 0
    evidence_used: int = 0

