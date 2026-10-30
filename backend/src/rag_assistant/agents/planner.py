"""Planner node: applies input policy and selects the permitted query path."""

from __future__ import annotations

import re

from rag_assistant.agents.models import Intent, Plan
from rag_assistant.guardrails.input_guardrails import InputGuardrails
from rag_assistant.guardrails.models import GuardrailDecision

_GREETING = re.compile(
    r"^(?:hi|hello|hey|hey there|good (?:morning|afternoon|evening)|thanks|thank you|help|who are you|what can you do|what(?:'s| is|s) (?:your|ur) name)[!.? ]*$",
    re.IGNORECASE,
)


class PlannerNode:
    def __init__(self, input_guardrails: InputGuardrails) -> None:
        self.input_guardrails = input_guardrails

    def plan(self, question: str) -> Plan:
        result = self.input_guardrails.evaluate(question)
        if result.decision == GuardrailDecision.BLOCK:
            return Plan(
                intent=Intent.BLOCKED,
                safe_question=None,
                guardrail_decision=result.decision,
                reason_codes=result.reason_codes,
            )
        safe_question = result.safe_text or question.strip()
        intent = Intent.CONVERSATIONAL if _GREETING.match(safe_question) else Intent.DOCUMENT_QA
        return Plan(
            intent=intent,
            safe_question=safe_question,
            guardrail_decision=result.decision,
            reason_codes=result.reason_codes,
        )

