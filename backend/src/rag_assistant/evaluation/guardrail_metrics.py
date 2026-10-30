from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.evaluation.dataset import GuardrailCase
from rag_assistant.guardrails.input_guardrails import InputGuardrails
from rag_assistant.guardrails.models import GuardrailDecision


@dataclass(frozen=True, slots=True)
class GuardrailEvaluation:
    total: int
    attack_cases: int
    benign_cases: int
    attacks_blocked: int
    benign_blocked: int
    attack_block_rate: float
    false_positive_rate: float
    failed_case_ids: tuple[str, ...]


def evaluate_input_guardrails(
    guardrails: InputGuardrails, cases: tuple[GuardrailCase, ...]
) -> GuardrailEvaluation:
    attacks = [case for case in cases if case.should_block]
    benign = [case for case in cases if not case.should_block]
    attacks_blocked = 0
    benign_blocked = 0
    failed = []
    for case in cases:
        blocked = guardrails.evaluate(case.text).decision == GuardrailDecision.BLOCK
        if case.should_block:
            attacks_blocked += int(blocked)
        else:
            benign_blocked += int(blocked)
        if blocked != case.should_block:
            failed.append(case.id)
    return GuardrailEvaluation(
        total=len(cases),
        attack_cases=len(attacks),
        benign_cases=len(benign),
        attacks_blocked=attacks_blocked,
        benign_blocked=benign_blocked,
        attack_block_rate=attacks_blocked / len(attacks) if attacks else 1.0,
        false_positive_rate=benign_blocked / len(benign) if benign else 0.0,
        failed_case_ids=tuple(failed),
    )

