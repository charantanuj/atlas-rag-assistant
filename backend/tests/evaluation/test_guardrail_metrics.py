from rag_assistant.evaluation.dataset import DEFAULT_GUARDRAIL_CASES
from rag_assistant.evaluation.guardrail_metrics import evaluate_input_guardrails
from rag_assistant.guardrails.input_guardrails import InputGuardrails


def test_default_adversarial_suite_meets_initial_safety_gate() -> None:
    result = evaluate_input_guardrails(InputGuardrails(), DEFAULT_GUARDRAIL_CASES)

    assert result.attack_block_rate == 1.0
    assert result.false_positive_rate == 0.0
    assert result.failed_case_ids == ()

