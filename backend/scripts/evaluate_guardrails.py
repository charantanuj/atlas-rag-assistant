"""Run the offline input-guardrail benchmark."""

from rag_assistant.evaluation.dataset import DEFAULT_GUARDRAIL_CASES
from rag_assistant.evaluation.guardrail_metrics import evaluate_input_guardrails
from rag_assistant.guardrails.input_guardrails import InputGuardrails


if __name__ == "__main__":
    result = evaluate_input_guardrails(InputGuardrails(), DEFAULT_GUARDRAIL_CASES)
    print(f"cases: {result.total}")
    print(f"attack_block_rate: {result.attack_block_rate:.3f}")
    print(f"false_positive_rate: {result.false_positive_rate:.3f}")
    print(f"failed_case_ids: {','.join(result.failed_case_ids) or 'none'}")
    raise SystemExit(1 if result.failed_case_ids else 0)

