from rag_assistant.guardrails.input_guardrails import InputGuardrails
from rag_assistant.guardrails.ml_injection import ModelInjectionDetector
from rag_assistant.guardrails.models import GuardrailDecision


def test_model_blocks_high_confidence_attack():
    detector = ModelInjectionDetector(model_name="unused", classifier=lambda text, **kwargs: [{"label": "INJECTION", "score": 0.98}])
    result = InputGuardrails(model_detector=detector).evaluate("A disguised attack")
    assert result.decision == GuardrailDecision.BLOCK
    assert "ML_PROMPT_INJECTION" in result.reason_codes


def test_model_failure_blocks_safely():
    def failure(*args, **kwargs):
        raise RuntimeError("unavailable")
    detector = ModelInjectionDetector(model_name="unused", classifier=failure)
    assert InputGuardrails(model_detector=detector).evaluate("A normal request").reason_codes == ("SAFETY_CLASSIFIER_UNAVAILABLE",)
