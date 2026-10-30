from __future__ import annotations

from rag_assistant.guardrails.models import GuardrailDecision, GuardrailResult
from rag_assistant.guardrails.pii_detection import SensitiveDataDetector
from rag_assistant.guardrails.prompt_injection import PromptInjectionDetector


class InputGuardrails:
    def __init__(
        self,
        *,
        maximum_characters: int = 4000,
        injection_detector: PromptInjectionDetector | None = None,
        sensitive_data_detector: SensitiveDataDetector | None = None,
        model_detector=None,
    ) -> None:
        self.maximum_characters = maximum_characters
        self.injection_detector = injection_detector or PromptInjectionDetector()
        self.sensitive_data_detector = sensitive_data_detector or SensitiveDataDetector()
        self.model_detector = model_detector

    def evaluate(self, text: str) -> GuardrailResult:
        normalized = " ".join(text.split())
        if not normalized:
            return GuardrailResult(
                GuardrailDecision.BLOCK, ("EMPTY_INPUT",), safe_text=None
            )
        if len(normalized) > self.maximum_characters:
            return GuardrailResult(
                GuardrailDecision.BLOCK, ("INPUT_TOO_LONG",), safe_text=None
            )
        injection = self.injection_detector.detect(normalized)
        if injection:
            return GuardrailResult(
                GuardrailDecision.BLOCK,
                tuple(finding.code for finding in injection),
                safe_text=None,
            )
        if self.model_detector is not None:
            try:
                model_findings = self.model_detector.detect(normalized)
            except Exception:
                return GuardrailResult(GuardrailDecision.BLOCK, ("SAFETY_CLASSIFIER_UNAVAILABLE",), safe_text=None)
            if model_findings:
                return GuardrailResult(GuardrailDecision.BLOCK, tuple(f.code for f in model_findings), safe_text=None)
        sensitive = self.sensitive_data_detector.detect(normalized)
        if sensitive:
            return GuardrailResult(
                GuardrailDecision.SANITIZE,
                tuple(sorted({finding.code for finding in sensitive})),
                safe_text=self.sensitive_data_detector.redact(normalized, sensitive),
            )
        return GuardrailResult(GuardrailDecision.PASS, safe_text=normalized)

