"""Optional local prompt-injection classifier; no model download until enabled."""
from threading import Lock

from rag_assistant.guardrails.prompt_injection import InjectionFinding


class ModelInjectionDetector:
    def __init__(self, *, model_name: str, threshold: float = 0.9, classifier=None):
        self.model_name = model_name
        self.threshold = threshold
        self._classifier = classifier
        self._lock = Lock()

    def detect(self, text: str) -> tuple[InjectionFinding, ...]:
        with self._lock:
            if self._classifier is None:
                from transformers import pipeline
                self._classifier = pipeline("text-classification", model=self.model_name, device=-1)
            output = self._classifier(text, truncation=True, max_length=512)
        result = output[0]
        label = str(result["label"]).upper()
        if label in {"INJECTION", "LABEL_1"} and float(result["score"]) >= self.threshold:
            return (InjectionFinding(code="ML_PROMPT_INJECTION", pattern="model_classifier"),)
        return ()
