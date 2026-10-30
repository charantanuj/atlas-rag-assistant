from __future__ import annotations

from typing import Any

from rag_assistant.generation.base import AnswerGenerator
from rag_assistant.generation.prompt import build_grounded_prompt
from rag_assistant.processing.reranking import RankedEvidence


class GeminiAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        thinking_level: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Gemini API key is required")
        self._model = model
        self.thinking_level = thinking_level
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client

    @property
    def model_id(self) -> str:
        return f"google:{self._model}"

    def generate(self, question: str, evidence: tuple[RankedEvidence, ...]) -> str:
        prompt = build_grounded_prompt(question, evidence)
        config = {"temperature": 0.0, "max_output_tokens": 800}
        if self.thinking_level:
            config["thinking_config"] = {"thinking_level": self.thinking_level}
        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty answer")
        return text

