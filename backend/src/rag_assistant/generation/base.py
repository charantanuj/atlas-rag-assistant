from __future__ import annotations

from abc import ABC, abstractmethod

from rag_assistant.processing.reranking import RankedEvidence


class AnswerGenerator(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def generate(self, question: str, evidence: tuple[RankedEvidence, ...]) -> str:
        """Generate answer prose using only the supplied evidence."""

