from __future__ import annotations

import re
from dataclasses import dataclass

from rag_assistant.agents.models import AgentResponse, AnswerStatus

_FACTUAL_SENTENCE = re.compile(r"[^.!?\n]{20,}[.!?](?:\s|$)")
_CITATION = re.compile(r"\[\d+]" )


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    abstention_correct: float
    citation_coverage: float
    safe_status: float


def evaluate_generation(
    response: AgentResponse,
    *,
    should_abstain: bool,
) -> GenerationMetrics:
    did_abstain = response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    abstention_correct = float(did_abstain == should_abstain)
    sentences = _FACTUAL_SENTENCE.findall(response.answer)
    coverage = (
        sum(bool(_CITATION.search(sentence)) for sentence in sentences) / len(sentences)
        if sentences and response.status == AnswerStatus.ANSWERED
        else 1.0
    )
    safe_status = float(
        response.status
        in {
            AnswerStatus.ANSWERED,
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            AnswerStatus.BLOCKED,
            AnswerStatus.CONVERSATIONAL,
        }
    )
    return GenerationMetrics(abstention_correct, coverage, safe_status)

