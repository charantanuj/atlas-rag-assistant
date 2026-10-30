from __future__ import annotations

from rag_assistant.agents.models import AgentResponse, AnswerStatus, Intent
from rag_assistant.agents.planner import PlannerNode
from rag_assistant.agents.responder import ResponderNode
from rag_assistant.agents.retrieval import RetrievalNode
from rag_assistant.observability.tracing import TraceRecorder
import logging

logger = logging.getLogger("uvicorn.error")


class AgentEngine:
    """Explicit three-node flow; suitable for later LangGraph wrapping."""

    def __init__(
        self,
        *,
        planner: PlannerNode,
        retrieval: RetrievalNode,
        responder: ResponderNode,
    ) -> None:
        self.planner = planner
        self.retrieval = retrieval
        self.responder = responder

    def run(
        self,
        question: str,
        *,
        document_ids: tuple[str, ...] = (),
        trace: TraceRecorder | None = None,
    ) -> AgentResponse:
        recorder = trace or TraceRecorder("req_untracked", sink=lambda event: logger.info(
            "rag_stage stage=%s duration_ms=%.2f status=%s", event.stage, event.duration_ms, event.status
        ))
        with recorder.span("planner") as attributes:
            plan = self.planner.plan(question)
            attributes["intent"] = plan.intent.value
            attributes["guardrail_decision"] = plan.guardrail_decision.value
        if plan.intent == Intent.BLOCKED:
            return AgentResponse(
                status=AnswerStatus.BLOCKED,
                answer="The request was blocked by the input safety policy.",
                reason_codes=plan.reason_codes,
            )
        if plan.intent == Intent.CONVERSATIONAL:
            return AgentResponse(
                status=AnswerStatus.CONVERSATIONAL,
                answer=(
                    "I'm Atlas. I can answer questions about the uploaded documents and provide "
                    "document, page, or slide citations."
                ),
                reason_codes=plan.reason_codes,
            )
        assert plan.safe_question is not None
        with recorder.span("retrieval") as attributes:
            retrieval = self.retrieval.retrieve(
                plan.safe_question, document_ids=document_ids
            )
            attributes["evidence_status"] = retrieval.status.value
            attributes["candidates_retrieved"] = retrieval.candidates_retrieved
            attributes["evidence_count"] = len(retrieval.evidence)
        with recorder.span("responder") as attributes:
            response = self.responder.respond(plan.safe_question, retrieval)
            attributes["answer_status"] = response.status.value
            attributes["citation_count"] = len(response.citations)
        return response


