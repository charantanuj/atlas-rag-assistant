from rag_assistant.agents.planner import PlannerNode
from rag_assistant.agents.models import Intent
from rag_assistant.guardrails.input_guardrails import InputGuardrails


def test_name_question_uses_conversational_path():
    planner = PlannerNode(InputGuardrails())
    assert planner.plan("whats ur name").intent == Intent.CONVERSATIONAL
    assert planner.plan("what is your name?").intent == Intent.CONVERSATIONAL
