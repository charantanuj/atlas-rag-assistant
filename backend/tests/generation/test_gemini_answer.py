from types import SimpleNamespace

from rag_assistant.generation.gemini import GeminiAnswerGenerator
from rag_assistant.processing.reranking import RankedEvidence


class FakeModels:
    def __init__(self):
        self.call = None

    def generate_content(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(text="Grounded answer [1].")


def test_gemini_answer_adapter_uses_grounded_prompt() -> None:
    models = FakeModels()
    generator = GeminiAnswerGenerator(
        api_key="unused",
        model="test-model",
        client=SimpleNamespace(models=models),
    )
    evidence = RankedEvidence(
        chunk_id="chk_1",
        text="Evidence text",
        metadata={"source_name": "source.pdf", "location_type": "page", "location": 1},
        vector_score=0.8,
        reranker_score=0.8,
        final_score=0.8,
    )

    answer = generator.generate("Question?", (evidence,))

    assert answer == "Grounded answer [1]."
    assert models.call["model"] == "test-model"
    assert "Evidence text" in models.call["contents"]
    assert models.call["config"]["temperature"] == 0.0

