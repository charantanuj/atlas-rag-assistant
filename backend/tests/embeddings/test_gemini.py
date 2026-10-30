from types import SimpleNamespace

from rag_assistant.embeddings.gemini import GeminiEmbeddingProvider


class FakeModels:
    def __init__(self) -> None:
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        values = [0.1, 0.2, 0.3]
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=values) for _ in kwargs["contents"]]
        )


def test_gemini_adapter_separates_document_and_query_tasks() -> None:
    models = FakeModels()
    provider = GeminiEmbeddingProvider(
        api_key="unused",
        model="test-model",
        dimension=3,
        client=SimpleNamespace(models=models),
    )

    documents = provider.embed_documents(["one", "two"])
    query = provider.embed_query("question")

    assert documents == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert query == [0.1, 0.2, 0.3]
    assert len(models.calls) == 2
    assert models.calls[0]["contents"] == ["one", "two"]
    assert models.calls[1]["contents"] == ["question"]

