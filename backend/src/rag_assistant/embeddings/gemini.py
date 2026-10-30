from __future__ import annotations

from typing import Any

from rag_assistant.embeddings.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding adapter. Construction is local; calls require an API key."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Gemini API key is required")
        self._model = model
        self._dimension = dimension
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client

    @property
    def model_id(self) -> str:
        return f"google:{self._model}:{self.dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            from google.genai import types

            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimension,
            )
        except ImportError:
            config = {
                "task_type": task_type,
                "output_dimensionality": self.dimension,
            }
        response = self.client.models.embed_content(
            model=self._model,
            contents=texts,
            config=config,
        )
        vectors = [list(item.values) for item in response.embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError("Gemini returned a different number of embeddings than inputs")
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError("Gemini returned an unexpected embedding dimension")
        return vectors

