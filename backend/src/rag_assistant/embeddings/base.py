from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable provider/model identifier used to version the vector collection."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one search query using the provider's retrieval-query mode."""

