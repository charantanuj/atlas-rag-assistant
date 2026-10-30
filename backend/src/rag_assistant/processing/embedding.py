"""Compatibility exports for embedding services."""

from rag_assistant.embeddings.base import EmbeddingProvider
from rag_assistant.embeddings.deterministic import DeterministicEmbeddingProvider
from rag_assistant.embeddings.gemini import GeminiEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "DeterministicEmbeddingProvider",
    "GeminiEmbeddingProvider",
]


