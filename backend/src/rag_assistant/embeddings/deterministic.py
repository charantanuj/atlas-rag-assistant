from __future__ import annotations

import math
import re
from hashlib import sha256

from rag_assistant.embeddings.base import EmbeddingProvider

_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Local hashing embeddings for repeatable tests; not a production semantic model."""

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"deterministic-hash-v1-{self.dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

