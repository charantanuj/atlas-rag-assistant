from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_assistant.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformer embeddings with normalized cosine vectors."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_folder: Path = Path("data/models"),
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings require sentence-transformers. Install backend/requirements.txt."
                ) from exc
            cache_folder.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(
                model_name,
                cache_folder=str(cache_folder),
                device=device,
            )
        self.model = model
        dimension_reader = getattr(
            self.model,
            "get_embedding_dimension",
            getattr(self.model, "get_sentence_embedding_dimension", None),
        )
        dimension = dimension_reader() if dimension_reader else None
        if not dimension:
            raise RuntimeError("The local embedding model did not report its dimension")
        self._dimension = int(dimension)

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self._model_name}:{self.dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        raw = values.tolist() if hasattr(values, "tolist") else values
        vectors = [[float(value) for value in vector] for vector in raw]
        if len(vectors) != len(texts) or any(len(item) != self.dimension for item in vectors):
            raise RuntimeError("Local embedding model returned an unexpected shape")
        return vectors
