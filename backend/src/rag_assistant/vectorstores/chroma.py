from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag_assistant.processing.chunking import DocumentChunk
from rag_assistant.vectorstores.models import VectorSearchResult


class CollectionVersionError(RuntimeError):
    pass


class ChromaVectorStore:
    def __init__(
        self,
        *,
        persist_path: Path,
        collection_name: str,
        embedding_model: str,
        dimension: int,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is not installed. Install backend/requirements.txt."
                ) from exc
            persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_path))
        self.client = client
        self.embedding_model = embedding_model
        self.dimension = dimension
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", collection_name)
        metadata = {
            "embedding_model": embedding_model,
            "dimension": dimension,
            "hnsw:space": "cosine",
        }
        self.collection = client.get_or_create_collection(name=safe_name, metadata=metadata)
        existing = self.collection.metadata or {}
        if (
            existing.get("embedding_model") != embedding_model
            or int(existing.get("dimension", -1)) != dimension
        ):
            raise CollectionVersionError(
                "Existing collection uses a different embedding model or dimension."
            )

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        if any(len(vector) != self.dimension for vector in embeddings):
            raise ValueError("Embedding dimension does not match the collection")
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict | None = None,
    ) -> list[VectorSearchResult]:
        if len(query_embedding) != self.dimension:
            raise ValueError("Query embedding dimension does not match the collection")
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        return [
            VectorSearchResult(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
                distance=float(distance),
                relevance=max(0.0, min(1.0, 1.0 - float(distance))),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
        return {
            "document_id": chunk.document_id,
            "source_name": chunk.source_name,
            "media_type": chunk.media_type,
            "location_type": chunk.location_type.value,
            "location": chunk.location,
            "heading": chunk.heading or "",
            "chunk_index": chunk.chunk_index,
            "text_hash": chunk.text_hash,
            "source_content_hash": chunk.source_content_hash,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "quality_flags": "|".join(flag.value for flag in chunk.quality_flags),
        }

