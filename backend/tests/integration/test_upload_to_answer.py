from __future__ import annotations

from io import BytesIO

from rag_assistant.agents.engine import AgentEngine
from rag_assistant.agents.models import AnswerStatus
from rag_assistant.agents.planner import PlannerNode
from rag_assistant.agents.responder import ResponderNode
from rag_assistant.agents.retrieval import RetrievalNode
from rag_assistant.api.service import LocalApplicationService
from rag_assistant.core.config import Settings
from rag_assistant.embeddings.deterministic import DeterministicEmbeddingProvider
from rag_assistant.guardrails.input_guardrails import InputGuardrails
from rag_assistant.guardrails.output_guardrails import OutputGuardrails
from rag_assistant.indexing.document_indexer import DocumentIndexingService
from rag_assistant.indexing.vector_indexer import VectorIndexer
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.processing.reranking import LexicalVectorReranker
from rag_assistant.processing.splitting import RecursiveTextSplitter
from rag_assistant.retrieval.service import RetrievalService
from rag_assistant.storage.normalized_store import NormalizedDocumentStore
from rag_assistant.vectorstores.models import VectorSearchResult


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def upsert(self, chunks, embeddings):
        for chunk, vector in zip(chunks, embeddings):
            self.rows.append((chunk, vector))

    def query(self, query_embedding, *, top_k, where=None):
        results = []
        allowed = set(where["document_id"]["$in"]) if where else None
        for chunk, vector in self.rows:
            if allowed is not None and chunk.document_id not in allowed:
                continue
            score = sum(left * right for left, right in zip(query_embedding, vector))
            score = max(0.0, min(1.0, score))
            results.append(VectorSearchResult(
                chunk_id=chunk.id, text=chunk.text,
                metadata={
                    "document_id": chunk.document_id, "source_name": chunk.source_name,
                    "location_type": chunk.location_type.value, "location": chunk.location,
                    "quality_flags": "|".join(flag.value for flag in chunk.quality_flags),
                },
                distance=1.0 - score, relevance=score,
            ))
        return sorted(results, key=lambda item: item.distance)[:top_k]


class GroundedGenerator:
    model_id = "test-grounded-generator"

    def generate(self, question, evidence):
        assert "seven years" in evidence[0].text.lower()
        return "Financial records must be retained for seven years [1]."


def test_complete_local_upload_to_cited_answer_workflow(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "metadata.sqlite3", chroma_path=tmp_path / "chroma",
    )
    embeddings = DeterministicEmbeddingProvider(128)
    vectors = InMemoryVectorStore()
    document_indexer = DocumentIndexingService(
        normalized_store=NormalizedDocumentStore(settings.data_dir / "normalized"),
        chunker=DocumentChunker(RecursiveTextSplitter()),
        vector_indexer=VectorIndexer(embeddings=embeddings, vector_store=vectors),
    )
    retrieval = RetrievalService(
        embeddings=embeddings, vector_store=vectors,
        reranker=LexicalVectorReranker(), retrieval_top_k=5,
        rerank_top_n=3, minimum_score=0.1,
    )
    agent = AgentEngine(
        planner=PlannerNode(InputGuardrails()),
        retrieval=RetrievalNode(retrieval),
        responder=ResponderNode(generator=GroundedGenerator(), output_guardrails=OutputGuardrails()),
    )
    application = LocalApplicationService(
        settings, agent_engine=agent, document_indexer=document_indexer,
    )

    document, duplicate = application.upload(
        "retention-policy.txt",
        BytesIO(b"The corporate policy requires financial records to be retained for seven years. Disposal requires compliance approval."),
    )
    response = application.ask(
        "How long must financial records be retained?", (document.record.id,),
    )

    assert duplicate is False
    assert document.chunk_count == 1
    assert response.status == AnswerStatus.ANSWERED
    assert response.answer.endswith("[1].")
    assert response.citations[0].document_name == "retention-policy.txt"
    assert response.citations[0].chunk_id.startswith("chk_")
    assert response.evidence_used == 1
