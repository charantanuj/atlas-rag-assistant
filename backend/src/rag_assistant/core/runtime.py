from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.agents.engine import AgentEngine
from rag_assistant.agents.planner import PlannerNode
from rag_assistant.agents.responder import ResponderNode
from rag_assistant.agents.retrieval import RetrievalNode
from rag_assistant.core.config import Settings
from rag_assistant.core.cache import TTLCache
from rag_assistant.embeddings.gemini import GeminiEmbeddingProvider
from rag_assistant.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from rag_assistant.generation.gemini import GeminiAnswerGenerator
from rag_assistant.guardrails.input_guardrails import InputGuardrails
from rag_assistant.guardrails.ml_injection import ModelInjectionDetector
from rag_assistant.guardrails.output_guardrails import OutputGuardrails
from rag_assistant.indexing.document_indexer import DocumentIndexingService
from rag_assistant.indexing.vector_indexer import VectorIndexer
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.processing.reranking import LexicalVectorReranker
from rag_assistant.processing.splitting import RecursiveTextSplitter
from rag_assistant.retrieval.service import RetrievalService
from rag_assistant.retrieval.bm25 import BM25Index
from rag_assistant.storage.normalized_store import NormalizedDocumentStore
from rag_assistant.vectorstores.chroma import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class RagRuntime:
    document_indexer: DocumentIndexingService
    agent_engine: AgentEngine | None


def build_rag_runtime(settings: Settings) -> RagRuntime | None:
    """Build local indexing independently from optional Gemini generation."""
    if settings.embedding_provider == "local":
        embeddings = SentenceTransformerEmbeddingProvider(
            model_name=settings.local_embedding_model,
            cache_folder=settings.local_model_cache,
            device=settings.local_embedding_device,
            batch_size=settings.local_embedding_batch_size,
        )
    else:
        if not settings.gemini_api_key:
            return None
        embeddings = GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
            dimension=768,
        )
    vector_store = ChromaVectorStore(
        persist_path=settings.chroma_path,
        collection_name=f"enterprise-rag-{settings.embedding_provider}-v1",
        embedding_model=embeddings.model_id,
        dimension=embeddings.dimension,
    )
    normalized_store = NormalizedDocumentStore(settings.data_dir / "normalized")
    document_indexer = DocumentIndexingService(
        normalized_store=normalized_store,
        chunker=DocumentChunker(RecursiveTextSplitter()),
        vector_indexer=VectorIndexer(embeddings=embeddings, vector_store=vector_store),
    )
    retrieval = RetrievalService(
        embeddings=embeddings,
        vector_store=vector_store,
        reranker=LexicalVectorReranker(),
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_n=settings.rerank_top_n,
        minimum_score=settings.min_relevance_score,
        keyword_index=BM25Index(vector_store.collection),
        query_cache=TTLCache(
            max_items=settings.query_cache_max_items,
            ttl_seconds=settings.query_cache_ttl_seconds,
        ),
    )
    engine = None
    if settings.gemini_api_key:
        engine = AgentEngine(
            planner=PlannerNode(InputGuardrails(model_detector=(
                ModelInjectionDetector(model_name=settings.injection_model_name, threshold=settings.injection_model_threshold)
                if settings.injection_model_enabled else None
            ))),
            retrieval=RetrievalNode(retrieval),
            responder=ResponderNode(
                generator=GeminiAnswerGenerator(
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_chat_model,
                    thinking_level=settings.gemini_thinking_level,
                ),
                output_guardrails=OutputGuardrails(),
            ),
        )
    return RagRuntime(document_indexer=document_indexer, agent_engine=engine)
