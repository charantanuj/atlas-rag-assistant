from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, Protocol

from rag_assistant.agents.engine import AgentEngine
from rag_assistant.agents.models import AgentResponse
from rag_assistant.core.config import Settings
from rag_assistant.core.cache import TTLCache
from rag_assistant.ingestion.pipeline import IngestionOutcome, IngestionPipeline
from rag_assistant.ingestion.registry import default_loader_registry
from rag_assistant.indexing.document_indexer import DocumentIndexingService
from rag_assistant.processing.splitting import RecursiveTextSplitter
from rag_assistant.storage.metadata_repository import MetadataRepository
from rag_assistant.storage.models import DocumentRecord
from rag_assistant.storage.normalized_store import NormalizedDocumentStore


class ApplicationError(Exception):
    def __init__(self, code: str, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.code, self.detail, self.status_code = code, detail, status_code


@dataclass(frozen=True, slots=True)
class DocumentView:
    record: DocumentRecord
    chunk_count: int


class ApplicationService(Protocol):
    def list_documents(self) -> list[DocumentView]: ...
    def upload(self, filename: str, stream: BinaryIO) -> tuple[DocumentView, bool]: ...
    def ask(self, question: str, document_ids: tuple[str, ...]) -> AgentResponse: ...
    def evaluation_summary(self) -> dict: ...
    def cache_summary(self) -> dict: ...


class LocalApplicationService:
    def __init__(
        self,
        settings: Settings,
        *,
        agent_engine: AgentEngine | None = None,
        document_indexer: DocumentIndexingService | None = None,
    ) -> None:
        self.settings = settings
        self.metadata = MetadataRepository(settings.sqlite_path)
        self.normalized = NormalizedDocumentStore(settings.data_dir / "normalized")
        self.pipeline = IngestionPipeline(
            registry=default_loader_registry(), metadata=self.metadata,
            normalized_store=self.normalized,
        )
        self.splitter = RecursiveTextSplitter()
        self.agent_engine = agent_engine
        self.document_indexer = document_indexer
        self.answer_cache: TTLCache[AgentResponse] = TTLCache(
            max_items=settings.answer_cache_max_items,
            ttl_seconds=settings.answer_cache_ttl_seconds,
        )

    def list_documents(self) -> list[DocumentView]:
        return [self._view(item) for item in self.metadata.list_all()]

    def upload(self, filename: str, stream: BinaryIO) -> tuple[DocumentView, bool]:
        safe_name = Path(filename or "").name
        extension = Path(safe_name).suffix.lower().lstrip(".")
        if not safe_name or extension not in self.settings.allowed_extension_set:
            raise ApplicationError("UNSUPPORTED_FORMAT", "The document format is not supported.", 415)
        total = 0
        staging_root = self.settings.data_dir / ".uploads"
        staging_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="rag-upload-", dir=staging_root) as temporary_directory:
            # Keep the staging path short on Windows; preserve the real name in metadata.
            temporary_path = Path(temporary_directory) / f"upload.{extension}"
            with temporary_path.open("wb") as temporary:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.settings.max_upload_bytes:
                        raise ApplicationError("FILE_TOO_LARGE", "The document exceeds the upload limit.", 413)
                    temporary.write(chunk)
            if total == 0:
                raise ApplicationError("EMPTY_FILE", "The uploaded document is empty.", 422)
            outcome: IngestionOutcome = self.pipeline.ingest(temporary_path, display_name=safe_name)
            self.answer_cache.clear()
            if not outcome.duplicate:
                if self.document_indexer is None:
                    failed = self.metadata.mark_failed(outcome.document.id, "INDEX_NOT_CONFIGURED")
                    raise ApplicationError(
                        "INDEX_NOT_CONFIGURED",
                        "Document parsing succeeded, but embedding and vector indexing are not configured.",
                        503,
                    )
                try:
                    self.document_indexer.index_document(outcome.document.id)
                except Exception as exc:
                    self.metadata.mark_failed(outcome.document.id, "INDEXING_FAILED")
                    raise ApplicationError(
                        "INDEXING_FAILED",
                        "Document parsing succeeded, but vector indexing failed.",
                        502,
                    ) from exc
            return self._view(outcome.document), outcome.duplicate

    def ask(self, question: str, document_ids: tuple[str, ...]) -> AgentResponse:
        if self.agent_engine is None:
            raise ApplicationError(
                "MODEL_NOT_CONFIGURED",
                "Question answering is unavailable until the embedding index and Gemini API key are configured.",
                503,
            )
        # Preserve exact input: case-sensitive technical queries may have different meanings.
        key = repr((question.strip(), tuple(sorted(document_ids))))
        cached = self.answer_cache.get(key)
        if cached is not None:
            return cached
        response = self.agent_engine.run(question, document_ids=document_ids)
        if response.status.value in {"answered", "conversational"}:
            self.answer_cache.set(key, response)
        return response

    def evaluation_summary(self) -> dict:
        return {
            "passed": False, "recall_at_k": 0.0, "reciprocal_rank": 0.0,
            "citation_precision": 0.0, "citation_completeness": 0.0,
            "abstention_accuracy": 0.0, "attack_block_rate": 1.0,
            "false_positive_rate": 0.0, "last_run_at": None,
            "failed_gates": ["CORPUS_EVALUATION_NOT_RUN"],
        }

    def cache_summary(self) -> dict:
        return {"answer": self.answer_cache.stats()}

    def _view(self, record: DocumentRecord) -> DocumentView:
        count = 0
        if record.normalized_path and record.normalized_path.exists():
            payload = self.normalized.read(record.id)
            count = sum(len(self.splitter.split(item["text"])) for item in payload["segments"])
        return DocumentView(record=record, chunk_count=count)


def build_local_application_service(settings: Settings) -> LocalApplicationService:
    from rag_assistant.core.runtime import build_rag_runtime

    runtime = build_rag_runtime(settings)
    return LocalApplicationService(
        settings,
        agent_engine=runtime.agent_engine if runtime else None,
        document_indexer=runtime.document_indexer if runtime else None,
    )
