from __future__ import annotations

from pathlib import Path

from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.ingestion.registry import default_loader_registry
from rag_assistant.storage.metadata_repository import MetadataRepository
from rag_assistant.storage.models import DocumentStatus
from rag_assistant.storage.normalized_store import NormalizedDocumentStore


def pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        registry=default_loader_registry(),
        metadata=MetadataRepository(tmp_path / "metadata.sqlite3"),
        normalized_store=NormalizedDocumentStore(tmp_path / "normalized"),
    )


def test_pipeline_persists_normalized_text_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("Evidence on the first line.\nMore evidence.", encoding="utf-8")
    service = pipeline(tmp_path)

    outcome = service.ingest(path)

    assert outcome.duplicate is False
    assert outcome.document.status == DocumentStatus.READY
    assert outcome.document.segment_count == 1
    assert outcome.document.character_count > 0
    assert outcome.document.normalized_path is not None
    assert outcome.document.normalized_path.exists()
    stored = service.normalized_store.read(outcome.document.id)
    assert stored["document_id"] == outcome.document.id
    assert stored["segments"][0]["text"] == "Evidence on the first line.\nMore evidence."


def test_pipeline_is_idempotent_by_file_hash(tmp_path: Path) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "renamed.txt"
    first_path.write_text("identical bytes", encoding="utf-8")
    second_path.write_text("identical bytes", encoding="utf-8")
    service = pipeline(tmp_path)

    first = service.ingest(first_path)
    second = service.ingest(second_path)

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.document.id == first.document.id
    assert len(service.metadata.list_all()) == 1


def test_pipeline_persists_synthetic_noise_flag(tmp_path: Path) -> None:
    path = tmp_path / "noise.html"
    path.write_text(
        "<h1>Generated Noise Content #1</h1>"
        "<p>Experimental observations indicate that systems improve.</p>",
        encoding="utf-8",
    )
    service = pipeline(tmp_path)

    outcome = service.ingest(path)

    assert "synthetic_noise" in outcome.document.quality_flags

