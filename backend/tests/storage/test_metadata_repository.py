from pathlib import Path

from rag_assistant.storage.metadata_repository import MetadataRepository
from rag_assistant.storage.models import DocumentStatus


def test_repository_tracks_processing_to_failure(tmp_path: Path) -> None:
    repository = MetadataRepository(tmp_path / "metadata.sqlite3")
    record = repository.begin(
        document_id="doc_failed",
        original_name="bad.pdf",
        media_type="application/pdf",
        extension="pdf",
        sha256="abc",
        size_bytes=10,
    )

    assert record.status == DocumentStatus.PROCESSING

    failed = repository.mark_failed(record.id, "INVALID_DOCUMENT")

    assert failed.status == DocumentStatus.FAILED
    assert failed.error_code == "INVALID_DOCUMENT"

