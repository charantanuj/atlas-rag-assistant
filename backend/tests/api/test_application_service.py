from io import BytesIO

import pytest

from rag_assistant.api.service import ApplicationError, LocalApplicationService
from rag_assistant.core.config import Settings


class RecordingIndexer:
    def __init__(self) -> None:
        self.document_ids = []

    def index_document(self, document_id):
        self.document_ids.append(document_id)


def service(tmp_path, indexer=None) -> LocalApplicationService:
    return LocalApplicationService(
        Settings(
            _env_file=None,
            data_dir=tmp_path / "data",
            sqlite_path=tmp_path / "metadata.sqlite3",
            chroma_path=tmp_path / "chroma",
        ),
        document_indexer=indexer or RecordingIndexer(),
    )


def test_real_upload_to_list_workflow(tmp_path) -> None:
    indexer = RecordingIndexer()
    application = service(tmp_path, indexer)
    view, duplicate = application.upload(
        "retention.txt", BytesIO(("Records must be retained for seven years. " * 80).encode()),
    )

    assert duplicate is False
    assert view.record.status.value == "ready"
    assert view.record.segment_count == 1
    assert view.chunk_count > 1
    assert indexer.document_ids == [view.record.id]
    assert application.list_documents()[0].record.original_name == "retention.txt"


def test_real_upload_rejects_unsupported_format(tmp_path) -> None:
    with pytest.raises(ApplicationError) as raised:
        service(tmp_path).upload("malware.exe", BytesIO(b"not accepted"))
    assert raised.value.code == "UNSUPPORTED_FORMAT"
    assert raised.value.status_code == 415
