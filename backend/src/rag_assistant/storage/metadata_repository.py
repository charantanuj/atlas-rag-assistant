from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from rag_assistant.storage.models import DocumentRecord, DocumentStatus


class MetadataRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                    status TEXT NOT NULL CHECK(status IN ('processing', 'ready', 'failed')),
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    character_count INTEGER NOT NULL DEFAULT 0,
                    normalized_path TEXT,
                    quality_flags TEXT NOT NULL DEFAULT '[]',
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
            )

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._to_record(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at, id"
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def begin(
        self,
        *,
        document_id: str,
        original_name: str,
        media_type: str,
        extension: str,
        sha256: str,
        size_bytes: int,
    ) -> DocumentRecord:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, original_name, media_type, extension, sha256, size_bytes,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    original_name,
                    media_type,
                    extension,
                    sha256,
                    size_bytes,
                    DocumentStatus.PROCESSING.value,
                    now,
                    now,
                ),
            )
        record = self.get(document_id)
        assert record is not None
        return record

    def mark_ready(
        self,
        document_id: str,
        *,
        segment_count: int,
        character_count: int,
        normalized_path: Path,
        quality_flags: tuple[str, ...],
    ) -> DocumentRecord:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, segment_count = ?, character_count = ?,
                    normalized_path = ?, quality_flags = ?, error_code = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    DocumentStatus.READY.value,
                    segment_count,
                    character_count,
                    str(normalized_path),
                    json.dumps(quality_flags),
                    datetime.now(UTC).isoformat(),
                    document_id,
                ),
            )
        record = self.get(document_id)
        assert record is not None
        return record

    def mark_failed(self, document_id: str, error_code: str) -> DocumentRecord:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, error_code = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    DocumentStatus.FAILED.value,
                    error_code,
                    datetime.now(UTC).isoformat(),
                    document_id,
                ),
            )
        record = self.get(document_id)
        assert record is not None
        return record

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    @staticmethod
    def _to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            original_name=row["original_name"],
            media_type=row["media_type"],
            extension=row["extension"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            status=DocumentStatus(row["status"]),
            segment_count=row["segment_count"],
            character_count=row["character_count"],
            normalized_path=Path(row["normalized_path"]) if row["normalized_path"] else None,
            quality_flags=tuple(json.loads(row["quality_flags"])),
            error_code=row["error_code"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

