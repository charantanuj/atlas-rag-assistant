from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_assistant.ingestion.models import NormalizedSegment


class NormalizedDocumentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, document_id: str, segments: list[NormalizedSegment]) -> Path:
        target = self.root / f"{document_id}.json"
        temporary = self.root / f".{document_id}.tmp"
        payload = {
            "schema_version": 1,
            "document_id": document_id,
            "segments": [self._serialize(item) for item in segments],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    def read(self, document_id: str) -> dict:
        return json.loads((self.root / f"{document_id}.json").read_text(encoding="utf-8"))

    @staticmethod
    def _serialize(segment: NormalizedSegment) -> dict:
        value = asdict(segment)
        value["location_type"] = segment.location_type.value
        value["quality_flags"] = [flag.value for flag in segment.quality_flags]
        return value

