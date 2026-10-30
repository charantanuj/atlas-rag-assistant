"""Ingest a local corpus into normalized JSON and SQLite metadata."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rag_assistant.ingestion.errors import IngestionError
from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.ingestion.registry import default_loader_registry
from rag_assistant.storage.metadata_repository import MetadataRepository
from rag_assistant.storage.normalized_store import NormalizedDocumentStore


def ingest_corpus(source_root: Path, output_root: Path, skip_pdf: bool) -> int:
    registry = default_loader_registry()
    pipeline = IngestionPipeline(
        registry=registry,
        metadata=MetadataRepository(output_root / "metadata.sqlite3"),
        normalized_store=NormalizedDocumentStore(output_root / "normalized"),
    )
    counts: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []

    for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
        extension = path.suffix.lower().lstrip(".")
        if extension not in registry.supported_extensions or (skip_pdf and extension == "pdf"):
            continue
        try:
            outcome = pipeline.ingest(path)
            counts["duplicate" if outcome.duplicate else "ingested"] += 1
            counts[f"status:{outcome.document.status.value}"] += 1
            for flag in outcome.document.quality_flags:
                counts[f"flag:{flag}"] += 1
        except (IngestionError, OSError) as exc:
            counts["failed"] += 1
            failures.append((path.name, getattr(exc, "code", "OS_ERROR")))

    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    for name, code in failures:
        print(f"failure: {name}: {code}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--skip-pdf", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(
        ingest_corpus(arguments.source_root, arguments.output_root, arguments.skip_pdf)
    )

