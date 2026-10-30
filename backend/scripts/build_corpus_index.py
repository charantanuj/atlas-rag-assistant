"""Build the persistent local RAG index from an entire document corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rag_assistant.api.service import build_local_application_service
from rag_assistant.core.config import get_settings
from rag_assistant.ingestion.registry import default_loader_registry


def build(source_root: Path) -> int:
    settings = get_settings()
    service = build_local_application_service(settings)
    registry = default_loader_registry()
    counts: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []

    files = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in registry.supported_extensions
    )
    print(f"Corpus: {source_root}")
    print(f"Supported files discovered: {len(files)}")
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}")
        try:
            with path.open("rb") as stream:
                document, duplicate = service.upload(path.name, stream)
            counts["duplicate" if duplicate else document.record.status.value] += 1
            counts["chunks"] += document.chunk_count
            for flag in document.record.quality_flags:
                counts[f"flag:{flag}"] += 1
        except Exception as exc:  # keep building; report every failed file at the end
            code = getattr(exc, "code", exc.__class__.__name__)
            counts["failed"] += 1
            failures.append((path.name, str(code)))
            print(f"  FAILED: {code}")

    print("\nIndex build summary")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    for name, code in failures:
        print(f"failure: {name}: {code}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    raise SystemExit(build(args.source_root))
