"""Read-only loader smoke test for a local corpus."""

from __future__ import annotations

import argparse
import mimetypes
from collections import Counter
from pathlib import Path
from uuid import uuid4

from rag_assistant.ingestion.base import file_hash
from rag_assistant.ingestion.errors import IngestionError
from rag_assistant.ingestion.models import SourceFile
from rag_assistant.ingestion.registry import default_loader_registry


def profile(root: Path, skip_pdf: bool) -> int:
    registry = default_loader_registry()
    totals: Counter[str] = Counter()
    failures: list[tuple[str, str, str]] = []

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        extension = path.suffix.lower().lstrip(".")
        if extension not in registry.supported_extensions or (skip_pdf and extension == "pdf"):
            continue
        try:
            loader = registry.for_path(path)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            source = SourceFile(
                path=path,
                document_id=f"doc_{uuid4().hex}",
                display_name=path.name,
                media_type=media_type,
                sha256=file_hash(path),
            )
            segments = loader.load(source)
            totals["files"] += 1
            totals["segments"] += len(segments)
            totals["characters"] += sum(len(segment.text) for segment in segments)
            totals[f"format:{extension}"] += 1
        except (IngestionError, OSError) as exc:
            failures.append((path.name, getattr(exc, "code", "OS_ERROR"), str(exc)))

    print(f"Files processed: {totals['files']}")
    print(f"Segments extracted: {totals['segments']}")
    print(f"Characters extracted: {totals['characters']}")
    for key, value in sorted(totals.items()):
        if key.startswith("format:"):
            print(f"{key}: {value}")
    print(f"Failures: {len(failures)}")
    for name, code, detail in failures:
        print(f"- {name}: {code}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    raise SystemExit(profile(args.root, args.skip_pdf))

