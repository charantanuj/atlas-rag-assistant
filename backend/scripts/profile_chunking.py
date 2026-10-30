"""Profile chunk production from normalized JSON without modifying the index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rag_assistant.ingestion.models import (
    LocationType,
    NormalizedSegment,
    QualityFlag,
)
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.processing.splitting import RecursiveTextSplitter


def load_segments(path: Path) -> list[NormalizedSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        NormalizedSegment(
            document_id=item["document_id"],
            source_name=item["source_name"],
            media_type=item["media_type"],
            location_type=LocationType(item["location_type"]),
            location=item["location"],
            text=item["text"],
            content_hash=item["content_hash"],
            heading=item.get("heading"),
            quality_flags=tuple(QualityFlag(flag) for flag in item["quality_flags"]),
        )
        for item in payload["segments"]
    ]


def profile(root: Path, chunk_size: int, overlap: int) -> None:
    chunker = DocumentChunker(RecursiveTextSplitter(chunk_size, overlap))
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for path in sorted(root.glob("*.json")):
        segments = load_segments(path)
        chunks = chunker.chunk_segments(segments)
        counts["documents"] += 1
        counts["segments"] += len(segments)
        counts["chunks"] += len(chunks)
        counts["noise_chunks"] += sum(
            QualityFlag.SYNTHETIC_NOISE in chunk.quality_flags for chunk in chunks
        )
        lengths.extend(len(chunk.text) for chunk in chunks)

    print(f"documents: {counts['documents']}")
    print(f"segments: {counts['segments']}")
    print(f"chunks: {counts['chunks']}")
    print(f"noise_chunks: {counts['noise_chunks']}")
    if lengths:
        print(f"min_chunk_chars: {min(lengths)}")
        print(f"max_chunk_chars: {max(lengths)}")
        print(f"mean_chunk_chars: {sum(lengths) / len(lengths):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("normalized_root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    arguments = parser.parse_args()
    profile(arguments.normalized_root, arguments.chunk_size, arguments.overlap)

