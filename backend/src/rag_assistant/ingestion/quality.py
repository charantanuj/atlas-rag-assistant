from __future__ import annotations

from collections import Counter
from dataclasses import replace

from rag_assistant.ingestion.models import NormalizedSegment, QualityFlag

_SYNTHETIC_MARKERS = (
    "generated noise content",
    "across heterogeneous runtime environments",
    "experimental observations indicate that",
    "researchers also noted side effects involving",
)


class QualityClassifier:
    def classify(self, segments: list[NormalizedSegment]) -> list[NormalizedSegment]:
        hashes = Counter(segment.content_hash for segment in segments)
        classified = []
        for segment in segments:
            flags = set(segment.quality_flags)
            lowered = segment.text.lower()
            if not segment.text.strip():
                flags.add(QualityFlag.EMPTY)
            if any(marker in lowered for marker in _SYNTHETIC_MARKERS):
                flags.add(QualityFlag.SYNTHETIC_NOISE)
            if hashes[segment.content_hash] > 1:
                flags.add(QualityFlag.POSSIBLE_DUPLICATE)
            classified.append(
                replace(segment, quality_flags=tuple(sorted(flags, key=lambda item: item.value)))
            )
        return classified

