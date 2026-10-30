from rag_assistant.ingestion.base import content_hash
from rag_assistant.ingestion.models import (
    LocationType,
    NormalizedSegment,
    QualityFlag,
)
from rag_assistant.ingestion.quality import QualityClassifier


def segment(text: str, location: int = 1) -> NormalizedSegment:
    return NormalizedSegment(
        document_id="doc_test",
        source_name="source.txt",
        media_type="text/plain",
        location_type=LocationType.LINE_RANGE,
        location=location,
        text=text,
        content_hash=content_hash(text),
    )


def test_quality_classifier_marks_synthetic_noise() -> None:
    value = segment(
        "Experimental observations indicate that cache behavior improves "
        "across heterogeneous runtime environments."
    )

    result = QualityClassifier().classify([value])[0]

    assert QualityFlag.SYNTHETIC_NOISE in result.quality_flags


def test_quality_classifier_marks_repeated_segments() -> None:
    first = segment("same text", 1)
    second = segment("same text", 2)

    results = QualityClassifier().classify([first, second])

    assert all(QualityFlag.POSSIBLE_DUPLICATE in item.quality_flags for item in results)

