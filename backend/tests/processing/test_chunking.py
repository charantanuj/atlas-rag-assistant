from rag_assistant.ingestion.base import content_hash
from rag_assistant.ingestion.models import (
    LocationType,
    NormalizedSegment,
    QualityFlag,
)
from rag_assistant.processing.chunking import DocumentChunker
from rag_assistant.processing.splitting import RecursiveTextSplitter


def source_segment(text: str, location: int = 7) -> NormalizedSegment:
    return NormalizedSegment(
        document_id="doc_policy",
        source_name="policy.pdf",
        media_type="application/pdf",
        location_type=LocationType.PAGE,
        location=location,
        text=text,
        content_hash=content_hash(text),
        heading="Approval rules",
        quality_flags=(QualityFlag.SYNTHETIC_NOISE,),
    )


def test_chunker_preserves_citation_and_quality_metadata() -> None:
    segment = source_segment("Evidence " * 40)
    chunker = DocumentChunker(RecursiveTextSplitter(chunk_size=100, chunk_overlap=20))

    chunks = chunker.chunk_segments([segment])

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc_policy" for chunk in chunks)
    assert all(chunk.source_name == "policy.pdf" for chunk in chunks)
    assert all(chunk.location_type == LocationType.PAGE for chunk in chunks)
    assert all(chunk.location == 7 for chunk in chunks)
    assert all(chunk.heading == "Approval rules" for chunk in chunks)
    assert all(QualityFlag.SYNTHETIC_NOISE in chunk.quality_flags for chunk in chunks)


def test_chunk_ids_are_deterministic() -> None:
    segment = source_segment("Deterministic content " * 20)
    chunker = DocumentChunker(RecursiveTextSplitter(chunk_size=90, chunk_overlap=15))

    first = chunker.chunk_segments([segment])
    second = chunker.chunk_segments([segment])

    assert [item.id for item in first] == [item.id for item in second]
    assert len({item.id for item in first}) == len(first)


def test_chunker_never_combines_locations() -> None:
    first = source_segment("First page text", location=1)
    second = source_segment("Second page text", location=2)

    chunks = DocumentChunker(RecursiveTextSplitter()).chunk_segments([first, second])

    assert [(item.location, item.text) for item in chunks] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]

