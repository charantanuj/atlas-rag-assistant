import pytest

from rag_assistant.processing.splitting import RecursiveTextSplitter


def test_short_text_stays_in_one_split() -> None:
    splits = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20).split("Short text.")

    assert len(splits) == 1
    assert splits[0].text == "Short text."
    assert (splits[0].start_char, splits[0].end_char) == (0, 11)


def test_long_text_is_bounded_and_overlapping() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    splitter = RecursiveTextSplitter(chunk_size=120, chunk_overlap=25)

    splits = splitter.split(text)

    assert len(splits) > 1
    assert all(len(item.text) <= 120 for item in splits)
    assert all(item.start_char < item.end_char for item in splits)
    assert all(
        current.start_char < previous.end_char
        for previous, current in zip(splits, splits[1:])
    )
    assert "word0" in splits[0].text
    assert "word99" in splits[-1].text


def test_splitter_prefers_paragraph_boundary() -> None:
    first = "A" * 70
    second = "B" * 70
    splits = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10).split(
        f"{first}\n\n{second}"
    )

    assert splits[0].text == first


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_splitter_configuration_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=size, chunk_overlap=overlap)


def test_empty_text_produces_no_chunks() -> None:
    assert RecursiveTextSplitter().split("  \n") == []

