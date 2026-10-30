import math

import pytest

from rag_assistant.embeddings.deterministic import DeterministicEmbeddingProvider


def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimension=32)

    first = provider.embed_query("memory consistency and cache coherence")
    second = provider.embed_query("memory consistency and cache coherence")

    assert first == second
    assert len(first) == 32
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_empty_input_returns_zero_vector() -> None:
    provider = DeterministicEmbeddingProvider(dimension=16)

    assert provider.embed_query("") == [0.0] * 16

