import time

from rag_assistant.core.cache import TTLCache


def test_cache_expires_entries():
    cache = TTLCache[str](max_items=2, ttl_seconds=0.01)
    cache.set("question", "answer")
    assert cache.get("question") == "answer"
    time.sleep(0.02)
    assert cache.get("question") is None


def test_cache_evicts_least_recently_used_entry():
    cache = TTLCache[int](max_items=2, ttl_seconds=10)
    cache.set("one", 1)
    cache.set("two", 2)
    assert cache.get("one") == 1
    cache.set("three", 3)
    assert cache.get("two") is None
    assert len(cache) == 2


def test_cache_clear_removes_entries():
    cache = TTLCache[str](max_items=2, ttl_seconds=10)
    cache.set("question", "answer")
    cache.clear()
    assert cache.get("question") is None
    assert len(cache) == 0
