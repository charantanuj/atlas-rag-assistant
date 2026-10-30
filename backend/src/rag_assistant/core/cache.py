from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Small thread-safe in-process LRU cache for hot RAG paths."""

    def __init__(self, *, max_items: int = 256, ttl_seconds: float = 900.0) -> None:
        if max_items < 1 or ttl_seconds <= 0:
            raise ValueError("cache size and TTL must be positive")
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if expires_at <= monotonic():
                del self._items[key]
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = (monotonic() + self.ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"items": len(self._items), "hits": self.hits, "misses": self.misses}
