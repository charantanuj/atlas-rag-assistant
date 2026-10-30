"""Local BM25 retrieval over the chunks already stored in Chroma."""
from collections import Counter
import math
import re
from threading import RLock

from rag_assistant.vectorstores.models import VectorSearchResult

TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)


class BM25Index:
    def __init__(self, collection):
        self.collection = collection
        self._lock = RLock()
        self._count = -1
        self._rows = []
        self._terms = []
        self._frequencies = Counter()
        self._average_length = 0.0

    def invalidate(self):
        with self._lock:
            self._count = -1

    def _refresh(self):
        count = self.collection.count()
        if count == self._count:
            return
        rows = []
        for offset in range(0, count, 1000):
            result = self.collection.get(limit=1000, offset=offset, include=["documents", "metadatas"])
            rows.extend(zip(result["ids"], result["documents"], result["metadatas"]))
        self._rows = rows
        self._terms = [Counter(TOKEN.findall((text + " " + str(meta.get("source_name", ""))).lower())) for _, text, meta in rows]
        self._frequencies = Counter(term for terms in self._terms for term in terms)
        self._average_length = sum(sum(terms.values()) for terms in self._terms) / max(len(rows), 1)
        self._count = count

    def search(self, query, *, top_k, document_ids=()):
        with self._lock:
            self._refresh()
            terms = set(TOKEN.findall(query.lower()))
            scored = []
            for row, frequencies in zip(self._rows, self._terms):
                chunk_id, text, metadata = row
                if document_ids and metadata.get("document_id") not in document_ids:
                    continue
                length = sum(frequencies.values())
                score = 0.0
                for term in terms:
                    frequency = frequencies.get(term, 0)
                    if frequency:
                        idf = math.log(1 + (len(self._rows) - self._frequencies[term] + 0.5) / (self._frequencies[term] + 0.5))
                        score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / max(self._average_length, 1)))
                if score > 0:
                    scored.append((score, chunk_id, text, metadata))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [VectorSearchResult(chunk_id=chunk_id, text=text, metadata=metadata, distance=1.0, relevance=0.0)
                    for _, chunk_id, text, metadata in scored[:top_k]]


def reciprocal_rank_fusion(semantic, lexical, *, k=60):
    scores, items = {}, {}
    for ranking in (semantic, lexical):
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (k + rank)
            items.setdefault(item.chunk_id, item)
    return [items[item] for item in sorted(scores, key=lambda item: (-scores[item], item))]
