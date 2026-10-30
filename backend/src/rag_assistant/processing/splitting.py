"""Text-boundary splitting primitives independent of document metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextSplit:
    index: int
    text: str
    start_char: int
    end_char: int


class RecursiveTextSplitter:
    """Character-bounded splitter that prefers semantic text boundaries."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " "),
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split(self, text: str) -> list[TextSplit]:
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [TextSplit(index=0, text=text.strip(), start_char=0, end_char=len(text))]

        results: list[TextSplit] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            hard_end = min(start + self.chunk_size, text_length)
            end = self._preferred_end(text, start, hard_end)
            if end <= start:
                end = hard_end

            raw = text[start:end]
            stripped = raw.strip()
            if stripped:
                leading = len(raw) - len(raw.lstrip())
                trailing = len(raw.rstrip())
                actual_start = start + leading
                actual_end = start + trailing
                results.append(
                    TextSplit(
                        index=len(results),
                        text=stripped,
                        start_char=actual_start,
                        end_char=actual_end,
                    )
                )
            if end >= text_length:
                break

            next_start = max(start + 1, end - self.chunk_overlap)
            while next_start < end and text[next_start].isspace():
                next_start += 1
            start = next_start
        return results

    def _preferred_end(self, text: str, start: int, hard_end: int) -> int:
        if hard_end == len(text):
            return hard_end
        minimum = start + int(self.chunk_size * 0.55)
        window = text[start:hard_end]
        for separator in self.separators:
            position = window.rfind(separator)
            if position >= 0:
                candidate = start + position + len(separator)
                if candidate >= minimum:
                    return candidate
        return hard_end

