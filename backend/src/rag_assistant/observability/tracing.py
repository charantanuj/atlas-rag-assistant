from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Callable, Iterator

from rag_assistant.observability.models import TraceEvent

_ALLOWED_ATTRIBUTE_TYPES = (str, int, float, bool)
_FORBIDDEN_KEYS = {
    "question",
    "query",
    "prompt",
    "answer",
    "document_text",
    "content",
    "api_key",
    "secret",
    "token",
}


class TraceRecorder:
    def __init__(
        self,
        request_id: str,
        sink: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.request_id = request_id
        self.events: list[TraceEvent] = []
        self.sink = sink

    @contextmanager
    def span(
        self, stage: str, **attributes: str | int | float | bool
    ) -> Iterator[dict[str, str | int | float | bool]]:
        safe_attributes = self._validate_attributes(attributes)
        started_at = datetime.now(UTC).isoformat()
        started = perf_counter()
        status = "ok"
        error_type = None
        try:
            yield safe_attributes
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            event = TraceEvent(
                request_id=self.request_id,
                stage=stage,
                started_at=started_at,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                status=status,
                attributes=self._validate_attributes(safe_attributes),
                error_type=error_type,
            )
            self.events.append(event)
            if self.sink:
                self.sink(event)

    @staticmethod
    def _validate_attributes(
        attributes: dict[str, object],
    ) -> dict[str, str | int | float | bool]:
        safe: dict[str, str | int | float | bool] = {}
        for key, value in attributes.items():
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"Trace attribute '{key}' may contain sensitive content")
            if not isinstance(value, _ALLOWED_ATTRIBUTE_TYPES):
                raise TypeError(f"Trace attribute '{key}' must be a scalar")
            safe[key] = value
        return safe

