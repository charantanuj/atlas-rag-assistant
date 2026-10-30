from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import Lock

from rag_assistant.observability.models import TraceEvent


@dataclass(frozen=True, slots=True)
class StageMetrics:
    calls: int
    errors: int
    average_duration_ms: float
    maximum_duration_ms: float


class InMemoryMetrics:
    """Local metrics sink; replaceable by OpenTelemetry/Prometheus later."""

    def __init__(self) -> None:
        self._calls: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def observe(self, event: TraceEvent) -> None:
        with self._lock:
            self._calls[event.stage] += 1
            self._errors[event.stage] += int(event.status == "error")
            self._durations[event.stage].append(event.duration_ms)

    def snapshot(self) -> dict[str, StageMetrics]:
        with self._lock:
            return {
                stage: StageMetrics(
                    calls=self._calls[stage],
                    errors=self._errors[stage],
                    average_duration_ms=(
                        sum(values) / len(values) if values else 0.0
                    ),
                    maximum_duration_ms=max(values, default=0.0),
                )
                for stage, values in self._durations.items()
            }

