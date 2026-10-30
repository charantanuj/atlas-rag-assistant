from rag_assistant.observability.metrics import InMemoryMetrics
from rag_assistant.observability.models import TraceEvent


def event(duration: float, status: str = "ok") -> TraceEvent:
    return TraceEvent("req", "retrieval", "now", duration, status)


def test_metrics_aggregate_calls_errors_and_latency() -> None:
    metrics = InMemoryMetrics()
    metrics.observe(event(10.0))
    metrics.observe(event(30.0, "error"))

    snapshot = metrics.snapshot()["retrieval"]

    assert snapshot.calls == 2
    assert snapshot.errors == 1
    assert snapshot.average_duration_ms == 20.0
    assert snapshot.maximum_duration_ms == 30.0

