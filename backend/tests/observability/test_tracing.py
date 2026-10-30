import pytest

from rag_assistant.observability.tracing import TraceRecorder


def test_trace_records_timing_and_safe_attributes() -> None:
    trace = TraceRecorder("req_test")

    with trace.span("retrieval", candidate_count=4) as attributes:
        attributes["result_count"] = 2

    event = trace.events[0]
    assert event.request_id == "req_test"
    assert event.stage == "retrieval"
    assert event.status == "ok"
    assert event.duration_ms >= 0
    assert event.attributes == {"candidate_count": 4, "result_count": 2}


@pytest.mark.parametrize("key", ["question", "prompt", "answer", "api_key", "token"])
def test_sensitive_trace_attributes_are_rejected(key: str) -> None:
    trace = TraceRecorder("req_test")

    with pytest.raises(ValueError):
        with trace.span("unsafe", **{key: "sensitive"}):
            pass


def test_trace_records_exception_type_without_exception_message() -> None:
    trace = TraceRecorder("req_test")

    with pytest.raises(RuntimeError):
        with trace.span("provider"):
            raise RuntimeError("secret provider response")

    event = trace.events[0]
    assert event.status == "error"
    assert event.error_type == "RuntimeError"
    assert "secret" not in str(event)

