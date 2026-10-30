from starlette.requests import Request
from starlette.responses import Response

from rag_assistant.api.health import health


def get_health(request_id: str | None = None):
    headers = []
    if request_id:
        headers.append((b"x-request-id", request_id.encode()))
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": headers})
    response = Response()
    body = health(request, response)
    return body, response


def test_health_is_available_without_paid_provider_credentials() -> None:
    body, response = get_health()

    assert body.status == "ok"
    assert body.components.configuration == "ok"
    assert body.components.gemini in {"configured", "not_configured"}
    assert body.request_id.startswith("req_")
    assert response.headers["X-Request-ID"] == body.request_id


def test_health_preserves_caller_request_id() -> None:
    body, response = get_health("req_test")

    assert body.request_id == "req_test"
    assert response.headers["X-Request-ID"] == "req_test"

