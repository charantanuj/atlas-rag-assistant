from uuid import uuid4

from fastapi import APIRouter, Request, Response

from rag_assistant import __version__
from rag_assistant.api.models import ComponentHealth, HealthResponse
from rag_assistant.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    settings: Settings = get_settings()
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid4().hex}"
    response.headers["X-Request-ID"] = request_id
    return HealthResponse(
        version=__version__,
        components=ComponentHealth(
            gemini="configured" if settings.gemini_api_key else "not_configured"
        ),
        request_id=request_id,
    )

