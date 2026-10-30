from __future__ import annotations

from uuid import uuid4
from time import perf_counter
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_assistant import __version__
from rag_assistant.api.health import router as health_router
from rag_assistant.api.auth import LoginRequest, LoginResponse, login, password_hash, require_auth
from rag_assistant.api.routes import router as application_router
from rag_assistant.api.service import ApplicationError, ApplicationService, build_local_application_service
from rag_assistant.core.config import get_settings

logger = logging.getLogger("rag_assistant.http")


def create_app(application_service: ApplicationService | None = None) -> FastAPI:
    app = FastAPI(
        title="Noise-Aware Technical Document RAG",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    settings = get_settings()
    app.state.application_service = application_service or build_local_application_service(settings)
    app.state.auth_required = application_service is None and settings.auth_enabled
    if app.state.auth_required and not settings.auth_password_hash:
        raise RuntimeError("AUTH_PASSWORD_HASH must be configured when AUTH_ENABLED=true")

    @app.post("/api/v1/auth/login", response_model=LoginResponse)
    async def auth_login(payload: LoginRequest):
        return login(payload.username, payload.password, expected_username=settings.auth_username,
                     expected_hash=settings.auth_password_hash or "", secret=settings.auth_secret)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid4().hex}"
        request.state.request_id = request_id
        protected = request.url.path.startswith(("/api/v1/documents", "/api/v1/questions", "/api/v1/evaluation", "/api/v1/observability"))
        if app.state.auth_required and protected:
            try:
                require_auth(request.headers.get("Authorization"), username=settings.auth_username,
                             password_hash_value=settings.auth_password_hash or "", secret=settings.auth_secret)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers={"X-Request-ID": request_id})
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Response-Time-Ms"] = str(round((perf_counter() - started) * 1000, 2))
        response.headers["X-Request-ID"] = request_id
        logger.info("request_complete method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
                    request.method, request.url.path, response.status_code,
                    (perf_counter() - started) * 1000, request_id)
        return response

    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            media_type="application/problem+json",
            content={
                "type": f"https://rag.local/problems/{exc.code.lower().replace('_', '-')}",
                "title": "Request could not be completed",
                "status": exc.status_code,
                "detail": exc.detail,
                "code": exc.code,
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            media_type="application/problem+json",
            content={
                "type": "https://rag.local/problems/validation-error",
                "title": "Request validation failed",
                "status": 422,
                "detail": "One or more request fields are invalid.",
                "code": "VALIDATION_ERROR",
                "request_id": request.headers.get("X-Request-ID", "req_unavailable"),
            },
        )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(application_router, prefix="/api/v1")
    return app


app = create_app()

