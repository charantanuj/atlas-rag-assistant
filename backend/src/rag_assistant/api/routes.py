from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from fastapi import APIRouter, File, Request, UploadFile

from rag_assistant.api.contracts import (
    CitationResponse, DocumentListResponse, DocumentResponse, DocumentUploadResponse,
    EvaluationSummaryResponse, QuestionRequest, QuestionResponse,
)
from rag_assistant.api.service import ApplicationService, DocumentView

router = APIRouter()


def _service(request: Request) -> ApplicationService:
    return request.app.state.application_service


def _document(view: DocumentView) -> DocumentResponse:
    item = view.record
    return DocumentResponse(
        id=item.id, name=item.original_name, media_type=item.media_type,
        size_bytes=item.size_bytes, status=item.status.value,
        segment_count=item.segment_count, chunk_count=view.chunk_count,
        quality_flags=list(item.quality_flags), created_at=item.created_at,
        error_code=item.error_code,
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(request: Request) -> DocumentListResponse:
    return DocumentListResponse(documents=[_document(item) for item in _service(request).list_documents()])


@router.post("/documents", response_model=DocumentUploadResponse, tags=["documents"])
def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentUploadResponse:
    view, duplicate = _service(request).upload(file.filename or "", file.file)
    return DocumentUploadResponse(document=_document(view), duplicate=duplicate)


@router.post("/questions", response_model=QuestionResponse, tags=["questions"])
def ask_question(payload: QuestionRequest, request: Request) -> QuestionResponse:
    result = _service(request).ask(payload.question, tuple(payload.document_ids))
    request_id = request.state.request_id
    return QuestionResponse(
        id=f"ans_{uuid4().hex}", status=result.status.value, answer=result.answer,
        citations=[CitationResponse(**asdict(citation)) for citation in result.citations],
        reason_codes=list(result.reason_codes), request_id=request_id,
        candidates_retrieved=result.candidates_retrieved, evidence_used=result.evidence_used,
    )


@router.get("/evaluation/summary", response_model=EvaluationSummaryResponse, tags=["evaluation"])
def evaluation_summary(request: Request) -> EvaluationSummaryResponse:
    return EvaluationSummaryResponse(**_service(request).evaluation_summary())


@router.get("/observability/cache", tags=["observability"])
def cache_summary(request: Request) -> dict:
    return _service(request).cache_summary()
