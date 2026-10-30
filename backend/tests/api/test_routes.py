from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from rag_assistant.agents.models import AgentResponse, AnswerStatus
from rag_assistant.api.service import ApplicationError, DocumentView
from rag_assistant.guardrails.models import Citation
from rag_assistant.main import create_app
from rag_assistant.storage.models import DocumentRecord, DocumentStatus


def record() -> DocumentRecord:
    return DocumentRecord(
        id="doc_123", original_name="policy.pdf", media_type="application/pdf",
        extension="pdf", sha256="a" * 64, size_bytes=1200,
        status=DocumentStatus.READY, segment_count=3, character_count=900,
        normalized_path=Path("normalized/doc_123.json"), quality_flags=(),
        error_code=None, created_at="2026-10-29T12:00:00+00:00",
        updated_at="2026-10-29T12:00:00+00:00",
    )


class FakeService:
    def list_documents(self):
        return [DocumentView(record(), 7)]

    def upload(self, filename, stream):
        assert filename == "policy.pdf"
        assert stream.read() == b"pdf bytes"
        return DocumentView(record(), 7), False

    def ask(self, question, document_ids):
        assert question == "What is the retention period?"
        assert document_ids == ("doc_123",)
        return AgentResponse(
            status=AnswerStatus.ANSWERED, answer="Seven years [1].",
            citations=(Citation(1, "chk_1", "doc_123", "policy.pdf", "page", 4, "Retain for seven years.", 0.92),),
            candidates_retrieved=12, evidence_used=1,
        )

    def evaluation_summary(self):
        return {
            "passed": True, "recall_at_k": 0.9, "reciprocal_rank": 0.8,
            "citation_precision": 1.0, "citation_completeness": 1.0,
            "abstention_accuracy": 1.0, "attack_block_rate": 1.0,
            "false_positive_rate": 0.0, "last_run_at": None, "failed_gates": [],
        }


def client(service=None) -> TestClient:
    return TestClient(create_app(service or FakeService()))


def test_list_documents_matches_frontend_contract() -> None:
    response = client().get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json()["documents"][0] == {
        "id": "doc_123", "name": "policy.pdf", "mediaType": "application/pdf",
        "sizeBytes": 1200, "status": "ready", "segmentCount": 3,
        "chunkCount": 7, "qualityFlags": [], "createdAt": "2026-10-29T12:00:00Z",
        "errorCode": None,
    }


def test_upload_uses_multipart_contract() -> None:
    response = client().post(
        "/api/v1/documents", files={"file": ("policy.pdf", BytesIO(b"pdf bytes"), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["document"]["chunkCount"] == 7
    assert response.json()["duplicate"] is False


def test_question_serializes_verified_citation_and_request_id() -> None:
    response = client().post(
        "/api/v1/questions", headers={"X-Request-ID": "req_test"},
        json={"question": "What is the retention period?", "document_ids": ["doc_123"]},
    )
    body = response.json()
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req_test"
    assert body["requestId"] == "req_test"
    assert body["citations"][0]["documentName"] == "policy.pdf"
    assert body["evidenceUsed"] == 1


def test_evaluation_summary_matches_frontend_contract() -> None:
    response = client().get("/api/v1/evaluation/summary")
    assert response.status_code == 200
    assert response.json()["attackBlockRate"] == 1.0
    assert response.json()["failedGates"] == []


class UnavailableService(FakeService):
    def ask(self, question, document_ids):
        raise ApplicationError("MODEL_NOT_CONFIGURED", "Configure Gemini first.", 503)


def test_service_errors_are_problem_details_and_do_not_fake_answers() -> None:
    response = client(UnavailableService()).post(
        "/api/v1/questions", json={"question": "Explain the policy", "document_ids": []},
    )
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "MODEL_NOT_CONFIGURED"
    assert response.json()["request_id"].startswith("req_")
