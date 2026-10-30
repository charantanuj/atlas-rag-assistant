from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class DocumentResponse(ApiModel):
    id: str
    name: str
    media_type: str = Field(serialization_alias="mediaType")
    size_bytes: int = Field(serialization_alias="sizeBytes")
    status: str
    segment_count: int = Field(serialization_alias="segmentCount")
    chunk_count: int = Field(serialization_alias="chunkCount")
    quality_flags: list[str] = Field(serialization_alias="qualityFlags")
    created_at: datetime = Field(serialization_alias="createdAt")
    error_code: str | None = Field(default=None, serialization_alias="errorCode")


class DocumentListResponse(ApiModel):
    documents: list[DocumentResponse]


class DocumentUploadResponse(ApiModel):
    document: DocumentResponse
    duplicate: bool


class QuestionRequest(ApiModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


class CitationResponse(ApiModel):
    id: int
    chunk_id: str = Field(serialization_alias="chunkId")
    document_id: str = Field(serialization_alias="documentId")
    document_name: str = Field(serialization_alias="documentName")
    location_type: str = Field(serialization_alias="locationType")
    location: int
    excerpt: str
    relevance: float


class QuestionResponse(ApiModel):
    id: str
    status: str
    answer: str
    citations: list[CitationResponse]
    reason_codes: list[str] = Field(serialization_alias="reasonCodes")
    request_id: str = Field(serialization_alias="requestId")
    candidates_retrieved: int = Field(serialization_alias="candidatesRetrieved")
    evidence_used: int = Field(serialization_alias="evidenceUsed")


class EvaluationSummaryResponse(ApiModel):
    passed: bool
    recall_at_k: float = Field(serialization_alias="recallAtK")
    reciprocal_rank: float = Field(serialization_alias="reciprocalRank")
    citation_precision: float = Field(serialization_alias="citationPrecision")
    citation_completeness: float = Field(serialization_alias="citationCompleteness")
    abstention_accuracy: float = Field(serialization_alias="abstentionAccuracy")
    attack_block_rate: float = Field(serialization_alias="attackBlockRate")
    false_positive_rate: float = Field(serialization_alias="falsePositiveRate")
    last_run_at: datetime | None = Field(default=None, serialization_alias="lastRunAt")
    failed_gates: list[str] = Field(serialization_alias="failedGates")
