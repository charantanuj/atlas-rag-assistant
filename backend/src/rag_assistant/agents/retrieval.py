"""Retrieval node: retrieves, filters, and reranks document evidence."""

from __future__ import annotations

from rag_assistant.retrieval.models import RetrievalResult
from rag_assistant.retrieval.service import RetrievalService


class RetrievalNode:
    def __init__(self, service: RetrievalService) -> None:
        self.service = service

    def retrieve(
        self, question: str, *, document_ids: tuple[str, ...] = ()
    ) -> RetrievalResult:
        return self.service.retrieve(question, document_ids=document_ids)

