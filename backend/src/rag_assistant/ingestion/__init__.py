"""Format-specific document ingestion."""

from rag_assistant.ingestion.registry import LoaderRegistry, default_loader_registry

__all__ = ["LoaderRegistry", "default_loader_registry"]

