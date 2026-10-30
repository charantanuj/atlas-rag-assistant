from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/metadata.sqlite3")
    chroma_path: Path = Path("data/chroma")
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    allowed_extensions: str = "pdf,txt,html,docx,pptx"
    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_thinking_level: str = "minimal"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_provider: str = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_device: str = "cpu"
    local_model_cache: Path = Path("data/models")
    local_embedding_batch_size: int = Field(default=8, ge=1, le=128)
    retrieval_top_k: int = Field(default=12, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.45, ge=0, le=1)
    answer_cache_max_items: int = Field(default=256, ge=1, le=10000)
    answer_cache_ttl_seconds: float = Field(default=900.0, gt=0, le=86400)
    query_cache_max_items: int = Field(default=512, ge=1, le=10000)
    query_cache_ttl_seconds: float = Field(default=3600.0, gt=0, le=86400)
    injection_model_enabled: bool = False
    injection_model_name: str = "protectai/deberta-v3-base-prompt-injection-v2"
    injection_model_threshold: float = Field(default=0.9, ge=0, le=1)
    auth_enabled: bool = False
    auth_username: str = "admin@example.com"
    auth_password_hash: str | None = None
    auth_secret: str = "change-this-local-secret"

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def empty_secret_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("embedding_provider")
    @classmethod
    def valid_embedding_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"local", "gemini"}:
            raise ValueError("embedding_provider must be 'local' or 'gemini'")
        return normalized

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        return frozenset(
            item.strip().lower().lstrip(".")
            for item in self.allowed_extensions.split(",")
            if item.strip()
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

