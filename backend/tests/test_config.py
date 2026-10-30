from pathlib import Path

from rag_assistant.core.config import Settings


def test_settings_normalize_extensions_and_empty_key() -> None:
    settings = Settings(
        allowed_extensions=".PDF, txt,HTML",
        gemini_api_key="",
        data_dir=Path("test-data"),
    )

    assert settings.allowed_extension_set == frozenset({"pdf", "txt", "html"})
    assert settings.gemini_api_key is None
    assert settings.max_upload_bytes == 50 * 1024 * 1024

