class IngestionError(Exception):
    """Base class for safe, expected ingestion failures."""

    code = "INGESTION_ERROR"


class UnsupportedFormatError(IngestionError):
    code = "UNSUPPORTED_FORMAT"


class InvalidDocumentError(IngestionError):
    code = "INVALID_DOCUMENT"


class EncryptedDocumentError(IngestionError):
    code = "ENCRYPTED_DOCUMENT"


class MissingParserDependencyError(IngestionError):
    code = "MISSING_PARSER_DEPENDENCY"

