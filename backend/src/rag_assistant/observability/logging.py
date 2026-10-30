from __future__ import annotations

import json
import logging
from dataclasses import asdict

from rag_assistant.observability.models import TraceEvent


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(payload, separators=(",", ":"))


class TraceEventLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("rag_assistant.trace")

    def __call__(self, event: TraceEvent) -> None:
        self.logger.info("trace_event %s", json.dumps(asdict(event), separators=(",", ":")))


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

