from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class TraceEvent:
    request_id: str
    stage: str
    started_at: str
    duration_ms: float
    status: str
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    operation: str
    occurred_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True, slots=True)
class CostEstimate:
    currency: str
    input_cost: float
    output_cost: float
    total_cost: float

