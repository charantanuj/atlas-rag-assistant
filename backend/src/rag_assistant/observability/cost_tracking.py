from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.observability.models import CostEstimate, TokenUsage


@dataclass(frozen=True, slots=True)
class ModelRate:
    input_per_million: float
    output_per_million: float
    currency: str = "USD"


class CostCalculator:
    """Uses injected rates so pricing changes do not require code changes."""

    def __init__(self, rates: dict[tuple[str, str], ModelRate]) -> None:
        self.rates = rates

    def estimate(self, usage: TokenUsage) -> CostEstimate | None:
        rate = self.rates.get((usage.provider, usage.model))
        if rate is None:
            return None
        input_cost = usage.input_tokens / 1_000_000 * rate.input_per_million
        output_cost = usage.output_tokens / 1_000_000 * rate.output_per_million
        return CostEstimate(
            currency=rate.currency,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )

