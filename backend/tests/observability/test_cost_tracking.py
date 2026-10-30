import pytest

from rag_assistant.observability.cost_tracking import CostCalculator, ModelRate
from rag_assistant.observability.models import TokenUsage


def test_cost_estimate_uses_injected_model_rates() -> None:
    calculator = CostCalculator({("google", "test-model"): ModelRate(1.0, 4.0)})
    usage = TokenUsage("google", "test-model", 500_000, 250_000, "generation")

    result = calculator.estimate(usage)

    assert result is not None
    assert result.input_cost == pytest.approx(0.5)
    assert result.output_cost == pytest.approx(1.0)
    assert result.total_cost == pytest.approx(1.5)


def test_unknown_model_has_no_invented_cost() -> None:
    calculator = CostCalculator({})

    assert calculator.estimate(TokenUsage("x", "y", 1, 1, "generation")) is None

