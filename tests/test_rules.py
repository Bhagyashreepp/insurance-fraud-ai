"""Tests for the deterministic rule baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.rules import RuleConfig, apply_rule_baseline, evaluate_claim_rules


def _features(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "claim_amount_vs_agent_historical_average": 1.0,
        "agent_historical_claim_count": 5,
        "agent_recent_claim_frequency": 2,
        "policy_active_indicator": 1,
        "duplicate_similarity": 0.0,
        "document_count": 1,
        "document_incident_date_difference_days": 0.0,
        "repeated_customer_claim_indicator": 0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("claim_amount_above_agent_average", {"claim_amount_vs_agent_historical_average": 3.0}),
        ("abnormal_agent_claim_frequency", {"agent_recent_claim_frequency": 8}),
        ("inactive_policy", {"policy_active_indicator": 0}),
        ("duplicate_claim", {"duplicate_similarity": 1.0}),
        ("incident_document_date_inconsistency", {"document_incident_date_difference_days": 30.0}),
        ("repeated_customer_claims", {"repeated_customer_claim_indicator": 1}),
    ],
)
def test_each_rule_triggers_with_a_human_readable_reason(name: str, overrides: dict[str, object]) -> None:
    evaluation = evaluate_claim_rules(_features(**overrides))
    result = next(result for result in evaluation.results if result.name == name)
    assert result.triggered
    assert result.risk_contribution > 0
    assert result.reason


def test_amount_rule_requires_sufficient_history() -> None:
    evaluation = evaluate_claim_rules(_features(claim_amount_vs_agent_historical_average=5.0, agent_historical_claim_count=2))
    result = next(result for result in evaluation.results if result.name == "claim_amount_above_agent_average")
    assert not result.triggered
    assert result.risk_contribution == 0.0
    assert result.reason.startswith("Not triggered")


def test_document_date_rule_requires_an_available_document() -> None:
    evaluation = evaluate_claim_rules(_features(document_count=0, document_incident_date_difference_days=90.0))
    result = next(result for result in evaluation.results if result.name == "incident_document_date_inconsistency")
    assert not result.triggered


def test_baseline_outputs_configurable_capped_score_and_reasons() -> None:
    config = RuleConfig(maximum_rule_risk_score=0.5)
    output = apply_rule_baseline(pd.DataFrame([_features(
        claim_amount_vs_agent_historical_average=3.0,
        agent_recent_claim_frequency=8,
        policy_active_indicator=0,
        duplicate_similarity=1.0,
        document_incident_date_difference_days=31.0,
        repeated_customer_claim_indicator=1,
    )]), config)
    assert output.loc[0, "rule_risk_score"] == 0.5
    assert "inactive" in output.loc[0, "rule_reasons"].lower()


def test_rule_baseline_is_deterministic_and_does_not_use_target_metadata() -> None:
    frame = pd.DataFrame([_features(), _features(duplicate_similarity=1.0)])
    frame["fraud_label"] = [0, 1]
    frame["injected_patterns"] = ["", "duplicate_claim"]
    frame["duplicate_of_claim_id"] = ["", "CLM-000001"]
    first = apply_rule_baseline(frame)
    frame[["fraud_label", "injected_patterns", "duplicate_of_claim_id"]] = [[1, "multi_signal", "CLM-999999"], [0, "", ""]]
    second = apply_rule_baseline(frame)
    pd.testing.assert_series_equal(first["rule_risk_score"], second["rule_risk_score"])
    pd.testing.assert_series_equal(first["rule_reasons"], second["rule_reasons"])
