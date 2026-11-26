"""Tests for deterministic risk aggregation and advance recommendations."""

from __future__ import annotations

import pytest

from src.risk.decision import DecisionConfig, RiskEvidence, aggregate_risk_and_decide


def _evidence(**changes) -> RiskEvidence:
    values = {
        "fraud_probability": 0.10,
        "rule_risk_score": 0.0,
        "agent_anomaly_score": 0.0,
        "maximum_prior_semantic_similarity": 0.0,
        "highly_similar_prior_claim_count": 0,
        "repeated_template_narrative_indicator": 0,
        "duplicate_similarity": 0.0,
        "policy_active_indicator": 1,
        "document_count": 1,
        "document_incident_date_difference_days": 1.0,
        "claim_amount": 10_000.0,
    }
    values.update(changes)
    return RiskEvidence(**values)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.2999, "low"), (0.30, "medium"), (0.6999, "medium"), (0.70, "high")],
)
def test_threshold_boundaries(probability: float, expected: str) -> None:
    assert aggregate_risk_and_decide(_evidence(fraud_probability=probability)).risk_level == expected


def test_low_risk_recommends_configured_advance_using_verified_amount() -> None:
    decision = aggregate_risk_and_decide(_evidence(verified_amount=8_000.0))
    assert decision.risk_level == "low"
    assert decision.recommended_advance == 1_600.0
    assert not decision.manual_review_required
    assert set(decision.model_dump()) == {
        "fraud_probability", "risk_level", "risk_factors", "recommended_advance", "manual_review_required", "decision_reasons"
    }


def test_medium_risk_limits_advance_and_requires_review() -> None:
    decision = aggregate_risk_and_decide(_evidence(fraud_probability=0.50))
    assert decision.risk_level == "medium"
    assert decision.recommended_advance == 1_000.0
    assert decision.manual_review_required


def test_high_risk_has_no_automatic_advance() -> None:
    decision = aggregate_risk_and_decide(_evidence(fraud_probability=0.70))
    assert decision.risk_level == "high"
    assert decision.recommended_advance == 0.0
    assert decision.manual_review_required


def test_high_claim_amount_applies_business_cap() -> None:
    decision = aggregate_risk_and_decide(_evidence(claim_amount=75_000.0))
    assert decision.risk_level == "low"
    assert decision.recommended_advance == DecisionConfig().high_claim_amount_advance_cap
    assert decision.manual_review_required
    assert any(factor.name == "high_claim_amount" for factor in decision.risk_factors)


def test_invalid_policy_is_a_high_risk_safeguard() -> None:
    decision = aggregate_risk_and_decide(_evidence(policy_active_indicator=0))
    assert decision.risk_level == "high"
    assert decision.recommended_advance == 0.0
    assert any(factor.name == "policy_not_active_at_incident" for factor in decision.risk_factors)


def test_document_inconsistency_elevates_low_risk_claim() -> None:
    decision = aggregate_risk_and_decide(_evidence(document_incident_date_difference_days=45.0))
    assert decision.risk_level == "medium"
    assert any(factor.name == "document_date_inconsistency" for factor in decision.risk_factors)


def test_multiple_simultaneous_risk_factors_are_all_auditable() -> None:
    decision = aggregate_risk_and_decide(
        _evidence(
            fraud_probability=0.40,
            rule_risk_score=0.75,
            agent_anomaly_score=0.90,
            maximum_prior_semantic_similarity=0.96,
            highly_similar_prior_claim_count=3,
            repeated_template_narrative_indicator=1,
            duplicate_similarity=1.0,
            document_incident_date_difference_days=60.0,
        )
    )
    assert decision.risk_level == "high"
    assert decision.recommended_advance == 0.0
    assert {factor.name for factor in decision.risk_factors} == {
        "elevated_rule_risk",
        "unusual_agent_behavior",
        "narrative_similarity",
        "repeated_narrative_template",
        "exact_prior_narrative_match",
        "document_date_inconsistency",
    }
