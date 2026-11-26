"""Tests for deterministic and mocked OpenAI explanation rendering."""

from __future__ import annotations

import json

import pytest

from src.explanations.service import ClaimExplanation, ExplanationEvidence, generate_explanation
from src.risk.decision import RiskEvidence, aggregate_risk_and_decide


def _evidence() -> ExplanationEvidence:
    decision = aggregate_risk_and_decide(
        RiskEvidence(
            claim_id="C-100",
            fraud_probability=0.55,
            rule_risk_score=0.70,
            agent_anomaly_score=0.85,
            maximum_prior_semantic_similarity=0.95,
            highly_similar_prior_claim_count=2,
            repeated_template_narrative_indicator=1,
            duplicate_similarity=0.0,
            policy_active_indicator=1,
            document_count=1,
            document_incident_date_difference_days=40,
            claim_amount=12_000,
        )
    )
    return ExplanationEvidence(
        claim_facts={"claim_id": "C-100", "claim_amount": 12_000.0, "loss_type": "water"},
        fraud_probability=decision.fraud_probability,
        risk_level=decision.risk_level,
        triggered_rules=decision.risk_factors,
        agent_anomaly=0.85,
        writing_similarity={"maximum_prior_semantic_similarity": 0.95, "highly_similar_prior_claim_count": 2},
        similar_prior_claim_references=["C-010", "C-044"],
        document_inconsistencies=["Available document date differs from incident date by 40 days."],
        policy_consistency=True,
        amount_anomaly=1.8,
        decision_recommendation=decision,
    )


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text})()


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


def test_missing_key_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    explanation = generate_explanation(_evidence())
    assert isinstance(explanation, ClaimExplanation)
    assert "not a finding of fraud" in explanation.uncertainty_limitations
    assert explanation.recommended_investigator_action.startswith("Complete manual review")


def test_openai_response_is_validated_and_cannot_change_decision(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = {
        "concise_claim_summary": "Claim C-100 is ready for review.",
        "key_risk_factors": ["Unusual agent behavior."],
        "supporting_evidence": ["Agent anomaly score is 0.85."],
        "uncertainty_limitations": "This is a review signal, not a finding of fraud.",
        "recommended_investigator_action": "Verify supporting documents before any advance.",
    }
    client = _FakeClient(json.dumps(payload))
    evidence = _evidence()
    explanation = generate_explanation(evidence, client=client)
    assert explanation == ClaimExplanation(**payload)
    request = client.responses.calls[0]
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert str(evidence.fraud_probability) in request["input"][1]["content"]
    assert str(evidence.decision_recommendation.recommended_advance) in request["input"][1]["content"]


def test_invalid_or_failed_api_response_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _FakeClient('{"concise_claim_summary": "missing required fields"}')
    explanation = generate_explanation(_evidence(), client=client)
    assert "synthetic-data risk estimate" in explanation.uncertainty_limitations


def test_explanation_evidence_rejects_raw_narrative_text() -> None:
    payload = _evidence().model_dump()
    payload["claim_facts"]["narrative"] = "Raw sensitive claim narrative"
    with pytest.raises(ValueError, match="minimum non-sensitive"):
        ExplanationEvidence(**payload)


def test_fallback_distinguishes_model_probability_from_safeguard_risk_level(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    payload = _evidence().model_dump()
    payload["fraud_probability"] = 0.05
    payload["risk_level"] = "high"
    payload["decision_recommendation"]["fraud_probability"] = 0.05
    payload["decision_recommendation"]["risk_level"] = "high"
    explanation = generate_explanation(ExplanationEvidence(**payload))
    assert "calibrated model probability" in explanation.concise_claim_summary
    assert "final high risk level" in explanation.concise_claim_summary
