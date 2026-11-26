"""Evidence-bound explanation rendering with an optional OpenAI adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.risk.decision import AdvanceDecision, RiskFactor


class ExplanationEvidence(BaseModel):
    """Minimum computed evidence permitted in an explanation request."""

    model_config = ConfigDict(extra="forbid")

    claim_facts: dict[str, str | int | float | bool | None]
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: str
    triggered_rules: list[RiskFactor] = Field(default_factory=list)
    agent_anomaly: float = Field(ge=0, le=1)
    writing_similarity: dict[str, float | int | bool | None]
    similar_prior_claim_references: list[str] = Field(default_factory=list)
    document_inconsistencies: list[str] = Field(default_factory=list)
    policy_consistency: bool
    amount_anomaly: float | None = None
    decision_recommendation: AdvanceDecision

    @field_validator("claim_facts")
    @classmethod
    def _allow_only_minimum_claim_facts(cls, value: dict[str, str | int | float | bool | None]) -> dict[str, str | int | float | bool | None]:
        allowed = {"claim_id", "claim_amount", "verified_amount", "loss_type", "incident_date", "submitted_date"}
        unexpected = sorted(set(value).difference(allowed))
        if unexpected:
            raise ValueError(f"claim_facts may contain only minimum non-sensitive fields: {unexpected}")
        return value


class ClaimExplanation(BaseModel):
    """Validated explanation sections; this object cannot alter a decision."""

    model_config = ConfigDict(extra="forbid")

    concise_claim_summary: str
    key_risk_factors: list[str]
    supporting_evidence: list[str]
    uncertainty_limitations: str
    recommended_investigator_action: str


@dataclass(frozen=True, slots=True)
class ExplanationConfig:
    """Non-secret OpenAI explanation configuration."""

    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 500


def deterministic_explanation(evidence: ExplanationEvidence) -> ClaimExplanation:
    """Render the evidence locally without adding facts or changing the decision."""
    facts = evidence.claim_facts
    claim_reference = str(facts.get("claim_id", "the claim"))
    amount = facts.get("claim_amount")
    amount_text = f" for ${float(amount):,.2f}" if isinstance(amount, (int, float)) else ""
    factors = [factor.name.replace("_", " ") for factor in evidence.triggered_rules] or ["No configured evidence factors were triggered."]
    support = [factor.evidence for factor in evidence.triggered_rules] or ["Decision uses the calibrated risk estimate and available claim-time evidence."]
    if evidence.document_inconsistencies:
        support.extend(evidence.document_inconsistencies)
    action = (
        "Complete manual review before any advance is issued."
        if evidence.decision_recommendation.manual_review_required
        else "Confirm standard verification requirements before issuing the configured advance."
    )
    return ClaimExplanation(
        concise_claim_summary=(
            f"{claim_reference}{amount_text} has a calibrated model probability of {evidence.fraud_probability:.2f} "
            f"and a final {evidence.risk_level} risk level after configured evidence safeguards."
        ),
        key_risk_factors=factors,
        supporting_evidence=support,
        uncertainty_limitations=(
            "This is a synthetic-data risk estimate based on available claim-time evidence. "
            "It is not a finding of fraud and should be reviewed with applicable policy and verification records."
        ),
        recommended_investigator_action=action,
    )


def _openai_prompt(evidence: ExplanationEvidence) -> str:
    return (
        "Write a concise, factual insurance-claim review explanation from only the JSON evidence below. "
        "Do not add facts, infer missing information, call the claim fraudulent, or alter the probability, risk level, "
        "manual-review flag, or advance recommendation. Return all five requested sections.\n\n"
        f"EVIDENCE:\n{json.dumps(evidence.model_dump(mode='json'), sort_keys=True)}"
    )


def _request_openai_explanation(client: Any, evidence: ExplanationEvidence, config: ExplanationConfig) -> ClaimExplanation:
    """Call Responses API with a strict JSON schema, then validate with Pydantic."""
    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "system",
                "content": "You are an evidence-bound claim-review writing assistant. Do not make decisions.",
            },
            {"role": "user", "content": _openai_prompt(evidence)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "claim_explanation",
                "schema": ClaimExplanation.model_json_schema(),
                "strict": True,
            }
        },
        max_output_tokens=config.max_output_tokens,
        store=False,
    )
    return ClaimExplanation.model_validate_json(response.output_text)


def generate_explanation(
    evidence: ExplanationEvidence,
    config: ExplanationConfig | None = None,
    client: Any | None = None,
) -> ClaimExplanation:
    """Use OpenAI only for wording; always fall back to deterministic evidence text."""
    config = config or ExplanationConfig()
    if not os.environ.get("OPENAI_API_KEY"):
        return deterministic_explanation(evidence)
    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return _request_openai_explanation(client, evidence, config)
    except Exception:
        return deterministic_explanation(evidence)
