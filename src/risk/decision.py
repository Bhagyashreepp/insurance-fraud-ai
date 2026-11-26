"""Deterministic aggregation of model evidence and advance recommendations.

This module returns review signals and a suggested advance workflow.  It does
not make legal, coverage, payment, or confirmed-fraud determinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator


RiskLevel = Literal["low", "medium", "high"]


class RiskEvidence(BaseModel):
    """Computed claim-time evidence supplied by upstream deterministic systems."""

    fraud_probability: float = Field(ge=0, le=1)
    rule_risk_score: float = Field(default=0.0, ge=0, le=1)
    agent_anomaly_score: float = Field(default=0.0, ge=0, le=1)
    maximum_prior_semantic_similarity: float = Field(default=0.0, ge=-1, le=1)
    highly_similar_prior_claim_count: int = Field(default=0, ge=0)
    repeated_template_narrative_indicator: int = Field(default=0, ge=0, le=1)
    duplicate_similarity: float = Field(default=0.0, ge=0, le=1)
    policy_active_indicator: int = Field(default=1, ge=0, le=1)
    document_count: int = Field(default=0, ge=0)
    document_incident_date_difference_days: float | None = Field(default=None, ge=0)
    claim_amount: float = Field(ge=0)
    verified_amount: float | None = Field(default=None, ge=0)
    claim_id: str | None = None

    @field_validator("repeated_template_narrative_indicator", "policy_active_indicator")
    @classmethod
    def _binary_indicator(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("Indicator fields must be zero or one.")
        return value


class RiskFactor(BaseModel):
    """An auditable risk factor tied to a computed observable value."""

    name: str
    evidence: str
    severity: Literal["review", "elevated", "high"]


class AdvanceDecision(BaseModel):
    """The complete stable contract for the deterministic decision engine."""

    fraud_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    risk_factors: list[RiskFactor]
    recommended_advance: float = Field(ge=0)
    manual_review_required: bool
    decision_reasons: list[str]


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    """Versionable thresholds and business caps for the advance workflow."""

    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.70
    low_risk_advance_percentage: float = 0.20
    medium_risk_advance_percentage: float = 0.10
    low_risk_advance_cap: float = 5_000.0
    medium_risk_advance_cap: float = 1_500.0
    high_claim_amount_threshold: float = 50_000.0
    high_claim_amount_advance_cap: float = 1_000.0
    rule_risk_factor_threshold: float = 0.50
    agent_anomaly_factor_threshold: float = 0.80
    semantic_similarity_factor_threshold: float = 0.90
    document_date_difference_threshold_days: float = 30.0
    medium_risk_manual_review_required: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.low_risk_threshold < self.high_risk_threshold < 1:
            raise ValueError("Risk thresholds must satisfy 0 < low < high < 1.")
        if any(value < 0 for value in (self.low_risk_advance_cap, self.medium_risk_advance_cap, self.high_claim_amount_advance_cap)):
            raise ValueError("Advance caps cannot be negative.")
        if any(not 0 <= value <= 1 for value in (self.low_risk_advance_percentage, self.medium_risk_advance_percentage)):
            raise ValueError("Advance percentages must be between zero and one.")


def _base_risk_level(probability: float, config: DecisionConfig) -> RiskLevel:
    if probability < config.low_risk_threshold:
        return "low"
    if probability < config.high_risk_threshold:
        return "medium"
    return "high"


def _at_least_medium(level: RiskLevel) -> RiskLevel:
    return "medium" if level == "low" else level


def _risk_factors(evidence: RiskEvidence, config: DecisionConfig) -> tuple[list[RiskFactor], RiskLevel]:
    """Build only factors that cite a concrete upstream feature or rule score."""
    factors: list[RiskFactor] = []
    level = _base_risk_level(evidence.fraud_probability, config)
    if evidence.rule_risk_score >= config.rule_risk_factor_threshold:
        factors.append(RiskFactor(name="elevated_rule_risk", severity="elevated", evidence=f"Rule risk score is {evidence.rule_risk_score:.2f}."))
    if evidence.agent_anomaly_score >= config.agent_anomaly_factor_threshold:
        factors.append(RiskFactor(name="unusual_agent_behavior", severity="elevated", evidence=f"Agent anomaly score is {evidence.agent_anomaly_score:.2f}."))
        level = _at_least_medium(level)
    if evidence.maximum_prior_semantic_similarity >= config.semantic_similarity_factor_threshold:
        factors.append(
            RiskFactor(
                name="narrative_similarity",
                severity="elevated",
                evidence=(
                    f"Maximum prior semantic similarity is {evidence.maximum_prior_semantic_similarity:.2f} "
                    f"across {evidence.highly_similar_prior_claim_count} highly similar prior claim(s)."
                ),
            )
        )
        level = _at_least_medium(level)
    if evidence.repeated_template_narrative_indicator:
        factors.append(RiskFactor(name="repeated_narrative_template", severity="elevated", evidence="A repeated narrative template indicator is present."))
        level = _at_least_medium(level)
    if evidence.duplicate_similarity >= 1.0:
        factors.append(RiskFactor(name="exact_prior_narrative_match", severity="high", evidence="Narrative exactly matches an earlier submitted claim."))
        level = "high"
    if not evidence.policy_active_indicator:
        factors.append(RiskFactor(name="policy_not_active_at_incident", severity="high", evidence="Policy consistency feature indicates the policy was not active at the incident date."))
        level = "high"
    if (
        evidence.document_incident_date_difference_days is not None
        and evidence.document_incident_date_difference_days >= config.document_date_difference_threshold_days
    ):
        factors.append(
            RiskFactor(
                name="document_date_inconsistency",
                severity="elevated",
                evidence=(
                    "Available document date differs from incident date by "
                    f"{evidence.document_incident_date_difference_days:.0f} days."
                ),
            )
        )
        level = _at_least_medium(level)
    if evidence.claim_amount >= config.high_claim_amount_threshold:
        factors.append(
            RiskFactor(
                name="high_claim_amount", severity="review", evidence=f"Reported claim amount is ${evidence.claim_amount:,.2f}."
            )
        )
    return factors, level


def aggregate_risk_and_decide(evidence: RiskEvidence, config: DecisionConfig | None = None) -> AdvanceDecision:
    """Return a deterministic, evidence-bound advance recommendation without an LLM."""
    config = config or DecisionConfig()
    factors, risk_level = _risk_factors(evidence, config)
    eligible_amount = min(evidence.claim_amount, evidence.verified_amount) if evidence.verified_amount is not None else evidence.claim_amount
    if risk_level == "high":
        advance = 0.0
        manual_review = True
        reasons = ["High-risk evidence requires manual review; no automatic advance is recommended."]
    elif risk_level == "medium":
        advance = min(eligible_amount * config.medium_risk_advance_percentage, config.medium_risk_advance_cap)
        manual_review = config.medium_risk_manual_review_required
        reasons = ["Medium-risk evidence supports only a limited advance pending additional verification."]
    else:
        advance = min(eligible_amount * config.low_risk_advance_percentage, config.low_risk_advance_cap)
        manual_review = False
        reasons = ["Low-risk evidence supports an advance within configured verification and business limits."]
    if evidence.claim_amount >= config.high_claim_amount_threshold and risk_level != "high":
        advance = min(advance, config.high_claim_amount_advance_cap)
        manual_review = True
        reasons.append("High reported amount applies the configured business cap and requires review.")
    if evidence.verified_amount is not None and evidence.verified_amount < evidence.claim_amount:
        reasons.append("Advance calculation uses the lower verified amount.")
    if factors:
        reasons.append("Review signals are based only on the observable evidence listed in risk factors.")
    return AdvanceDecision(
        fraud_probability=evidence.fraud_probability,
        risk_level=risk_level,
        risk_factors=factors,
        recommended_advance=round(advance, 2),
        manual_review_required=manual_review,
        decision_reasons=reasons,
    )
