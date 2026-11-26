"""Transparent, deterministic claim-risk rules.

Rules operate exclusively on the point-in-time feature schema. They return
review signals and probabilistic risk contributions, not fraud findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Versionable thresholds and risk contributions for the rule baseline."""

    minimum_agent_history_for_amount_rule: int = 3
    amount_vs_agent_average_threshold: float = 2.5
    recent_agent_claim_frequency_threshold: int = 8
    duplicate_similarity_threshold: float = 1.0
    document_date_difference_threshold_days: int = 30
    amount_rule_contribution: float = 0.25
    agent_frequency_rule_contribution: float = 0.20
    inactive_policy_rule_contribution: float = 0.30
    duplicate_claim_rule_contribution: float = 0.35
    document_date_rule_contribution: float = 0.20
    repeated_customer_rule_contribution: float = 0.15
    maximum_rule_risk_score: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_agent_history_for_amount_rule < 1:
            raise ValueError("minimum_agent_history_for_amount_rule must be at least one.")
        if self.amount_vs_agent_average_threshold <= 1:
            raise ValueError("amount_vs_agent_average_threshold must exceed one.")
        if self.recent_agent_claim_frequency_threshold < 1:
            raise ValueError("recent_agent_claim_frequency_threshold must be positive.")
        if self.document_date_difference_threshold_days < 0:
            raise ValueError("document_date_difference_threshold_days cannot be negative.")
        contributions = (
            self.amount_rule_contribution,
            self.agent_frequency_rule_contribution,
            self.inactive_policy_rule_contribution,
            self.duplicate_claim_rule_contribution,
            self.document_date_rule_contribution,
            self.repeated_customer_rule_contribution,
        )
        if any(not 0 <= contribution <= 1 for contribution in contributions):
            raise ValueError("Rule contributions must be between zero and one.")
        if not 0 < self.maximum_rule_risk_score <= 1:
            raise ValueError("maximum_rule_risk_score must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Auditable outcome of one deterministic rule."""

    name: str
    triggered: bool
    risk_contribution: float
    reason: str


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """Aggregate baseline result plus every individual rule outcome."""

    rule_risk_score: float
    rule_reasons: tuple[str, ...]
    results: tuple[RuleResult, ...]


REQUIRED_FEATURE_COLUMNS = frozenset(
    {
        "claim_amount_vs_agent_historical_average",
        "agent_historical_claim_count",
        "agent_recent_claim_frequency",
        "policy_active_indicator",
        "duplicate_similarity",
        "document_count",
        "document_incident_date_difference_days",
        "repeated_customer_claim_indicator",
    }
)


def _result(name: str, triggered: bool, contribution: float, reason: str) -> RuleResult:
    """Return a rule result with contribution set to zero when not triggered."""
    return RuleResult(
        name=name,
        triggered=triggered,
        risk_contribution=contribution if triggered else 0.0,
        reason=reason if triggered else "Not triggered: configured rule threshold was not met.",
    )


def evaluate_claim_rules(features: Mapping[str, Any], config: RuleConfig | None = None) -> RuleEvaluation:
    """Evaluate all deterministic rules for a single point-in-time feature row."""
    config = config or RuleConfig()
    missing = REQUIRED_FEATURE_COLUMNS.difference(features)
    if missing:
        raise ValueError(f"Features missing required rule inputs: {sorted(missing)}")

    amount_ratio = float(features["claim_amount_vs_agent_historical_average"])
    history_count = int(features["agent_historical_claim_count"])
    recent_frequency = int(features["agent_recent_claim_frequency"])
    policy_active = int(features["policy_active_indicator"])
    duplicate_similarity = float(features["duplicate_similarity"])
    document_count = int(features["document_count"])
    date_difference = features["document_incident_date_difference_days"]
    repeated_customer = int(features["repeated_customer_claim_indicator"])

    results = (
        _result(
            "claim_amount_above_agent_average",
            history_count >= config.minimum_agent_history_for_amount_rule and amount_ratio >= config.amount_vs_agent_average_threshold,
            config.amount_rule_contribution,
            f"Claim amount is {amount_ratio:.2f}× the agent's historical average across {history_count} prior claims.",
        ),
        _result(
            "abnormal_agent_claim_frequency",
            recent_frequency >= config.recent_agent_claim_frequency_threshold,
            config.agent_frequency_rule_contribution,
            f"Agent submitted {recent_frequency} claims in the configured recent-history window.",
        ),
        _result(
            "inactive_policy",
            policy_active == 0,
            config.inactive_policy_rule_contribution,
            "Policy is inactive or not in force for the reported incident date.",
        ),
        _result(
            "duplicate_claim",
            duplicate_similarity >= config.duplicate_similarity_threshold,
            config.duplicate_claim_rule_contribution,
            f"Claim narrative exactly matches a prior submitted narrative (similarity {duplicate_similarity:.2f}).",
        ),
        _result(
            "incident_document_date_inconsistency",
            document_count > 0 and pd.notna(date_difference) and float(date_difference) >= config.document_date_difference_threshold_days,
            config.document_date_rule_contribution,
            f"Available document date differs from the incident date by {float(date_difference):.0f} days.",
        ),
        _result(
            "repeated_customer_claims",
            repeated_customer == 1,
            config.repeated_customer_rule_contribution,
            "Customer has reached the configured prior-claim frequency threshold.",
        ),
    )
    reasons = tuple(result.reason for result in results if result.triggered)
    score = min(config.maximum_rule_risk_score, sum(result.risk_contribution for result in results))
    return RuleEvaluation(rule_risk_score=round(score, 6), rule_reasons=reasons, results=results)


def apply_rule_baseline(features: pd.DataFrame, config: RuleConfig | None = None) -> pd.DataFrame:
    """Add transparent rule outputs to a feature frame without mutating it."""
    config = config or RuleConfig()
    missing = REQUIRED_FEATURE_COLUMNS.difference(features.columns)
    if missing:
        raise ValueError(f"Feature frame missing required rule inputs: {sorted(missing)}")
    output = features.copy()
    evaluations = [evaluate_claim_rules(row, config) for row in output.to_dict(orient="records")]
    output["rule_risk_score"] = [evaluation.rule_risk_score for evaluation in evaluations]
    output["rule_reasons"] = ["; ".join(evaluation.rule_reasons) for evaluation in evaluations]
    return output
