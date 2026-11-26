"""Tests for point-in-time Isolation Forest agent anomaly scoring."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.models.agent_anomaly import (
    PROFILE_COLUMNS,
    AgentAnomalyConfig,
    AgentAnomalyDetector,
    build_agent_behavior_profiles,
    score_agent_behavior,
)


def _claims() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"claim_id": "C1", "agent_id": "A1", "loss_type": "water", "claim_amount": 100.0, "narrative": "Short claim.", "submitted_date": "2024-01-01"},
            {"claim_id": "C2", "agent_id": "A1", "loss_type": "water", "claim_amount": 120.0, "narrative": "Short claim.", "submitted_date": "2024-01-02"},
            {"claim_id": "C3", "agent_id": "A1", "loss_type": "fire", "claim_amount": 130.0, "narrative": "A somewhat longer claim narrative.", "submitted_date": "2024-01-02"},
            {"claim_id": "C4", "agent_id": "A1", "loss_type": "fire", "claim_amount": 500.0, "narrative": "A much longer claim narrative for testing.", "submitted_date": "2024-01-03"},
        ]
    )


def test_profiles_are_strictly_point_in_time() -> None:
    profiles = build_agent_behavior_profiles(_claims()).set_index("claim_id")
    assert profiles.loc["C1", "agent_historical_claim_count"] == 0
    assert profiles.loc["C2", "agent_historical_claim_count"] == 1
    assert profiles.loc["C3", "agent_historical_claim_count"] == 1  # C2 is same-day and unavailable.
    assert profiles.loc["C4", "agent_historical_claim_count"] == 3


def test_detector_rejects_target_metadata_and_scores_between_zero_and_one() -> None:
    profiles = build_agent_behavior_profiles(_claims())
    detector = AgentAnomalyDetector(AgentAnomalyConfig(n_estimators=20))
    with pytest.raises(ValueError, match="target metadata"):
        detector.fit(profiles.assign(fraud_label=0))
    detector.fit(profiles.loc[:, PROFILE_COLUMNS])
    scores = detector.score(profiles.loc[:, PROFILE_COLUMNS])
    assert scores.between(0, 1).all()


def test_scoring_is_deterministic_and_excludes_target_metadata() -> None:
    profiles = build_agent_behavior_profiles(_claims())
    config = AgentAnomalyConfig(minimum_history_count=1, training_fraction=0.5, n_estimators=20, random_state=7)
    first = score_agent_behavior(profiles, config)
    profiles["fraud_label"] = [0, 1, 1, 0]
    profiles["injected_patterns"] = ["", "", "multi_signal", ""]
    second = score_agent_behavior(profiles, config)
    pdt.assert_frame_equal(first, second)


def test_outlying_profile_scores_higher_than_typical_profiles() -> None:
    normal = pd.DataFrame([{column: 0.1 + index * 0.001 for column in PROFILE_COLUMNS} for index in range(30)])
    outlier = pd.DataFrame([{column: 15.0 for column in PROFILE_COLUMNS}])
    profiles = pd.concat([normal, outlier], ignore_index=True)
    detector = AgentAnomalyDetector(AgentAnomalyConfig(n_estimators=100, random_state=42))
    detector.fit(profiles)
    scores = detector.score(profiles)
    assert scores.iloc[-1] > scores.iloc[:-1].median()
