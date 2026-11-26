"""Tests for chronological, target-safe supervised risk modeling."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.frozen import FrozenEstimator

from src.models import fraud_risk
from src.models.fraud_risk import (
    FraudRiskModelConfig,
    chronological_train_validation_test_split,
    select_model_feature_columns,
    train_fraud_risk_model,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(60):
        label = int(index % 5 == 0)
        rows.append(
            {
                "claim_id": f"C{index:03d}",
                "submitted_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "claim_amount_vs_agent_historical_average": 3.0 if label else 0.9 + (index % 3) * 0.1,
                "agent_behavioral_deviation_score": 2.0 if label else 0.1,
                "agent_anomaly_score": 0.8 if label else 0.1,
                "rule_risk_score": 0.7 if label else 0.05,
                "maximum_prior_semantic_similarity": 0.95 if label else 0.1,
                "duplicate_similarity": float(label),
                "policy_active_indicator": 0 if label else 1,
                "document_incident_date_difference_days": 50.0 if label else 1.0,
                "claim_frequency": float(index % 4),
                "fraud_label": label,
            }
        )
    return pd.DataFrame(rows)


def _config(tmp_path) -> FraudRiskModelConfig:
    return FraudRiskModelConfig(
        n_estimators=25,
        max_depth=3,
        artifact_dir=tmp_path / "artifacts",
        random_state=7,
    )


def test_chronological_split_keeps_submission_dates_in_separate_cohorts(tmp_path) -> None:
    split = chronological_train_validation_test_split(_training_frame(), _config(tmp_path))
    train_dates = set(split.train.submitted_date)
    validation_dates = set(split.validation.submitted_date)
    test_dates = set(split.test.submitted_date)
    assert not train_dates & validation_dates
    assert not train_dates & test_dates
    assert not validation_dates & test_dates
    assert max(train_dates) < min(validation_dates) < min(test_dates)


@pytest.mark.parametrize("forbidden_column", ["fraud_label", "scenario_type", "injected_patterns", "payment_outcome"])
def test_forbidden_columns_are_rejected_from_model_schema(forbidden_column: str) -> None:
    frame = _training_frame().assign(**{forbidden_column: 1})
    with pytest.raises(ValueError, match="Forbidden"):
        select_model_feature_columns(frame, requested=["rule_risk_score", forbidden_column])


def test_training_calibrates_probabilities_and_persists_safe_artifacts(tmp_path) -> None:
    result = train_fraud_risk_model(_training_frame(), _config(tmp_path))
    assert result.test_predictions["fraud_probability"].between(0, 1).all()
    assert "fraud_label" not in result.feature_columns
    assert "rule_risk_score" not in result.feature_columns
    assert "policy_active_indicator" not in result.feature_columns
    assert all("scenario" not in column and "inject" not in column for column in result.feature_columns)
    assert set(result.feature_importance) == set(result.feature_columns)
    assert result.metrics["test"]["pr_auc"] is not None
    assert result.metrics["test"]["roc_auc"] is not None
    assert len(result.metrics["test"]["confusion_matrix"]) == 2
    assert "calibration" in result.metrics["test"]
    assert set(result.artifact_paths) == {"model", "feature_schema", "training_config", "metrics", "feature_importance"}
    assert all(path.is_file() for path in result.artifact_paths.values())
    schema = json.loads(result.artifact_paths["feature_schema"].read_text(encoding="utf-8"))
    assert schema["feature_columns"] == list(result.feature_columns)
    loaded_model = joblib.load(result.artifact_paths["model"])
    matrix = _training_frame().loc[:, result.feature_columns]
    assert loaded_model.predict_proba(matrix)[:, 1].min() >= 0
    assert loaded_model.predict_proba(matrix)[:, 1].max() <= 1


def test_xgboost_uses_train_and_frozen_calibration_uses_validation_only(tmp_path, monkeypatch) -> None:
    frame = _training_frame()
    config = _config(tmp_path)
    split = chronological_train_validation_test_split(frame, config)
    xgboost_fit_indices: list[list[int]] = []
    calibration_fit_indices: list[list[int]] = []
    original_xgboost_fit = fraud_risk.XGBClassifier.fit
    original_calibration_fit = fraud_risk.CalibratedClassifierCV.fit

    def record_xgboost_fit(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        xgboost_fit_indices.append(list(x.index))
        return original_xgboost_fit(self, x, y, *args, **kwargs)

    def record_calibration_fit(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        calibration_fit_indices.append(list(x.index))
        return original_calibration_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(fraud_risk.XGBClassifier, "fit", record_xgboost_fit)
    monkeypatch.setattr(fraud_risk.CalibratedClassifierCV, "fit", record_calibration_fit)
    result = train_fraud_risk_model(frame, config)

    assert xgboost_fit_indices == [list(split.train.index)]
    assert calibration_fit_indices == [list(split.validation.index)]
    assert not set(calibration_fit_indices[0]).intersection(split.test.index)
    assert isinstance(result.model.estimator, FrozenEstimator)


def test_future_or_metadata_values_cannot_affect_model_feature_matrix(tmp_path) -> None:
    baseline = _training_frame()
    changed = baseline.copy()
    changed.loc[changed.index >= 48, "claim_amount_vs_agent_historical_average"] = 999_999.0
    changed["injected_patterns"] = "synthetic_only"
    with pytest.raises(ValueError, match="Forbidden"):
        train_fraud_risk_model(changed, _config(tmp_path))
    # The train-only cohort itself has not been modified, confirming chronological boundaries.
    split = chronological_train_validation_test_split(baseline, _config(tmp_path))
    changed_split = chronological_train_validation_test_split(changed.drop(columns=["injected_patterns"]), _config(tmp_path))
    np.testing.assert_array_equal(
        split.train["claim_amount_vs_agent_historical_average"].to_numpy(),
        changed_split.train["claim_amount_vs_agent_historical_average"].to_numpy(),
    )
