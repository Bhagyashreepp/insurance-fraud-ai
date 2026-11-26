"""Chronological, calibrated XGBoost claim-risk modeling.

The model estimates suspicious-pattern risk in the synthetic training data.  It
is a review signal, not a finding of fraud or a claim/payment determination.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import select
from xgboost import XGBClassifier

from src.db.database import DEFAULT_DATABASE_PATH, create_engine, get_session_factory
from src.db.models import Claim
from src.features.engine import build_features_from_database

from .agent_anomaly import build_agent_behavior_profiles, score_agent_behavior
from .rules import apply_rule_baseline


FORBIDDEN_FEATURE_TERMS = (
    "fraud_label",
    "scenario",
    "inject",
    "duplicate_of_claim_id",
    "outcome",
    "payment",
    "investigation",
    "resolution",
)
# Rule risk and policy eligibility remain transparent deterministic safeguards.
# Keeping them out of the supervised matrix preserves the meaning of
# ``fraud_probability`` as the calibrated XGBoost suspicious-pattern estimate;
# safeguards can then independently elevate final risk without being folded
# back into that probability.
NON_FEATURE_COLUMNS = frozenset(
    {"claim_id", "submitted_date", "rule_reasons", "rule_risk_score", "policy_active_indicator"}
)


@dataclass(frozen=True, slots=True)
class FraudRiskModelConfig:
    """Versioned training, split, calibration, and artifact settings."""

    random_state: int = 42
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    classification_threshold: float = 0.50
    n_estimators: int = 250
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 2.0
    subsample: float = 0.90
    colsample_bytree: float = 0.90
    artifact_dir: Path = Path("artifacts/fraud_risk_model")
    model_version: str = "fraud-risk-xgboost-v1"

    def __post_init__(self) -> None:
        if not np.isclose(self.train_fraction + self.validation_fraction + self.test_fraction, 1.0):
            raise ValueError("Train, validation, and test fractions must sum to 1.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All split fractions must be positive.")
        if not 0 < self.classification_threshold < 1:
            raise ValueError("classification_threshold must be between zero and one.")
        if self.n_estimators < 1 or self.max_depth < 1:
            raise ValueError("n_estimators and max_depth must be positive.")
        if self.learning_rate <= 0 or self.min_child_weight <= 0:
            raise ValueError("learning_rate and min_child_weight must be positive.")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Rows split by non-overlapping submission-date ranges."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FraudRiskTrainingResult:
    """Model outputs and artifact locations from one reproducible training run."""

    model: CalibratedClassifierCV
    feature_columns: tuple[str, ...]
    metrics: dict[str, Any]
    feature_importance: dict[str, float]
    artifact_paths: dict[str, Path]
    test_predictions: pd.DataFrame


def _is_forbidden_feature(column: str) -> bool:
    normalized = column.lower()
    return any(term in normalized for term in FORBIDDEN_FEATURE_TERMS)


def select_model_feature_columns(frame: pd.DataFrame, requested: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return approved numeric feature columns, rejecting leakage-prone inputs."""
    candidates = list(requested) if requested is not None else list(frame.columns)
    forbidden = sorted(column for column in candidates if _is_forbidden_feature(column))
    if forbidden:
        raise ValueError(f"Forbidden target, scenario, or post-decision feature columns: {forbidden}")
    missing = sorted(set(candidates).difference(frame.columns))
    if missing:
        raise ValueError(f"Requested feature columns are missing: {missing}")
    columns = tuple(
        column
        for column in candidates
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    )
    if not columns:
        raise ValueError("No approved numeric model feature columns are available.")
    return columns


def chronological_train_validation_test_split(frame: pd.DataFrame, config: FraudRiskModelConfig | None = None) -> ChronologicalSplit:
    """Split by whole submission dates so no date appears in multiple cohorts."""
    config = config or FraudRiskModelConfig()
    if "submitted_date" not in frame:
        raise ValueError("A submitted_date column is required for chronological splitting.")
    ordered = frame.copy()
    ordered["submitted_date"] = pd.to_datetime(ordered["submitted_date"])
    dates = np.sort(ordered["submitted_date"].dropna().unique())
    if len(dates) < 3:
        raise ValueError("At least three distinct submission dates are required for chronological splitting.")
    train_end = max(1, int(len(dates) * config.train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (config.train_fraction + config.validation_fraction)))
    if validation_end >= len(dates):
        validation_end = len(dates) - 1
    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]
    if not len(validation_dates) or not len(test_dates):
        raise ValueError("Split fractions leave an empty validation or test date range.")
    return ChronologicalSplit(
        train=ordered.loc[ordered["submitted_date"].isin(train_dates)].copy(),
        validation=ordered.loc[ordered["submitted_date"].isin(validation_dates)].copy(),
        test=ordered.loc[ordered["submitted_date"].isin(test_dates)].copy(),
    )


def _evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate classification, ranking, confusion, and calibration metrics."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    unique_labels = y_true.nunique()
    calibration_true, calibration_predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if unique_labels == 2 else None,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if unique_labels == 2 else None,
        "confusion_matrix": matrix.astype(int).tolist(),
        "calibration": {
            "brier_score": float(brier_score_loss(y_true, probabilities)),
            "mean_predicted_probability": [float(value) for value in calibration_predicted],
            "observed_positive_rate": [float(value) for value in calibration_true],
        },
    }


def _calibrator(classifier: XGBClassifier) -> CalibratedClassifierCV:
    """Calibrate an already-fitted training estimator on validation data only."""
    return CalibratedClassifierCV(estimator=FrozenEstimator(classifier), method="sigmoid", cv=None)


def _validate_target(split: ChronologicalSplit) -> None:
    for name, cohort in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        if "fraud_label" not in cohort:
            raise ValueError("fraud_label is required as the training/evaluation target.")
        if cohort["fraud_label"].nunique() != 2:
            raise ValueError(f"The chronological {name} split must contain both target classes.")


def train_fraud_risk_model(frame: pd.DataFrame, config: FraudRiskModelConfig | None = None) -> FraudRiskTrainingResult:
    """Train XGBoost on approved features and calibrate only on held-out validation rows."""
    config = config or FraudRiskModelConfig()
    if "fraud_label" not in frame:
        raise ValueError("fraud_label is required as the training target but cannot be a model feature.")
    feature_columns = select_model_feature_columns(frame.drop(columns=["fraud_label"]))
    split = chronological_train_validation_test_split(frame, config)
    _validate_target(split)
    x_train = split.train.loc[:, feature_columns]
    y_train = split.train["fraud_label"].astype(int)
    x_validation = split.validation.loc[:, feature_columns]
    y_validation = split.validation["fraud_label"].astype(int)
    x_test = split.test.loc[:, feature_columns]
    y_test = split.test["fraud_label"].astype(int)
    positive_count = int(y_train.sum())
    scale_pos_weight = float((len(y_train) - positive_count) / positive_count)
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=config.random_state,
        n_jobs=1,
        tree_method="hist",
    )
    classifier.fit(x_train, y_train)
    calibrated_model = _calibrator(classifier)
    calibrated_model.fit(x_validation, y_validation)
    validation_probabilities = calibrated_model.predict_proba(x_validation)[:, 1]
    test_probabilities = calibrated_model.predict_proba(x_test)[:, 1]
    rule_probabilities = split.test["rule_risk_score"].astype(float).clip(0, 1).to_numpy()
    metrics = {
        "model_version": config.model_version,
        "split_counts": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "split_date_ranges": {
            name: {"start": str(cohort["submitted_date"].min().date()), "end": str(cohort["submitted_date"].max().date())}
            for name, cohort in (("train", split.train), ("validation", split.validation), ("test", split.test))
        },
        "class_imbalance": {"train_positive_count": positive_count, "scale_pos_weight": scale_pos_weight},
        "validation": _evaluate_probabilities(y_validation, validation_probabilities, config.classification_threshold),
        "test": _evaluate_probabilities(y_test, test_probabilities, config.classification_threshold),
        "rule_only_test": _evaluate_probabilities(y_test, rule_probabilities, config.classification_threshold),
    }
    feature_importance = {
        column: float(importance)
        for column, importance in zip(feature_columns, classifier.feature_importances_, strict=True)
    }
    predictions = split.test.loc[:, ["claim_id", "submitted_date", "fraud_label"]].copy()
    predictions["fraud_probability"] = test_probabilities
    predictions["rule_only_risk_score"] = rule_probabilities
    artifact_paths = persist_fraud_risk_artifacts(calibrated_model, feature_columns, config, metrics, feature_importance)
    return FraudRiskTrainingResult(
        model=calibrated_model,
        feature_columns=feature_columns,
        metrics=metrics,
        feature_importance=feature_importance,
        artifact_paths=artifact_paths,
        test_predictions=predictions,
    )


def persist_fraud_risk_artifacts(
    model: CalibratedClassifierCV,
    feature_columns: tuple[str, ...],
    config: FraudRiskModelConfig,
    metrics: dict[str, Any],
    feature_importance: dict[str, float],
) -> dict[str, Path]:
    """Persist model, approved schema, configuration, metrics, and importance."""
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": config.artifact_dir / "model.joblib",
        "feature_schema": config.artifact_dir / "feature_schema.json",
        "training_config": config.artifact_dir / "training_config.json",
        "metrics": config.artifact_dir / "metrics.json",
        "feature_importance": config.artifact_dir / "feature_importance.json",
    }
    joblib.dump(model, paths["model"])
    paths["feature_schema"].write_text(json.dumps({"feature_columns": feature_columns}, indent=2), encoding="utf-8")
    paths["training_config"].write_text(json.dumps(asdict(config), default=str, indent=2), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["feature_importance"].write_text(json.dumps(feature_importance, indent=2), encoding="utf-8")
    return paths


def _claims_for_modeling(database_path: Path) -> pd.DataFrame:
    """Load the minimal fields needed for target joining and anomaly scoring."""
    engine = create_engine(database_path)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        return pd.read_sql(
            select(Claim.claim_id, Claim.agent_id, Claim.loss_type, Claim.claim_amount, Claim.narrative, Claim.submitted_date, Claim.fraud_label),
            session.connection(),
        )


def build_supervised_training_frame(database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Build the approved feature frame, then join the target after feature construction.

    Only the target is selected here.  Scenario and injection metadata never
    enter the frame; agent anomaly scores are calculated from their own safe,
    point-in-time source columns.
    """
    claims = _claims_for_modeling(database_path)
    # The supervised schema intentionally uses non-semantic claim features;
    # live semantic evidence is supplied from the persisted point-in-time
    # embedding index to the deterministic decision layer.  Avoid rebuilding
    # the expensive all-pairs similarity matrix during model training.
    features = build_features_from_database(database_path, include_similarity=False)
    rules = apply_rule_baseline(features)
    anomaly_scores = score_agent_behavior(build_agent_behavior_profiles(claims.drop(columns=["fraud_label"])))
    output = rules.merge(anomaly_scores.loc[:, ["claim_id", "agent_anomaly_score"]], on="claim_id", how="left", validate="one_to_one")
    output["agent_anomaly_score"] = output["agent_anomaly_score"].fillna(0.0)
    return output.merge(claims.loc[:, ["claim_id", "fraud_label"]], on="claim_id", how="inner", validate="one_to_one")
