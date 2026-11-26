"""Point-in-time agent behavioral anomaly detection with Isolation Forest.

This module produces an unsupervised behavioral-anomaly signal. It is a review
signal only and does not establish that any claim or agent is fraudulent.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import select

from src.db.database import DEFAULT_DATABASE_PATH, create_engine, get_session_factory
from src.db.models import Claim

LOGGER = logging.getLogger(__name__)
LOSS_TYPES = ("water", "wind", "fire", "theft", "liability")


@dataclass(frozen=True, slots=True)
class AgentAnomalyConfig:
    """Deterministic settings for historical profiles and Isolation Forest."""

    random_state: int = 42
    contamination: float = 0.10
    minimum_history_count: int = 3
    training_fraction: float = 0.70
    n_estimators: int = 200
    output_path: Path = Path("data/processed/agent_anomaly_scores.parquet")

    def __post_init__(self) -> None:
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5].")
        if self.minimum_history_count < 1:
            raise ValueError("minimum_history_count must be positive.")
        if not 0 < self.training_fraction < 1:
            raise ValueError("training_fraction must be in (0, 1).")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")


PROFILE_COLUMNS = (
    "agent_historical_claim_count",
    "agent_historical_average_claim_amount",
    "agent_historical_claim_amount_variance",
    "agent_claims_previous_7_days",
    "agent_claims_previous_30_days",
    "agent_average_description_length",
    "agent_submission_weekday_mean",
    "agent_submission_weekday_std",
    "current_amount_deviation_from_agent",
    "current_description_length_deviation_from_agent",
    "current_submission_weekday_deviation_from_agent",
    "current_claim_type_historical_frequency",
    "agent_claim_type_water_proportion",
    "agent_claim_type_wind_proportion",
    "agent_claim_type_fire_proportion",
    "agent_claim_type_theft_proportion",
    "agent_claim_type_liability_proportion",
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    data = np.asarray(values, dtype=float)
    return float(data.mean()), float(data.std(ddof=0))


def build_agent_behavior_profiles(claims: pd.DataFrame) -> pd.DataFrame:
    """Build profiles using only claims submitted strictly before each claim.

    Claims from the same submission date are scored before that date is added
    to any agent's history, preventing same-day information leakage.
    """
    required = {"claim_id", "agent_id", "loss_type", "claim_amount", "narrative", "submitted_date"}
    missing = required.difference(claims.columns)
    if missing:
        raise ValueError(f"Claims missing profile inputs: {sorted(missing)}")
    ordered = claims.loc[:, list(required)].copy()
    ordered["submitted_date"] = pd.to_datetime(ordered["submitted_date"])
    ordered = ordered.sort_values(["submitted_date", "claim_id"], kind="stable")

    amounts: dict[str, list[float]] = defaultdict(list)
    descriptions: dict[str, list[float]] = defaultdict(list)
    weekdays: dict[str, list[float]] = defaultdict(list)
    submission_dates: dict[str, deque[pd.Timestamp]] = defaultdict(deque)
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows: list[dict[str, Any]] = []

    for submitted_date, day_claims in ordered.groupby("submitted_date", sort=True):
        for _, claim in day_claims.iterrows():
            agent_id = str(claim["agent_id"])
            amount = float(claim["claim_amount"])
            description_length = float(len(str(claim["narrative"])))
            weekday = float(pd.Timestamp(submitted_date).weekday())
            amount_mean, amount_std = _mean_std(amounts[agent_id])
            description_mean, description_std = _mean_std(descriptions[agent_id])
            weekday_mean, weekday_std = _mean_std(weekdays[agent_id])
            dates = submission_dates[agent_id]
            cutoff_30 = pd.Timestamp(submitted_date) - pd.DateOffset(days=30)
            cutoff_7 = pd.Timestamp(submitted_date) - pd.DateOffset(days=7)
            while dates and dates[0] < cutoff_30:
                dates.popleft()
            history_count = len(amounts[agent_id])
            profile: dict[str, Any] = {
                "claim_id": claim["claim_id"],
                "agent_id": agent_id,
                "submitted_date": pd.Timestamp(submitted_date),
                "agent_historical_claim_count": history_count,
                "agent_historical_average_claim_amount": amount_mean,
                "agent_historical_claim_amount_variance": amount_std**2,
                "agent_claims_previous_7_days": sum(date >= cutoff_7 for date in dates),
                "agent_claims_previous_30_days": len(dates),
                "agent_average_description_length": description_mean,
                "agent_submission_weekday_mean": weekday_mean,
                "agent_submission_weekday_std": weekday_std,
                "current_amount_deviation_from_agent": (amount - amount_mean) / amount_std if amount_std else 0.0,
                "current_description_length_deviation_from_agent": (description_length - description_mean) / description_std if description_std else 0.0,
                "current_submission_weekday_deviation_from_agent": (weekday - weekday_mean) / weekday_std if weekday_std else 0.0,
                "current_claim_type_historical_frequency": type_counts[agent_id][str(claim["loss_type"])] / history_count if history_count else 0.0,
            }
            for loss_type in LOSS_TYPES:
                profile[f"agent_claim_type_{loss_type}_proportion"] = type_counts[agent_id][loss_type] / history_count if history_count else 0.0
            rows.append(profile)

        # Update only after every same-day profile has been produced.
        for _, claim in day_claims.iterrows():
            agent_id = str(claim["agent_id"])
            amounts[agent_id].append(float(claim["claim_amount"]))
            descriptions[agent_id].append(float(len(str(claim["narrative"]))))
            weekdays[agent_id].append(float(pd.Timestamp(submitted_date).weekday()))
            submission_dates[agent_id].append(pd.Timestamp(submitted_date))
            type_counts[agent_id][str(claim["loss_type"])] += 1

    return pd.DataFrame(rows).sort_values("claim_id", kind="stable").reset_index(drop=True)


class AgentAnomalyDetector:
    """Isolation Forest wrapper with deterministic percentile-based scoring."""

    def __init__(self, config: AgentAnomalyConfig | None = None) -> None:
        self.config = config or AgentAnomalyConfig()
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self._training_raw_scores: np.ndarray | None = None

    def fit(self, profiles: pd.DataFrame) -> "AgentAnomalyDetector":
        """Fit only on selected unsupervised profile columns; labels are rejected."""
        forbidden = {"fraud_label", "injected_patterns", "duplicate_of_claim_id"}.intersection(profiles.columns)
        if forbidden:
            raise ValueError(f"Training profiles must not contain target metadata: {sorted(forbidden)}")
        self._model.fit(profiles.loc[:, PROFILE_COLUMNS])
        self._training_raw_scores = np.sort(self._model.score_samples(profiles.loc[:, PROFILE_COLUMNS]))
        return self

    def score(self, profiles: pd.DataFrame) -> pd.Series:
        """Return a 0–1 anomaly score; higher values mean more unusual behavior."""
        if self._training_raw_scores is None:
            raise RuntimeError("Fit the detector before scoring.")
        raw_scores = self._model.score_samples(profiles.loc[:, PROFILE_COLUMNS])
        percentile = np.searchsorted(self._training_raw_scores, raw_scores, side="right") / len(self._training_raw_scores)
        return pd.Series(np.clip(1.0 - percentile, 0.0, 1.0), index=profiles.index, name="agent_anomaly_score")


def score_agent_behavior(profiles: pd.DataFrame, config: AgentAnomalyConfig | None = None) -> pd.DataFrame:
    """Fit chronologically and score eligible claims without future training data.

    A fixed early training segment is used. Claims before the cutoff or without
    enough agent history receive a neutral score of 0.0 and are marked unscored.
    """
    config = config or AgentAnomalyConfig()
    ordered = profiles.sort_values(["submitted_date", "claim_id"], kind="stable").reset_index(drop=True)
    cutoff_index = max(1, int(len(ordered) * config.training_fraction))
    cutoff_date = ordered.loc[cutoff_index - 1, "submitted_date"]
    eligible = ordered["agent_historical_claim_count"] >= config.minimum_history_count
    training_mask = (ordered["submitted_date"] <= cutoff_date) & eligible
    if int(training_mask.sum()) < 2:
        raise ValueError("Insufficient historical profiles to fit Isolation Forest.")
    detector = AgentAnomalyDetector(config).fit(ordered.loc[training_mask, list(PROFILE_COLUMNS)])
    score_mask = (ordered["submitted_date"] > cutoff_date) & eligible
    output = ordered.loc[:, ["claim_id", "agent_id", "submitted_date"]].copy()
    output["agent_anomaly_scored"] = score_mask.astype(int)
    output["agent_anomaly_score"] = 0.0
    if score_mask.any():
        output.loc[score_mask, "agent_anomaly_score"] = detector.score(ordered.loc[score_mask, list(PROFILE_COLUMNS)]).to_numpy()
    return output.sort_values("claim_id", kind="stable").reset_index(drop=True)


def _claims_from_database(database_path: Path) -> pd.DataFrame:
    engine = create_engine(database_path)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        return pd.read_sql(
            select(Claim.claim_id, Claim.agent_id, Claim.loss_type, Claim.claim_amount, Claim.narrative, Claim.submitted_date),
            session.connection(),
        )


def build_scores_from_database(database_path: Path = DEFAULT_DATABASE_PATH, config: AgentAnomalyConfig | None = None) -> pd.DataFrame:
    """Build point-in-time profiles and unsupervised scores from SQLite claims."""
    return score_agent_behavior(build_agent_behavior_profiles(_claims_from_database(database_path)), config)


def write_agent_anomaly_scores(scores: pd.DataFrame, output_path: Path) -> Path:
    """Write score outputs without target labels or scenario metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_path, index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Isolation Forest agent-behavior anomaly scores.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=AgentAnomalyConfig().output_path)
    return parser.parse_args()


def main() -> None:
    """Run the unsupervised agent-anomaly scoring workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    scores = build_scores_from_database(args.database)
    output_path = write_agent_anomaly_scores(scores, args.output)
    LOGGER.info("Generated %s agent anomaly scores at %s", len(scores), output_path)


if __name__ == "__main__":
    main()
