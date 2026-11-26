"""Local, evidence-bound orchestration of the completed claim-risk pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import select

from src.db.database import DEFAULT_DATABASE_PATH, create_engine, get_session_factory
from src.db.models import Claim
from src.explanations.service import ClaimExplanation, ExplanationEvidence, generate_explanation
from src.models.narrative_similarity import DEFAULT_EMBEDDING_INDEX_PATH, NarrativeEmbeddingIndex
from src.models.rules import RuleEvaluation, apply_rule_baseline, evaluate_claim_rules
from src.risk.decision import AdvanceDecision, RiskEvidence, aggregate_risk_and_decide

LOGGER = logging.getLogger(__name__)


class ClaimNotFoundError(LookupError):
    """Raised when a requested claim does not exist in the local database."""


class PipelineNotReadyError(RuntimeError):
    """Raised when required local artifacts have not been generated."""


class ClaimFacts(BaseModel):
    claim_id: str
    policy_id: str
    customer_id: str
    agent_id: str
    contractor_id: str
    loss_type: str
    incident_date: str
    submitted_date: str
    claim_amount: float


class ClaimAnalysis(BaseModel):
    """Combined API-ready result; no labels or injection metadata are exposed."""

    claim: ClaimFacts
    fraud_probability: float
    rule_risk_score: float
    agent_anomaly_score: float
    agent_anomaly_scored: int
    agent_behavior_history: dict[str, float | int]
    writing_similarity: dict[str, float | int]
    duplicate_similarity: float
    policy_active_indicator: int
    document_incident_date_difference_days: float | None
    triggered_rules: list[str]
    similar_prior_claim_references: list[str]
    decision: AdvanceDecision
    explanation: ClaimExplanation


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    database_path: Path = DEFAULT_DATABASE_PATH
    feature_path: Path = Path("data/processed/features.parquet")
    anomaly_score_path: Path = Path("data/processed/agent_anomaly_scores.parquet")
    embedding_index_path: Path = DEFAULT_EMBEDDING_INDEX_PATH
    model_path: Path = Path("artifacts/fraud_risk_model/model.joblib")
    feature_schema_path: Path = Path("artifacts/fraud_risk_model/feature_schema.json")


class FraudPipelineService:
    """Coordinates local artifacts and deterministic evidence without logging text."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._features: pd.DataFrame | None = None
        self._anomaly_scores: pd.DataFrame | None = None
        self._model: object | None = None
        self._feature_schema: tuple[str, ...] | None = None
        self._embedding_index: NarrativeEmbeddingIndex | None = None
        self._analysis_cache: dict[str, ClaimAnalysis] = {}
        self._overview_cache: dict[str, float | int] | None = None

    def _load_artifacts(self) -> None:
        required = (
            self.config.feature_path,
            self.config.anomaly_score_path,
            self.config.embedding_index_path,
            self.config.model_path,
            self.config.feature_schema_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise PipelineNotReadyError(f"Required local artifacts are missing: {', '.join(missing)}")
        if self._features is None:
            self._features = pd.read_parquet(self.config.feature_path).set_index("claim_id", drop=False)
            self._anomaly_scores = pd.read_parquet(self.config.anomaly_score_path).set_index("claim_id", drop=False)
            self._model = joblib.load(self.config.model_path)
            payload = json.loads(self.config.feature_schema_path.read_text(encoding="utf-8"))
            self._feature_schema = tuple(payload["feature_columns"])
            self._embedding_index = NarrativeEmbeddingIndex.read(self.config.embedding_index_path)

    def _claim_record(self, claim_id: str) -> tuple[ClaimFacts, pd.Series]:
        engine = create_engine(self.config.database_path)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            frame = pd.read_sql(
                select(
                    Claim.claim_id,
                    Claim.policy_id,
                    Claim.customer_id,
                    Claim.agent_id,
                    Claim.contractor_id,
                    Claim.loss_type,
                    Claim.incident_date,
                    Claim.submitted_date,
                    Claim.claim_amount,
                    Claim.narrative,
                ).where(Claim.claim_id == claim_id),
                session.connection(),
            )
        if frame.empty:
            raise ClaimNotFoundError(f"Claim {claim_id} was not found.")
        row = frame.iloc[0]
        facts = ClaimFacts(
            claim_id=str(row.claim_id), policy_id=str(row.policy_id), customer_id=str(row.customer_id), agent_id=str(row.agent_id),
            contractor_id=str(row.contractor_id), loss_type=str(row.loss_type), incident_date=str(pd.Timestamp(row.incident_date).date()),
            submitted_date=str(pd.Timestamp(row.submitted_date).date()), claim_amount=float(row.claim_amount),
        )
        return facts, row

    def get_claim(self, claim_id: str) -> ClaimFacts:
        """Return reviewer-safe claim facts without a raw narrative or labels."""
        return self._claim_record(claim_id)[0]

    def _similarity(self, claim: pd.Series) -> tuple[dict[str, float | int], list[str]]:
        if self._embedding_index is None:
            raise PipelineNotReadyError("Narrative embedding index is unavailable.")
        return self._embedding_index.score_claim(claim)

    def analyze_claim(self, claim_id: str) -> ClaimAnalysis:
        """Run the ordered local pipeline and cache its deterministic result."""
        if claim_id in self._analysis_cache:
            return self._analysis_cache[claim_id]
        self._load_artifacts()
        facts, raw_claim = self._claim_record(claim_id)
        assert self._features is not None and self._anomaly_scores is not None and self._model is not None and self._feature_schema is not None
        if claim_id not in self._features.index:
            raise PipelineNotReadyError(f"Point-in-time feature row for {claim_id} is unavailable.")
        feature_row = self._features.loc[claim_id].copy()
        # The persisted local index filters to strictly earlier submission dates
        # before every query and never re-embeds historical narratives.
        similarity, references = self._similarity(raw_claim)
        for column, value in similarity.items():
            feature_row[column] = value
        ruled = apply_rule_baseline(pd.DataFrame([feature_row]))
        feature_row = ruled.iloc[0].copy()
        anomaly_score = float(self._anomaly_scores.loc[claim_id, "agent_anomaly_score"]) if claim_id in self._anomaly_scores.index else 0.0
        anomaly_scored = int(self._anomaly_scores.loc[claim_id, "agent_anomaly_scored"]) if claim_id in self._anomaly_scores.index else 0
        feature_row["agent_anomaly_score"] = anomaly_score
        matrix = pd.DataFrame([{column: float(feature_row.get(column, 0.0)) if pd.notna(feature_row.get(column, 0.0)) else 0.0 for column in self._feature_schema}])
        fraud_probability = float(self._model.predict_proba(matrix)[:, 1][0])  # type: ignore[union-attr]
        document_difference = feature_row.get("document_incident_date_difference_days")
        document_difference = None if pd.isna(document_difference) else float(document_difference)
        decision = aggregate_risk_and_decide(
            RiskEvidence(
                claim_id=claim_id,
                fraud_probability=fraud_probability,
                rule_risk_score=float(feature_row["rule_risk_score"]),
                agent_anomaly_score=anomaly_score,
                maximum_prior_semantic_similarity=float(similarity["maximum_prior_semantic_similarity"]),
                highly_similar_prior_claim_count=int(similarity["highly_similar_prior_claim_count"]),
                repeated_template_narrative_indicator=int(similarity["repeated_template_narrative_indicator"]),
                duplicate_similarity=float(feature_row["duplicate_similarity"]),
                policy_active_indicator=int(feature_row["policy_active_indicator"]),
                document_count=int(feature_row["document_count"]),
                document_incident_date_difference_days=document_difference,
                claim_amount=facts.claim_amount,
            )
        )
        rules: RuleEvaluation = evaluate_claim_rules(feature_row.to_dict())
        document_notes = [] if document_difference is None else [f"Document-to-incident date difference: {document_difference:.0f} days."]
        explanation = generate_explanation(
            ExplanationEvidence(
                claim_facts={"claim_id": facts.claim_id, "claim_amount": facts.claim_amount, "loss_type": facts.loss_type,
                             "incident_date": facts.incident_date, "submitted_date": facts.submitted_date},
                fraud_probability=fraud_probability, risk_level=decision.risk_level, triggered_rules=decision.risk_factors,
                agent_anomaly=anomaly_score, writing_similarity=similarity, similar_prior_claim_references=references,
                document_inconsistencies=document_notes, policy_consistency=bool(feature_row["policy_active_indicator"]),
                amount_anomaly=float(feature_row.get("claim_amount_vs_agent_historical_average", 0.0)), decision_recommendation=decision,
            )
        )
        analysis = ClaimAnalysis(
            claim=facts, fraud_probability=fraud_probability, rule_risk_score=float(feature_row["rule_risk_score"]),
            agent_anomaly_score=anomaly_score, agent_anomaly_scored=anomaly_scored,
            agent_behavior_history={
                "historical_claim_count": int(feature_row.get("agent_historical_claim_count", 0)),
                "recent_claim_frequency": int(feature_row.get("agent_recent_claim_frequency", 0)),
                "historical_average_claim_amount": float(feature_row.get("agent_historical_average_claim_amount", 0.0)),
            },
            writing_similarity=similarity, duplicate_similarity=float(feature_row["duplicate_similarity"]),
            policy_active_indicator=int(feature_row["policy_active_indicator"]), document_incident_date_difference_days=document_difference,
            triggered_rules=list(rules.rule_reasons), similar_prior_claim_references=references, decision=decision, explanation=explanation,
        )
        self._analysis_cache[claim_id] = analysis
        LOGGER.info("Completed local claim analysis", extra={"claim_id": claim_id, "risk_level": decision.risk_level})
        return analysis

    def list_claim_ids(self, limit: int = 250) -> list[str]:
        engine = create_engine(self.config.database_path)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            values = session.scalars(select(Claim.claim_id).order_by(Claim.submitted_date.desc()).limit(limit)).all()
        return [str(value) for value in values]

    def dashboard_overview(self) -> dict[str, float | int]:
        """Return local portfolio totals from the same persisted scoring artifacts."""
        if self._overview_cache is not None:
            return self._overview_cache
        self._load_artifacts()
        assert self._features is not None and self._anomaly_scores is not None and self._model is not None and self._feature_schema is not None
        features = apply_rule_baseline(self._features.reset_index(drop=True)).merge(
            self._anomaly_scores.loc[:, ["claim_id", "agent_anomaly_score"]].reset_index(drop=True), on="claim_id", how="left", validate="one_to_one"
        )
        features["agent_anomaly_score"] = features["agent_anomaly_score"].fillna(0.0)
        matrix = pd.DataFrame(
            {
                column: pd.to_numeric(features.get(column, 0.0), errors="coerce").fillna(0.0)
                if isinstance(features.get(column, 0.0), pd.Series)
                else 0.0
                for column in self._feature_schema
            }
        )
        probabilities = self._model.predict_proba(matrix)[:, 1]  # type: ignore[union-attr]
        decisions = [
            aggregate_risk_and_decide(
                RiskEvidence(
                    fraud_probability=float(probability), rule_risk_score=float(row.rule_risk_score),
                    agent_anomaly_score=float(row.agent_anomaly_score), duplicate_similarity=float(row.duplicate_similarity),
                    policy_active_indicator=int(row.policy_active_indicator), document_count=int(row.document_count),
                    document_incident_date_difference_days=None if pd.isna(row.document_incident_date_difference_days) else float(row.document_incident_date_difference_days),
                    claim_amount=float(row.claim_amount),
                )
            )
            for probability, row in zip(probabilities, features.itertuples(index=False), strict=True)
        ]
        levels = pd.Series([decision.risk_level for decision in decisions]).value_counts()
        self._overview_cache = {
            "total_claims": len(features), "low_risk_claims": int(levels.get("low", 0)), "medium_risk_claims": int(levels.get("medium", 0)),
            "high_risk_claims": int(levels.get("high", 0)), "high_risk_claim_count": int(levels.get("high", 0)),
            "total_claim_exposure": float(features["claim_amount"].sum()),
            "recommended_advance_exposure": round(float(sum(decision.recommended_advance for decision in decisions)), 2),
        }
        return self._overview_cache

    def demo_claims(self) -> dict[str, str]:
        """Return evidence-accurate, stable dashboard examples without labels."""
        return {
            "Genuine low-risk case — CLM-013104": "CLM-013104",
            "Writing-pattern review case — CLM-000102": "CLM-000102",
            "Agent-specific writing-pattern case — CLM-015214": "CLM-015214",
            "High model-risk case — CLM-018567": "CLM-018567",
            "Safeguard-driven review case — CLM-048504": "CLM-048504",
        }
