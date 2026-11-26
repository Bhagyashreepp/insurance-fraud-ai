"""Integration tests for the local FastAPI claim-review contract."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app, get_pipeline_service
from src.explanations.service import ClaimExplanation
from src.risk.decision import AdvanceDecision, RiskFactor
from src.service.pipeline import ClaimAnalysis, ClaimFacts, ClaimNotFoundError


class _FakeService:
    def __init__(self) -> None:
        self.claim = ClaimFacts(
            claim_id="C-1", policy_id="P-1", customer_id="U-1", agent_id="A-1", contractor_id="K-1",
            loss_type="water", incident_date="2024-01-01", submitted_date="2024-01-02", claim_amount=1000.0,
        )
        decision = AdvanceDecision(
            fraud_probability=0.2, risk_level="low", risk_factors=[], recommended_advance=200.0,
            manual_review_required=False, decision_reasons=["Configured low-risk advance."],
        )
        self.analysis = ClaimAnalysis(
            claim=self.claim, fraud_probability=0.2, rule_risk_score=0.0, agent_anomaly_score=0.1, agent_anomaly_scored=0,
            agent_behavior_history={"historical_claim_count": 0, "recent_claim_frequency": 0, "historical_average_claim_amount": 0.0},
            writing_similarity={"maximum_prior_semantic_similarity": 0.1, "highly_similar_prior_claim_count": 0},
            duplicate_similarity=0.0, policy_active_indicator=1, document_incident_date_difference_days=1.0,
            triggered_rules=[], similar_prior_claim_references=[], decision=decision,
            explanation=ClaimExplanation(
                concise_claim_summary="C-1 has a low risk estimate.", key_risk_factors=["No configured evidence factors were triggered."],
                supporting_evidence=["Available claim-time evidence."], uncertainty_limitations="Not a finding of fraud.",
                recommended_investigator_action="Complete standard verification.",
            ),
        )

    def get_claim(self, claim_id: str) -> ClaimFacts:
        if claim_id != "C-1":
            raise ClaimNotFoundError("not found")
        return self.claim

    def analyze_claim(self, claim_id: str) -> ClaimAnalysis:
        self.get_claim(claim_id)
        return self.analysis

    def list_claim_ids(self, limit: int) -> list[str]:
        return ["C-1"]

    def dashboard_overview(self) -> dict[str, float | int]:
        return {"total_claims": 1, "low_risk_claims": 1, "medium_risk_claims": 0, "high_risk_claims": 0,
                "high_risk_claim_count": 0, "total_claim_exposure": 1000.0, "recommended_advance_exposure": 200.0}

    def demo_claims(self) -> dict[str, str]:
        return {"Legitimate-style claim": "C-1", "Writing-similarity case": "C-1", "Agent behavioral anomaly": "C-1",
                "Document inconsistency": "C-1", "High-risk multi-signal claim": "C-1"}


def _client() -> TestClient:
    app.dependency_overrides[get_pipeline_service] = _FakeService
    return TestClient(app)


def test_health_and_claim_endpoints() -> None:
    client = _client()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/claims/C-1").status_code == 200
    assert client.get("/claims/C-404").status_code == 404
    assert client.get("/claims?limit=1").json()["claim_ids"] == ["C-1"]
    assert client.get("/claims?limit=50000").status_code == 200


def test_analysis_related_endpoints_are_consistent_without_openai() -> None:
    client = _client()
    analysis = client.post("/claims/C-1/analyze")
    assert analysis.status_code == 200
    assert analysis.json()["decision"]["recommended_advance"] == 200.0
    risk = client.get("/claims/C-1/risk").json()
    decision = client.get("/claims/C-1/decision").json()
    explanation = client.get("/claims/C-1/explanation").json()
    assert risk["fraud_probability"] == decision["fraud_probability"] == 0.2
    assert explanation["uncertainty_limitations"] == "Not a finding of fraud."


def test_dashboard_support_endpoints() -> None:
    client = _client()
    overview = client.get("/dashboard/overview")
    assert overview.status_code == 200
    assert overview.json()["total_claims"] == 1
    assert len(client.get("/dashboard/demo-claims").json()["claims"]) == 5
