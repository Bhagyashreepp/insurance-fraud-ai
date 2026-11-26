"""Focused dashboard tests for selected-claim resolution and stale-result prevention."""

from __future__ import annotations

from urllib.error import HTTPError

from app.ui import streamlit_app


class _Column:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events

    def metric(self, label: str, value: str) -> None:
        self.events.append((label, value))


class _StreamlitRecorder:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.metrics: list[tuple[str, str]] = []
        self.json_values: list[object] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def subheader(self, _: str) -> None:
        return None

    def columns(self, count: int) -> list[_Column]:
        return [_Column(self.metrics) for _ in range(count)]

    def caption(self, _: str) -> None:
        return None

    def markdown(self, _: str) -> None:
        return None

    def json(self, value: object) -> None:
        self.json_values.append(value)

    def write(self, *_: object) -> None:
        return None

    def info(self, _: str) -> None:
        return None

    def success(self, _: str) -> None:
        return None


def _valid_analysis(claim_id: str) -> dict:
    return {
        "claim": {"claim_id": claim_id, "agent_id": "AGT-1"},
        "fraud_probability": 0.2,
        "decision": {"risk_level": "low", "recommended_advance": 200.0, "manual_review_required": False, "risk_factors": []},
        "agent_anomaly_score": 0.1,
        "agent_anomaly_scored": 1,
        "agent_behavior_history": {"historical_claim_count": 4, "recent_claim_frequency": 1, "historical_average_claim_amount": 900.0},
        "writing_similarity": {
            "maximum_prior_semantic_similarity": 0.1,
            "average_same_agent_prior_semantic_similarity": 0.05,
            "maximum_other_agent_prior_semantic_similarity": 0.1,
            "highly_similar_prior_claim_count": 0,
            "repeated_template_narrative_indicator": 0,
        },
        "similar_prior_claim_references": [],
        "triggered_rules": [],
        "explanation": {
            "concise_claim_summary": "Summary.", "key_risk_factors": [], "supporting_evidence": [],
            "uncertainty_limitations": "Limitations.", "recommended_investigator_action": "Review.",
        },
    }


def test_selected_claim_renders_only_when_response_claim_id_matches(monkeypatch) -> None:
    recorder = _StreamlitRecorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)
    monkeypatch.setattr(streamlit_app, "_api_post", lambda _: _valid_analysis("CLM-048504"))
    streamlit_app._show_review("CLM-048504")
    assert not recorder.errors
    assert [label for label, _ in recorder.metrics] == [
        "Calibrated model probability",
        "Final risk level",
        "Advance recommendation",
        "Manual-review requirement",
    ]
    assert recorder.json_values[0]["Agent ID"] == "AGT-1"
    assert recorder.json_values[1]["Average same-agent prior semantic similarity"] == 0.05
    assert recorder.json_values[1]["Maximum other-agent prior semantic similarity"] == 0.1


def test_mismatched_or_missing_claim_hides_stale_analysis(monkeypatch) -> None:
    recorder = _StreamlitRecorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)
    monkeypatch.setattr(streamlit_app, "_api_post", lambda _: _valid_analysis("CLM-013104"))
    streamlit_app._show_review("CLM-048504")
    assert recorder.errors == ["Claim analysis did not match the selected claim. No result is displayed."]
    assert not recorder.metrics

    recorder = _StreamlitRecorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)
    monkeypatch.setattr(
        streamlit_app,
        "_api_post",
        lambda _: (_ for _ in ()).throw(HTTPError("http://local/claims/missing", 404, "Not Found", {}, None)),
    )
    streamlit_app._show_review("CLM-NOT-FOUND")
    assert recorder.errors == ["Claim not found."]
    assert not recorder.metrics
