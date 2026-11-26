"""Local reviewer dashboard backed exclusively by the FastAPI service."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEMO_DESCRIPTIONS = {
    "Genuine low-risk case — CLM-013104": "Low model probability with no current evidence safeguard requiring review.",
    "Writing-pattern review case — CLM-000102": "Low model probability with strong narrative-reuse evidence.",
    "Agent-specific writing-pattern case — CLM-015214": "Strong same-agent and cross-agent narrative reuse evidence.",
    "High model-risk case — CLM-018567": "High calibrated model probability with multiple independent evidence signals.",
    "Safeguard-driven review case — CLM-048504": "Low calibrated model probability, but an insurance safeguard independently requires review.",
}


@st.cache_data(ttl=30)
def _api_get(path: str) -> dict:
    request = Request(f"{API_URL}{path}", method="GET")
    with urlopen(request, timeout=60) as response:  # noqa: S310 - configured local service URL
        return json.loads(response.read().decode("utf-8"))


def _api_post(path: str) -> dict:
    request = Request(f"{API_URL}{path}", method="POST")
    with urlopen(request, timeout=120) as response:  # noqa: S310 - configured local service URL
        return json.loads(response.read().decode("utf-8"))


def _show_overview() -> None:
    st.subheader("Overview")
    overview = _api_get("/dashboard/overview")
    columns = st.columns(4)
    columns[0].metric("Total claims", overview["total_claims"])
    columns[1].metric("High-risk claims", overview["high_risk_claim_count"])
    columns[2].metric("Claim exposure", f"${overview['total_claim_exposure']:,.0f}")
    columns[3].metric("Recommended advance exposure", f"${overview['recommended_advance_exposure']:,.0f}")
    st.caption(
        f"Low: {overview['low_risk_claims']:,} · Medium: {overview['medium_risk_claims']:,} · High: {overview['high_risk_claims']:,}. "
        "These are elevated-risk review categories, not fraud findings."
    )


def _show_review(claim_id: str) -> None:
    try:
        analysis = _api_post(f"/claims/{claim_id}/analyze")
    except HTTPError as exc:
        if exc.code == 404:
            st.error("Claim not found.")
        else:
            st.error(f"Claim analysis is unavailable: {exc}")
        return
    if analysis.get("claim", {}).get("claim_id") != claim_id:
        st.error("Claim analysis did not match the selected claim. No result is displayed.")
        return
    decision = analysis["decision"]
    st.subheader(f"Claim review: {claim_id}")
    probability, risk, advance, review = st.columns(4)
    probability.metric("Calibrated model probability", f"{analysis['fraud_probability']:.1%}")
    risk.metric("Final risk level", decision["risk_level"].upper())
    advance.metric("Advance recommendation", f"${decision['recommended_advance']:,.2f}")
    review.metric("Manual-review requirement", "Requires investigation" if decision["manual_review_required"] else "Standard verification")
    st.caption("Safeguards and business caps can require manual review independently of calibrated model probability.")
    st.markdown("#### Evidence signals")
    st.markdown("##### Agent behavioral evidence")
    agent_history = analysis.get("agent_behavior_history", {})
    st.json(
        {
            "Agent ID": analysis["claim"]["agent_id"],
            "Agent anomaly score": analysis["agent_anomaly_score"],
            "Anomaly score available/scored": bool(analysis["agent_anomaly_scored"]),
            "Historical agent claim count": agent_history.get("historical_claim_count", "Not available"),
            "Recent agent submission frequency": agent_history.get("recent_claim_frequency", "Not available"),
            "Historical average claim amount": agent_history.get("historical_average_claim_amount", "Not available"),
        }
    )
    st.markdown("##### Writing-pattern analysis")
    similarity = analysis["writing_similarity"]
    st.json(
        {
            "Maximum prior semantic similarity": similarity.get("maximum_prior_semantic_similarity", 0.0),
            "Average same-agent prior semantic similarity": similarity.get("average_same_agent_prior_semantic_similarity", 0.0),
            "Maximum other-agent prior semantic similarity": similarity.get("maximum_other_agent_prior_semantic_similarity", 0.0),
            "Highly similar prior claim count": similarity.get("highly_similar_prior_claim_count", 0),
            "Repeated narrative/template indicator": bool(similarity.get("repeated_template_narrative_indicator", 0)),
        }
    )
    st.caption("Semantic narrative similarity identifies repeated narrative patterns and unusual reuse; it is not an AI detector.")
    st.markdown("##### Similar prior claims")
    st.write(analysis["similar_prior_claim_references"] or "No qualifying prior claims met the configured similarity threshold.")
    st.markdown("##### Triggered rules and review factors")
    st.write(analysis["triggered_rules"] or "No deterministic rule signals were triggered.")
    st.write([factor["evidence"] for factor in decision["risk_factors"]] or "No additional review factors were triggered.")
    st.markdown("##### Explanation")
    explanation = analysis["explanation"]
    st.write(explanation["concise_claim_summary"])
    st.write("Key risk factors:", explanation["key_risk_factors"])
    st.write("Supporting evidence:", explanation["supporting_evidence"])
    st.info(explanation["uncertainty_limitations"])
    st.success(explanation["recommended_investigator_action"])


def main() -> None:
    st.set_page_config(page_title="Claim Evidence Review", layout="wide")
    st.title("Claim Evidence Review")
    st.caption("Local decision-support dashboard. Evidence indicates review priority; it does not confirm fraud.")
    try:
        _show_overview()
        demo = _api_get("/dashboard/demo-claims")["claims"]
        claims = _api_get("/claims?limit=50000")["claim_ids"]
        selected_demo = st.selectbox("Demo claims (observable-signal examples)", list(demo), format_func=lambda label: f"{label}: {demo[label]}")
        st.caption(DEMO_DESCRIPTIONS.get(selected_demo, "Selected claim review."))
        claim_options = list(dict.fromkeys([demo[selected_demo], *claims]))
        claim_id = st.selectbox(
            "Claim ID",
            claim_options,
            accept_new_options=True,
            key=f"claim_id_{selected_demo}",
            help="Search all local claim IDs or enter an ID to verify it directly.",
        )
        if claim_id:
            _show_review(str(claim_id))
    except (HTTPError, URLError, KeyError, ValueError) as exc:
        st.error(f"Local API is unavailable or returned invalid data: {exc}")
        st.code("uvicorn app.api.main:app --reload")


if __name__ == "__main__":
    main()
