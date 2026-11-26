"""FastAPI endpoints around the local, evidence-bound fraud-risk pipeline."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.explanations.service import ClaimExplanation
from src.risk.decision import AdvanceDecision
from src.service.pipeline import ClaimAnalysis, ClaimFacts, ClaimNotFoundError, FraudPipelineService, PipelineNotReadyError

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Insurance Claim Risk Review API", version="0.1.0")


class HealthResponse(BaseModel):
    status: str


class ClaimListResponse(BaseModel):
    claim_ids: list[str]


class DashboardOverviewResponse(BaseModel):
    total_claims: int
    low_risk_claims: int
    medium_risk_claims: int
    high_risk_claims: int
    high_risk_claim_count: int
    total_claim_exposure: float
    recommended_advance_exposure: float


class DemoClaimsResponse(BaseModel):
    claims: dict[str, str]


class RiskResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: str
    risk_factors: list[dict[str, str]]


@lru_cache(maxsize=1)
def get_pipeline_service() -> FraudPipelineService:
    return FraudPipelineService()


def _analysis_or_http_error(claim_id: str, service: FraudPipelineService) -> ClaimAnalysis:
    try:
        return service.analyze_claim(claim_id)
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PipelineNotReadyError as exc:
        LOGGER.warning("Local pipeline is not ready", extra={"claim_id": claim_id})
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary
        LOGGER.exception("Claim analysis failed", extra={"claim_id": claim_id})
        raise HTTPException(status_code=500, detail="Claim analysis could not be completed.") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/claims", response_model=ClaimListResponse)
def list_claims(limit: int = Query(default=250, ge=1, le=50_000), service: FraudPipelineService = Depends(get_pipeline_service)) -> ClaimListResponse:
    return ClaimListResponse(claim_ids=service.list_claim_ids(limit))


@app.get("/claims/{claim_id}", response_model=ClaimFacts)
def get_claim(claim_id: str, service: FraudPipelineService = Depends(get_pipeline_service)) -> ClaimFacts:
    try:
        return service.get_claim(claim_id)
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/claims/{claim_id}/analyze", response_model=ClaimAnalysis)
def analyze_claim(claim_id: str, service: FraudPipelineService = Depends(get_pipeline_service)) -> ClaimAnalysis:
    return _analysis_or_http_error(claim_id, service)


@app.get("/claims/{claim_id}/risk", response_model=RiskResponse)
def get_risk(claim_id: str, service: FraudPipelineService = Depends(get_pipeline_service)) -> RiskResponse:
    analysis = _analysis_or_http_error(claim_id, service)
    return RiskResponse(fraud_probability=analysis.fraud_probability, risk_level=analysis.decision.risk_level,
                        risk_factors=[factor.model_dump() for factor in analysis.decision.risk_factors])


@app.get("/claims/{claim_id}/explanation", response_model=ClaimExplanation)
def get_explanation(claim_id: str, service: FraudPipelineService = Depends(get_pipeline_service)) -> ClaimExplanation:
    return _analysis_or_http_error(claim_id, service).explanation


@app.get("/claims/{claim_id}/decision", response_model=AdvanceDecision)
def get_decision(claim_id: str, service: FraudPipelineService = Depends(get_pipeline_service)) -> AdvanceDecision:
    return _analysis_or_http_error(claim_id, service).decision


@app.get("/dashboard/overview", response_model=DashboardOverviewResponse)
def dashboard_overview(service: FraudPipelineService = Depends(get_pipeline_service)) -> DashboardOverviewResponse:
    try:
        return DashboardOverviewResponse(**service.dashboard_overview())
    except PipelineNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/dashboard/demo-claims", response_model=DemoClaimsResponse)
def dashboard_demo_claims(service: FraudPipelineService = Depends(get_pipeline_service)) -> DemoClaimsResponse:
    try:
        return DemoClaimsResponse(claims=service.demo_claims())
    except PipelineNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
