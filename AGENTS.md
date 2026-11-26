# Agent Guide

## Purpose

Build an MVP that assesses the trustworthiness of insurance-claim evidence for an advance-payment decision. The system produces probabilistic risk assessments, not findings of confirmed fraud, and it is not an AI-authorship detector.

## Intended architecture

Synthetic relational data in SQLite flows through feature engineering, deterministic checks, agent-level Isolation Forest anomaly scoring, narrative similarity scoring, and an XGBoost claim model. A risk aggregator and configurable advance-decision engine produce the final decision. FastAPI exposes the service and Streamlit supplies a review UI. OpenAI is used only to turn already-computed evidence into an explanation, with a deterministic fallback.

See `ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md` before changing implementation design.

## Planned setup and commands

The implementation will target Python 3.11+ and define dependencies in `pyproject.toml`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
```

## Coding conventions

- Use Python type hints, Pydantic models at API boundaries, and small single-purpose modules.
- Use `pathlib`, structured logging, explicit configuration objects, and deterministic random seeds.
- Keep business thresholds configurable; do not bury them in endpoint or UI code.
- Prefer unit tests for rules and feature transforms, plus focused integration tests for scoring/API flows.
- Keep generated data, trained models, and logs out of version control.

## Security and privacy

- Read secrets only from environment variables; never commit API keys, `.env` files, or personally identifying source data.
- Treat all claim text and documents as sensitive. Avoid logging raw narratives, document contents, customer addresses, or secrets.
- The OpenAI explanation request must receive only the minimum computed evidence needed. It may not be used to calculate the probability or decision.

## ML and data-leakage rules

- `fraud_label` is training/evaluation target only and must never be a model or rule input.
- Split data temporally before fitting feature statistics, encoders, similarity indexes, anomaly detectors, or the classifier. Features for a claim may use only information available by that claim's submission time.
- Never use post-outcome fields (investigation results, payment outcomes, future claims, or manually resolved labels) as features.
- Fit preprocessing and calibration on training data only. Persist feature schema, seed, model version, and threshold configuration.
- Similarity and duplicate features must exclude the claim itself and future claims.

## Communication rule

Describe results as a risk estimate, suspicious pattern, or review signal—never as confirmed fraud. Do not make legal, coverage, or claim-payment determinations.
