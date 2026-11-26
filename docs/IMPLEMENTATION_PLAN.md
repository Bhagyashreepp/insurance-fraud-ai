# Implementation Plan

## Delivery principles

Build a reproducible local MVP in vertical slices. Each phase should produce testable behavior, preserve point-in-time correctness, and maintain an evidence trail from raw claim facts to the final recommendation. No model prediction may be described as confirmed fraud.

## Phase 1 — Project foundation

1. Add `pyproject.toml`, package layout, `.gitignore`, configuration, logging, and test fixtures.
2. Define SQLAlchemy entities and Pydantic request/response models.
3. Add SQLite initialization and repository tests.
4. Document required environment variables, especially optional `OPENAI_API_KEY`.

**Acceptance:** a clean environment installs, database schemas initialize, and pytest runs.

## Phase 2 — Reproducible synthetic dataset

1. Create seeded Faker-based generators for agents, customers, policies, contractors, claims, and documents.
2. Generate chronological base claims before injecting suspicious scenarios.
3. Inject all nine required patterns, including multi-pattern cases, to about 10% of roughly 50,000 claims.
4. Store target labels and scenario metadata outside the scoring-feature contract.
5. Add distribution, referential-integrity, chronology, seed-reproducibility, and fraud-rate tests.

**Acceptance:** a single command creates a valid SQLite dataset with the intended entity relationships and scenario mix.

## Phase 3 — Deterministic evidence and features

1. Implement as-of-time data access so a claim only sees earlier records.
2. Implement transparent duplicate, amount, policy, date/document, customer frequency, contractor concentration, and cross-field rules.
3. Build reusable feature transforms with documented definitions and missing-value behavior.
4. Test every rule using both positive and negative fixtures; add leakage tests using deliberately future-only signals.

**Acceptance:** each triggered factor links to a specific observable fact and rule threshold.

## Phase 4 — Behavioral and text signals

1. Aggregate historical agent-period behavior and fit Isolation Forest on training-period aggregates.
2. Embed narratives using sentence-transformers and maintain a prior-claim similarity index suitable for the local dataset.
3. Exclude self-matches, future claims, labels, and injection metadata from these calculations.
4. Test deterministic preprocessing, temporal boundaries, and expected duplicate/near-duplicate detection.

**Acceptance:** anomaly and similarity scores are reproducible and valid point-in-time features.

## Phase 5 — Supervised risk model

1. Make a chronological train/validation/test split.
2. Train XGBoost with class-imbalance handling on approved feature columns only.
3. Evaluate precision/recall, PR-AUC, ROC-AUC, calibration, and threshold tradeoffs; record a baseline rule-only comparison.
4. Calibrate the selected model on held-out validation data; persist model, feature schema, seed, metrics, and version.
5. Add tests asserting that forbidden target/outcome columns cannot enter the feature matrix.

**Acceptance:** a reproducible training command produces a versioned artifact and evaluation report without leakage.

## Phase 6 — Risk and advance decisioning

1. Define a versioned threshold configuration and a structured evidence object.
2. Combine rules, anomaly/similarity signals, and calibrated model output into low/medium/high risk.
3. Implement the pure advance-decision policy: low-risk eligible, medium-risk limited or verification-required, high-risk no automatic advance/manual review.
4. Test boundaries, overrides, caps, and traceability of the six required output fields.

**Acceptance:** identical evidence and configuration always produce the same recommendation and rationale.

## Phase 7 — Explanation, API, and UI

1. Implement deterministic evidence-to-explanation templates first.
2. Add an optional OpenAI adapter that receives only already-calculated evidence and cannot modify the decision; fall back on failure or missing key.
3. Expose score and claim-retrieval endpoints through FastAPI with Pydantic schemas.
4. Add a Streamlit reviewer page showing claim facts, risk factors, decision, and explanation.
5. Add API and UI-adjacent integration tests.

**Acceptance:** a reviewer can score a claim locally and see stable results with or without an OpenAI key.

## Phase 8 — Quality and handoff

1. Add command-line workflows for generate, train, evaluate, and serve.
2. Validate logs do not expose raw sensitive content or secrets.
3. Complete README setup, API examples, model limitations, and operational runbook.
4. Run the full test suite and a reproducible end-to-end demo.

**Acceptance:** a new developer can install, generate data, train, run the service, and review a claim using documentation alone.

## Proposed package layout

```text
app/
  api/              # FastAPI routes and Pydantic schemas
  config/           # settings, thresholds, logging
  data/             # SQLAlchemy models, repositories, synthetic generation
  features/         # point-in-time feature builders
  rules/            # deterministic evidence checks
  modeling/         # anomaly, similarity, training, inference
  risk/              # aggregation and advance decisioning
  explanations/     # deterministic and optional OpenAI adapters
  ui/               # Streamlit application
tests/
docs/
artifacts/          # ignored generated models/reports
```

## Deliberate decisions to validate during implementation

- Business owners will need to approve threshold values, advance caps, and mandatory-review overrides.
- Synthetic labels evaluate the MVP mechanics; they must not be represented as real-world fraud performance.
- Model monitoring, authentication/authorization, data retention, and human-review workflows belong in a later production-hardening phase.
