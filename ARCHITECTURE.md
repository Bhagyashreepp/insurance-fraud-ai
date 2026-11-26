# Architecture

## Objective and boundaries

The engine evaluates the evidence quality and suspicious-pattern risk of an incoming insurance claim to support an advance-payment workflow. It is not a fraud-adjudication, coverage, legal, or AI-text-detection system. Its output is an auditable, probabilistic recommendation that a human can review.

## System flow

```text
Synthetic data generator
        ↓
SQLite / SQLAlchemy relational store
        ↓
Point-in-time feature builder
        ├── deterministic fraud-pattern rules
        ├── Isolation Forest agent-behavior score
        └── sentence-transformer narrative similarity / duplicate signals
        ↓
XGBoost claim-risk classifier
        ↓
Risk aggregation and safeguards
        ↓
Configurable advance-payment decision engine
        ↓
Evidence-based explanation (OpenAI optional; deterministic fallback)
        ↓
FastAPI and Streamlit reviewer interface
```

## Components

| Component | Responsibility |
| --- | --- |
| `data_generation` | Produce seeded, realistic entity relationships and ~50,000 claims, with an approximately 90/10 legitimate-to-injected-suspicious mix. |
| `persistence` | SQLAlchemy models, SQLite initialization, repositories, and data access. |
| `features` | Build point-in-time numeric, categorical, temporal, relational, document, and text-similarity features. |
| `rules` | Deterministically identify duplicate, policy, date, amount, customer, contractor, and cross-field signals. |
| `anomaly` | Train and score Isolation Forest on historical agent behavior. |
| `similarity` | Generate narrative embeddings and retrieve prior semantically similar narratives; include only earlier claims. |
| `modeling` | Train, validate, calibrate, version, and score an XGBoost classifier. |
| `risk` | Combine score, hard safeguards, factors, and thresholds into a risk level. |
| `advance` | Apply transparent configurable limits to recommend an advance and review requirement. |
| `explanations` | Render supplied evidence into plain language; optionally call OpenAI after scoring is complete. |
| `api` / `ui` | Pydantic FastAPI contracts and Streamlit claim-review workflow. |

## Relational data model

`agents`, `customers`, `policies`, `contractors`, `claims`, and `documents` are separate tables with foreign keys. A claim belongs to one customer and policy, is submitted or handled by an agent, may reference a contractor, and has zero or more documents.

Claims include incident, submission, and processing timestamps; reported amount; loss type; narrative; and synthetic training target metadata kept separate from production scoring inputs. Documents store type, upload date, declared document date, and validation-relevant metadata—not actual sensitive files for the MVP.

## Synthetic scenarios

The generator creates normal relationships and injects labelled scenarios: exact/near duplicates, inflated amounts relative to comparable claims, abnormal agent submission rate, recycled narratives, date conflicts, inactive/invalid policies, concentrated contractor behavior, repeat-customer patterns, and multi-signal combinations. Scenario metadata is retained for test/evaluation traceability but excluded from model features.

## Feature and modeling design

Features include amount percentile versus prior comparable claims; agent/customer/contractor historical counts and rates; policy status at incident date; incident-to-submission and document-to-incident intervals; document completeness; duplicate hashes; prior semantic-similarity maximum; and agent anomaly score.

An Isolation Forest is fit on historical agent-period aggregates. Sentence-transformers creates narrative embeddings; cosine similarity is measured only against an index of prior narratives. XGBoost consumes engineered features only, never `fraud_label` or scenario metadata. A time-based train/validation/test split, fit-on-train preprocessing, and post-training calibration protect against leakage and make scores interpretable as probabilities.

## Decision contract

Risk aggregation preserves evidence and applies explicit overrides for severe data-quality issues. The advance engine is a pure, configurable function:

| Risk level | Decision |
| --- | --- |
| Low | Eligible for a configured advance, capped by claim amount and business maximum. |
| Medium | Limited advance or request for additional verification; human review can be configured. |
| High | No automatic advance and manual review required. |

Every decision returns its inputs, triggered factors, rule/configuration version, model version, and deterministic explanation. OpenAI may improve wording but cannot change probability, risk level, factors, advance amount, or review flag. If no API key is present or the call fails, the deterministic explanation is returned.

## Operational design

Configuration comes from environment variables and a versioned, non-secret settings file. The MVP uses local SQLite and filesystem model artifacts. Structured logs capture identifiers, component versions, and events while excluding raw sensitive content. The API validates inputs with Pydantic and the UI is a reviewer-facing presentation layer only.

## Non-goals for the MVP

- Production fraud adjudication or autonomous denial/payment.
- AI-generated-text classification.
- Real customer data ingestion or document OCR.
- Kubernetes, AWS, Airflow, Neo4j, Qdrant, Redis, Kafka, or comparable infrastructure.
