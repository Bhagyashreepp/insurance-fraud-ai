"""Tests for SQLite schema creation and generated-data seeding."""

from __future__ import annotations

from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from src.data.config import GeneratorConfig
from src.data.generator import generate_dataset, write_dataset
from src.db.database import create_engine, create_schema, get_session_factory
from src.db.models import Agent, Claim, Contractor, Customer, Document, Policy
from src.db.seed import database_row_counts, seed_database


def _seeded_database(tmp_path):
    dataset = generate_dataset(
        GeneratorConfig(seed=42, agent_count=100, customer_count=400, policy_count=200, contractor_count=50, claim_count=1_000)
    )
    raw_dir = tmp_path / "raw"
    write_dataset(dataset, raw_dir)
    database_path = tmp_path / "insurance.db"
    counts = seed_database(raw_dir, database_path)
    return dataset, database_path, counts


def test_database_creation_creates_all_tables(tmp_path) -> None:
    engine = create_engine(tmp_path / "empty.db")
    create_schema(engine)
    assert set(inspect(engine).get_table_names()) == {"agents", "customers", "policies", "contractors", "claims", "documents"}


def test_seed_inserts_all_records(tmp_path) -> None:
    dataset, database_path, inserted = _seeded_database(tmp_path)
    persisted = database_row_counts(create_engine(database_path))
    assert inserted == {name: len(table) for name, table in dataset.tables().items()}
    assert persisted == inserted


def test_foreign_key_relationships_and_constraints(tmp_path) -> None:
    _, database_path, _ = _seeded_database(tmp_path)
    session_factory = get_session_factory(create_engine(database_path))
    with session_factory() as session:
        claim = session.scalar(select(Claim).limit(1))
        assert claim is not None
        assert claim.agent.agent_id == claim.agent_id
        assert claim.customer.customer_id == claim.customer_id
        assert claim.policy.policy_id == claim.policy_id
        assert claim.contractor.contractor_id == claim.contractor_id
        assert len(claim.documents) >= 1

    with session_factory.begin() as session:
        session.add(
            Document(
                document_id="DOC-INVALID",
                claim_id="CLM-NOT-REAL",
                document_type="invoice",
                document_date=claim.incident_date,
                uploaded_date=claim.submitted_date,
                document_reference="REF-INVALID",
            )
        )
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
        else:
            raise AssertionError("SQLite foreign-key enforcement did not reject an invalid document claim_id")


def test_query_claims(tmp_path) -> None:
    _, database_path, _ = _seeded_database(tmp_path)
    session_factory = get_session_factory(create_engine(database_path))
    with session_factory() as session:
        claims = session.scalars(select(Claim).where(Claim.fraud_label == 1).limit(10)).all()
        assert len(claims) == 10
        assert all(claim.injected_patterns for claim in claims)


def test_query_agent_historical_claims(tmp_path) -> None:
    _, database_path, _ = _seeded_database(tmp_path)
    session_factory = get_session_factory(create_engine(database_path))
    with session_factory() as session:
        agent = session.scalar(select(Agent).join(Agent.claims).limit(1))
        assert agent is not None
        history = session.scalars(select(Claim).where(Claim.agent_id == agent.agent_id).order_by(Claim.submitted_date)).all()
        assert history
        assert [claim.submitted_date for claim in history] == sorted(claim.submitted_date for claim in history)
