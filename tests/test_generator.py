"""Tests for synthetic insurance dataset generation."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.data.config import GeneratorConfig
from src.data.generator import generate_dataset, write_dataset
from src.data.models import PATTERN_NAMES


def _config(seed: int = 42) -> GeneratorConfig:
    return GeneratorConfig(seed=seed, agent_count=100, customer_count=400, policy_count=200, contractor_count=50, claim_count=1_000)


def test_row_counts() -> None:
    dataset = generate_dataset(_config())
    assert len(dataset.agents) == 100
    assert len(dataset.customers) == 400
    assert len(dataset.policies) == 200
    assert len(dataset.contractors) == 50
    assert len(dataset.claims) == 1_000
    assert len(dataset.documents) >= 1_000


def test_relationships_are_valid() -> None:
    dataset = generate_dataset(_config())
    assert set(dataset.policies.customer_id).issubset(set(dataset.customers.customer_id))
    assert set(dataset.claims.agent_id).issubset(set(dataset.agents.agent_id))
    assert set(dataset.claims.contractor_id).issubset(set(dataset.contractors.contractor_id))
    assert set(dataset.claims.policy_id).issubset(set(dataset.policies.policy_id))
    assert set(dataset.claims.customer_id).issubset(set(dataset.customers.customer_id))
    assert set(dataset.documents.claim_id).issubset(set(dataset.claims.claim_id))
    policy_customer = dataset.policies.set_index("policy_id").customer_id
    assert (dataset.claims.customer_id == dataset.claims.policy_id.map(policy_customer)).all()


def test_fraud_distribution_is_ten_percent() -> None:
    claims = generate_dataset(_config()).claims
    assert claims.fraud_label.sum() == 100
    assert claims.fraud_label.mean() == 0.10
    assert (claims.loc[claims.fraud_label == 0, "injected_patterns"] == "").all()


def test_generation_is_deterministic() -> None:
    first = generate_dataset(_config(seed=7))
    second = generate_dataset(_config(seed=7))
    for name, first_table in first.tables().items():
        pdt.assert_frame_equal(first_table, second.tables()[name])


def test_all_required_patterns_are_represented() -> None:
    claims = generate_dataset(_config()).claims
    injected = set(pattern for value in claims.injected_patterns for pattern in value.split("|") if pattern)
    assert set(PATTERN_NAMES).issubset(injected)
    multi = claims[claims.injected_patterns.str.contains("multi_signal")]
    assert not multi.empty
    assert multi.injected_patterns.str.count(r"\|").ge(4).all()


def test_normal_claims_have_unique_narratives_and_active_policy_coverage() -> None:
    dataset = generate_dataset(_config())
    normal = dataset.claims.loc[dataset.claims.fraud_label == 0]
    policies = dataset.policies.set_index("policy_id")
    assert normal.narrative.nunique() == len(normal)
    normal_policies = policies.loc[normal.policy_id]
    incident_dates = pd.to_datetime(normal.incident_date).to_numpy()
    assert (pd.to_datetime(normal_policies.effective_date).to_numpy() <= incident_dates).all()
    assert (incident_dates <= pd.to_datetime(normal_policies.expiration_date).to_numpy()).all()
    assert (normal_policies.policy_status.to_numpy() == "active").all()


def test_reused_narrative_scenario_contains_agent_specific_reuse() -> None:
    claims = generate_dataset(_config()).claims
    reused = claims.loc[claims.injected_patterns.str.contains("reused_narrative")]
    assert reused.agent_id.value_counts().max() > 1


def test_writer_outputs_csv_tables(tmp_path) -> None:
    dataset = generate_dataset(_config())
    paths = write_dataset(dataset, tmp_path / "raw")
    assert set(paths) == set(dataset.tables())
    reloaded = pd.read_csv(paths["claims"])
    assert len(reloaded) == len(dataset.claims)
