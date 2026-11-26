"""Generate a deterministic, relational synthetic insurance claims dataset.

The fraud label and injected-pattern metadata are ground truth for future
training/evaluation only. They must never be admitted to a prediction feature
matrix.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from faker import Faker

from .config import GeneratorConfig
from .models import PATTERN_NAMES, SyntheticDataset

LOGGER = logging.getLogger(__name__)
LOSS_TYPES = ("water", "wind", "fire", "theft", "liability")
DOCUMENT_TYPES = ("repair_estimate", "photo_inventory", "incident_report", "invoice")
NARRATIVE_TEMPLATES = {
    "water": "Water damage was reported in the {room} after a plumbing leak on {incident_date}.",
    "wind": "Wind damage affected the {room} during a storm on {incident_date}.",
    "fire": "Smoke and fire damage was reported in the {room} on {incident_date}.",
    "theft": "Property was reported stolen from the {room} on {incident_date}.",
    "liability": "An injury-related incident was reported near the {room} on {incident_date}.",
}
AMOUNT_MEDIANS = {"water": 7_500, "wind": 9_500, "fire": 18_000, "theft": 4_000, "liability": 12_000}


def _ids(prefix: str, count: int) -> np.ndarray:
    return np.array([f"{prefix}-{number:06d}" for number in range(1, count + 1)])


def _date_string(value: np.datetime64 | pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _append_pattern(claims: pd.DataFrame, indices: Iterable[int], pattern: str) -> None:
    for index in indices:
        current = claims.at[index, "injected_patterns"]
        claims.at[index, "injected_patterns"] = pattern if not current else f"{current}|{pattern}"
        claims.at[index, "fraud_label"] = 1


def _build_agents(config: GeneratorConfig, faker: Faker, rng: np.random.Generator) -> pd.DataFrame:
    regions = np.array(["west", "south", "midwest", "northeast"])
    return pd.DataFrame(
        {
            "agent_id": _ids("AGT", config.agent_count),
            "agent_name": [faker.name() for _ in range(config.agent_count)],
            "region": rng.choice(regions, size=config.agent_count),
            "hire_date": pd.Timestamp("2016-01-01")
            + pd.to_timedelta(rng.integers(0, 2_920, size=config.agent_count), unit="D"),
            "active": rng.random(config.agent_count) > 0.03,
        }
    )


def _build_customers(config: GeneratorConfig, faker: Faker, rng: np.random.Generator) -> pd.DataFrame:
    states = np.array(["CA", "TX", "FL", "NY", "IL", "WA", "CO", "GA"])
    return pd.DataFrame(
        {
            "customer_id": _ids("CUS", config.customer_count),
            "customer_name": [faker.name() for _ in range(config.customer_count)],
            "state": rng.choice(states, size=config.customer_count),
            "customer_since": pd.Timestamp("2012-01-01")
            + pd.to_timedelta(rng.integers(0, 4_745, size=config.customer_count), unit="D"),
        }
    )


def _build_policies(config: GeneratorConfig, customer_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    # Active synthetic policies deliberately cover the complete base-claim
    # window (2024-01 through early 2026).  Inactive-policy evidence is
    # injected separately, rather than accidentally applied to normal claims.
    start = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, size=config.policy_count), unit="D")
    term_days = rng.choice(np.array([1_095, 1_460]), size=config.policy_count, p=[0.70, 0.30])
    status = rng.choice(np.array(["active", "inactive", "cancelled"]), size=config.policy_count, p=[0.88, 0.08, 0.04])
    return pd.DataFrame(
        {
            "policy_id": _ids("POL", config.policy_count),
            "customer_id": rng.choice(customer_ids, size=config.policy_count, replace=False),
            "policy_type": rng.choice(np.array(["homeowners", "renters", "small_business"]), size=config.policy_count, p=[0.72, 0.20, 0.08]),
            "effective_date": start,
            "expiration_date": start + pd.to_timedelta(term_days, unit="D"),
            "policy_status": status,
            "coverage_limit": rng.choice(np.array([50_000, 100_000, 250_000, 500_000]), size=config.policy_count),
        }
    )


def _build_contractors(config: GeneratorConfig, faker: Faker, rng: np.random.Generator) -> pd.DataFrame:
    specialties = np.array(["restoration", "roofing", "fire_remediation", "contents", "general_repair"])
    return pd.DataFrame(
        {
            "contractor_id": _ids("CON", config.contractor_count),
            "contractor_name": [faker.company() for _ in range(config.contractor_count)],
            "specialty": rng.choice(specialties, size=config.contractor_count),
            "licensed": rng.random(config.contractor_count) > 0.04,
            "active": rng.random(config.contractor_count) > 0.03,
        }
    )


def _build_base_claims(
    config: GeneratorConfig, policies: pd.DataFrame, agent_ids: np.ndarray, contractor_ids: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    claim_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, size=config.claim_count), unit="D")
    loss_types = rng.choice(np.array(LOSS_TYPES), size=config.claim_count)
    amounts = np.array([rng.lognormal(np.log(AMOUNT_MEDIANS[loss_type]), 0.45) for loss_type in loss_types]).round(2)
    active_policies = policies.loc[policies["policy_status"] == "active"].reset_index(drop=True)
    policy_rows = active_policies.iloc[rng.integers(0, len(active_policies), size=config.claim_count)].reset_index(drop=True)
    claims = pd.DataFrame(
        {
            "claim_id": _ids("CLM", config.claim_count),
            "policy_id": policy_rows["policy_id"],
            "customer_id": policy_rows["customer_id"],
            "agent_id": rng.choice(agent_ids, size=config.claim_count),
            "contractor_id": rng.choice(contractor_ids, size=config.claim_count),
            "loss_type": loss_types,
            "incident_date": claim_dates,
            "submitted_date": claim_dates + pd.to_timedelta(rng.integers(1, 15, size=config.claim_count), unit="D"),
            "claim_amount": amounts,
            "fraud_label": np.zeros(config.claim_count, dtype=np.int8),
            "injected_patterns": "",
            "duplicate_of_claim_id": "",
        }
    )
    # Normal claims must not look like a large repeated-template cluster.
    # These deterministic detail combinations retain realistic structured
    # claim content without consuming additional RNG state or changing the
    # seeded injected-scenario selection below.
    observations = np.array(
        [
            "localized staining on the lower wall and baseboard",
            "moisture affecting trim, flooring, and nearby stored contents",
            "surface damage limited to a small section of the reported area",
            "displaced materials and visible debris requiring documentation",
            "damage to fixtures with a preliminary repair assessment needed",
            "a contained area requiring drying, cleaning, and follow-up inspection",
            "damage photographed by the policyholder before temporary mitigation",
            "visible impact marks and loose material near the reported location",
            "a repairable condition with no additional affected rooms reported",
            "damage requiring a contractor estimate after the initial inspection",
        ]
    )
    causes = np.array(
        [
            "The policyholder described the sequence of events during the first contact.",
            "The initial report identified the affected property and the apparent cause.",
            "The loss description was recorded before any repair work was requested.",
            "The claimant reported that the condition was noticed shortly after the incident.",
            "The file notes a prompt report and a request for a documented inspection.",
            "The claimant described the location, visible condition, and immediate safety steps.",
            "A preliminary review was requested before selecting a repair provider.",
            "The report distinguishes the observed condition from estimated repair scope.",
            "The policyholder provided a concise timeline and identified the affected items.",
            "The claim was submitted with a request to review supporting documentation.",
        ]
    )
    affected_items = np.array(
        [
            "Drywall and trim were identified for a closer review.",
            "Flooring and adjacent fixtures were listed for inspection.",
            "Stored personal property was separated from the repair estimate.",
            "Exterior materials were documented before any work was scheduled.",
            "Cabinetry and nearby contents were noted in the first report.",
            "Electrical fixtures were excluded pending a qualified inspection.",
            "Interior finishes were identified as the primary repair scope.",
            "A single appliance area was documented for follow-up assessment.",
            "Ceiling material and wall finishes were photographed separately.",
            "The report distinguishes emergency mitigation from permanent repairs.",
        ]
    )
    next_steps = np.array(
        [
            "The claimant requested a review before authorizing any repair work.",
            "Supporting photographs and an estimate will be reviewed with the file.",
            "A contractor visit was requested after coverage information is confirmed.",
            "The claimant was asked to retain documentation of temporary measures.",
            "The file notes that repair scope remains subject to verification.",
            "A follow-up inspection was requested to confirm the reported condition.",
            "The claimant requested guidance on the next evidence-submission step.",
            "The report records that no final repair amount has been agreed.",
            "The initial file was submitted for an evidence and policy review.",
            "A documented estimate is requested before any advance recommendation.",
        ]
    )
    claims["narrative"] = [
        (
            NARRATIVE_TEMPLATES[loss_type].format(
                room=rng.choice(["kitchen", "basement", "garage", "bedroom", "living room"]), incident_date=_date_string(incident_date)
            )
            + f" Initial observation: {observations[int(str(claim_id).split('-')[-1]) % len(observations)]}."
            + f" {causes[(int(str(claim_id).split('-')[-1]) // len(observations)) % len(causes)]}"
            + f" {affected_items[(int(str(claim_id).split('-')[-1]) // (len(observations) * len(causes))) % len(affected_items)]}"
            + f" {next_steps[(int(str(claim_id).split('-')[-1]) // (len(observations) * len(causes) * len(affected_items))) % len(next_steps)]}"
            + f" Claim reference: {claim_id}."
        )
        for claim_id, loss_type, incident_date in zip(claims["claim_id"], claims["loss_type"], claims["incident_date"], strict=True)
    ]
    return claims.sort_values(["submitted_date", "claim_id"], kind="stable").reset_index(drop=True)


def _build_documents(claims: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    counts = rng.integers(1, 4, size=len(claims))
    claim_positions = np.repeat(np.arange(len(claims)), counts)
    documents = pd.DataFrame(
        {
            "document_id": _ids("DOC", len(claim_positions)),
            "claim_id": claims["claim_id"].to_numpy()[claim_positions],
            "document_type": rng.choice(np.array(DOCUMENT_TYPES), size=len(claim_positions)),
            "document_date": claims["incident_date"].to_numpy()[claim_positions]
            + pd.to_timedelta(rng.integers(-2, 5, size=len(claim_positions)), unit="D"),
            "uploaded_date": claims["submitted_date"].to_numpy()[claim_positions]
            + pd.to_timedelta(rng.integers(0, 5, size=len(claim_positions)), unit="D"),
            "document_reference": [f"REF-{number:08d}" for number in range(1, len(claim_positions) + 1)],
        }
    )
    return documents


def _inject_patterns(
    claims: pd.DataFrame,
    documents: pd.DataFrame,
    policies: pd.DataFrame,
    agents: pd.DataFrame,
    contractors: pd.DataFrame,
    suspicious_count: int,
    rng: np.random.Generator,
) -> None:
    if suspicious_count < len(PATTERN_NAMES):
        raise ValueError("At least 90 claims are required so every pattern can be represented at 10%.")
    selected = rng.choice(claims.index.to_numpy(), size=suspicious_count, replace=False)
    groups = np.array_split(selected, len(PATTERN_NAMES))
    pattern_groups = dict(zip(PATTERN_NAMES, groups, strict=True))

    duplicate_indices = pattern_groups["duplicate_claim"]
    source_indices = rng.choice(claims.index.difference(duplicate_indices), size=len(duplicate_indices), replace=False)
    source = claims.loc[source_indices].reset_index(drop=True)
    claims.loc[duplicate_indices, ["policy_id", "customer_id", "loss_type", "incident_date", "narrative", "claim_amount"]] = source[
        ["policy_id", "customer_id", "loss_type", "incident_date", "narrative", "claim_amount"]
    ].to_numpy()
    claims.loc[duplicate_indices, "duplicate_of_claim_id"] = source["claim_id"].to_numpy()
    _append_pattern(claims, duplicate_indices, "duplicate_claim")

    inflated = pattern_groups["amount_inflation"]
    claims.loc[inflated, "claim_amount"] = (claims.loc[inflated, "claim_amount"] * rng.uniform(2.5, 5.0, len(inflated))).round(2)
    _append_pattern(claims, inflated, "amount_inflation")

    abnormal = pattern_groups["abnormal_agent_frequency"]
    hot_agents = rng.choice(agents["agent_id"].to_numpy(), size=max(1, min(10, len(abnormal))), replace=False)
    claims.loc[abnormal, "agent_id"] = np.resize(hot_agents, len(abnormal))
    _append_pattern(claims, abnormal, "abnormal_agent_frequency")

    reused = pattern_groups["reused_narrative"]
    canonical = "Emergency mitigation and repair were requested after a reported water loss at the residence."
    claims.loc[reused, "narrative"] = canonical
    # A small, labelled subset deliberately reuses writing through the same
    # agents so the MVP can demonstrate agent-specific narrative patterns.
    reuse_agents = agents["agent_id"].to_numpy()[:max(1, min(8, len(reused)))]
    claims.loc[reused, "agent_id"] = np.resize(reuse_agents, len(reused))
    _append_pattern(claims, reused, "reused_narrative")

    mismatch = pattern_groups["incident_document_date_mismatch"]
    mismatch_ids = set(claims.loc[mismatch, "claim_id"])
    mismatch_mask = documents["claim_id"].isin(mismatch_ids)
    documents.loc[mismatch_mask, "document_date"] = (
        documents.loc[mismatch_mask, "document_date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(75, "D")
    )
    _append_pattern(claims, mismatch, "incident_document_date_mismatch")

    invalid = pattern_groups["invalid_or_inactive_policy"]
    inactive_policies = policies.loc[policies["policy_status"] != "active", ["policy_id", "customer_id"]]
    replacements = inactive_policies.iloc[np.resize(np.arange(len(inactive_policies)), len(invalid))]
    claims.loc[invalid, ["policy_id", "customer_id"]] = replacements[["policy_id", "customer_id"]].to_numpy()
    _append_pattern(claims, invalid, "invalid_or_inactive_policy")

    contractor_indices = pattern_groups["suspicious_contractor_association"]
    suspicious_contractors = rng.choice(contractors["contractor_id"].to_numpy(), size=max(1, min(8, len(contractor_indices))), replace=False)
    claims.loc[contractor_indices, "contractor_id"] = np.resize(suspicious_contractors, len(contractor_indices))
    _append_pattern(claims, contractor_indices, "suspicious_contractor_association")

    repeated = pattern_groups["repeated_customer_claims"]
    repeat_customers = rng.choice(policies["customer_id"].to_numpy(), size=max(1, min(15, len(repeated))), replace=False)
    policy_by_customer = policies.drop_duplicates("customer_id").set_index("customer_id")
    chosen_customers = np.resize(repeat_customers, len(repeated))
    claims.loc[repeated, "customer_id"] = chosen_customers
    claims.loc[repeated, "policy_id"] = policy_by_customer.loc[chosen_customers, "policy_id"].to_numpy()
    _append_pattern(claims, repeated, "repeated_customer_claims")

    multi = pattern_groups["multi_signal"]
    multi_source = claims.loc[rng.choice(claims.index.difference(multi), size=len(multi), replace=False)].reset_index(drop=True)
    claims.loc[multi, ["policy_id", "customer_id", "loss_type", "incident_date", "narrative"]] = multi_source[
        ["policy_id", "customer_id", "loss_type", "incident_date", "narrative"]
    ].to_numpy()
    claims.loc[multi, "duplicate_of_claim_id"] = multi_source["claim_id"].to_numpy()
    claims.loc[multi, "claim_amount"] = (claims.loc[multi, "claim_amount"] * 3.5).round(2)
    claims.loc[multi, "contractor_id"] = suspicious_contractors[0]
    multi_ids = set(claims.loc[multi, "claim_id"])
    multi_mask = documents["claim_id"].isin(multi_ids)
    documents.loc[multi_mask, "document_date"] = (
        documents.loc[multi_mask, "document_date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(90, "D")
    )
    for pattern in ("multi_signal", "duplicate_claim", "amount_inflation", "incident_document_date_mismatch", "suspicious_contractor_association"):
        _append_pattern(claims, multi, pattern)


def _inject_unlabelled_writing_review_fixture(claims: pd.DataFrame) -> None:
    """Create a tiny reproducible review-only narrative cohort for the local demo.

    These records deliberately remain unlabelled: semantic reuse is a review
    signal, not a fraud finding.  They make the distinction visible without
    making normal claims broadly repetitive or exposing injection metadata to
    scoring code.
    """
    target = claims.index[claims["claim_id"] == "CLM-000102"]
    if not len(target) or int(claims.at[target[0], "fraud_label"]) != 0:
        return
    target_date = pd.Timestamp(claims.at[target[0], "submitted_date"])
    prior = claims.loc[
        (claims["fraud_label"] == 0) & (pd.to_datetime(claims["submitted_date"]) < target_date)
    ].sort_values(["submitted_date", "claim_id"], kind="stable").head(12).index
    fixture = pd.Index([target[0]]).append(prior)
    for index in fixture:
        claim_id = str(claims.at[index, "claim_id"])
        claims.at[index, "narrative"] = (
            "Emergency mitigation and repair were requested after a reported water loss at the residence. "
            f"Review file marker: {claim_id}."
        )


def generate_dataset(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Create a seeded, relational dataset in memory.

    The function does not write files; use :func:`write_dataset` or the module
    CLI for CSV output.
    """
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)
    Faker.seed(config.seed)
    faker = Faker("en_US")
    faker.seed_instance(config.seed)

    agents = _build_agents(config, faker, rng)
    customers = _build_customers(config, faker, rng)
    policies = _build_policies(config, customers["customer_id"].to_numpy(), rng)
    contractors = _build_contractors(config, faker, rng)
    claims = _build_base_claims(config, policies, agents["agent_id"].to_numpy(), contractors["contractor_id"].to_numpy(), rng)
    documents = _build_documents(claims, rng)
    _inject_patterns(claims, documents, policies, agents, contractors, config.suspicious_claim_count, rng)
    _inject_unlabelled_writing_review_fixture(claims)
    return SyntheticDataset(agents, customers, policies, contractors, claims, documents)


def write_dataset(dataset: SyntheticDataset, output_dir: Path) -> dict[str, Path]:
    """Write each entity table as CSV and return its paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in dataset.tables().items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, date_format="%Y-%m-%d")
        paths[name] = path
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic insurance claim data.")
    parser.add_argument("--claims", type=int, default=50_000, help="Number of claims to generate (default: 50000).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="CSV output directory.")
    return parser.parse_args()


def main() -> None:
    """Run the dataset generator CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    config = GeneratorConfig(claim_count=args.claims, seed=args.seed, output_dir=args.output_dir)
    dataset = generate_dataset(config)
    paths = write_dataset(dataset, config.output_dir)
    LOGGER.info("Generated %s claims (%s labelled suspicious) in %s", len(dataset.claims), int(dataset.claims["fraud_label"].sum()), config.output_dir)
    LOGGER.info("Wrote tables: %s", ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
