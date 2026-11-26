"""Dataset contracts used by the synthetic generator.

These are simple data contracts, not database models. Persistence is outside
this milestone.
"""

from dataclasses import dataclass
from typing import Final

import pandas as pd


PATTERN_NAMES: Final[tuple[str, ...]] = (
    "duplicate_claim",
    "amount_inflation",
    "abnormal_agent_frequency",
    "reused_narrative",
    "incident_document_date_mismatch",
    "invalid_or_inactive_policy",
    "suspicious_contractor_association",
    "repeated_customer_claims",
    "multi_signal",
)

ENTITY_TABLES: Final[tuple[str, ...]] = (
    "agents",
    "customers",
    "policies",
    "contractors",
    "claims",
    "documents",
)


@dataclass(slots=True)
class SyntheticDataset:
    """In-memory synthetic entity tables."""

    agents: pd.DataFrame
    customers: pd.DataFrame
    policies: pd.DataFrame
    contractors: pd.DataFrame
    claims: pd.DataFrame
    documents: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return tables using their stable output file names."""
        return {name: getattr(self, name) for name in ENTITY_TABLES}
