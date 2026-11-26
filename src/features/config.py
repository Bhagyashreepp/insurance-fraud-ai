"""Configuration for claim feature construction."""

from dataclasses import dataclass
from pathlib import Path

from src.models.narrative_similarity import NarrativeSimilarityConfig


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Explicit, versionable feature thresholds for the local MVP."""

    recent_frequency_window_days: int = 30
    repeated_customer_threshold: int = 3
    narrative_similarity: NarrativeSimilarityConfig = NarrativeSimilarityConfig()
    output_path: Path = Path("data/processed/features.parquet")

    def __post_init__(self) -> None:
        if self.recent_frequency_window_days <= 0:
            raise ValueError("recent_frequency_window_days must be positive.")
        if self.repeated_customer_threshold <= 0:
            raise ValueError("repeated_customer_threshold must be positive.")
