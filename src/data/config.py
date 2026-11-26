"""Configuration for reproducible synthetic insurance-data generation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Controls dataset size, seed, and output location.

    The default entity counts are the requested MVP scale. Smaller counts are
    supported for tests; the suspicious-claim count remains 10 percent rounded
    down, with a minimum of one claim when claims are requested.
    """

    seed: int = 42
    agent_count: int = 5_000
    customer_count: int = 20_000
    policy_count: int = 10_000
    contractor_count: int = 2_000
    claim_count: int = 50_000
    output_dir: Path = Path("data/raw")
    suspicious_rate: float = 0.10

    def __post_init__(self) -> None:
        counts = (
            self.agent_count,
            self.customer_count,
            self.policy_count,
            self.contractor_count,
            self.claim_count,
        )
        if any(count <= 0 for count in counts):
            raise ValueError("All entity counts must be positive.")
        if not 0 < self.suspicious_rate < 1:
            raise ValueError("suspicious_rate must be between 0 and 1.")

    @property
    def suspicious_claim_count(self) -> int:
        return max(1, int(self.claim_count * self.suspicious_rate))
