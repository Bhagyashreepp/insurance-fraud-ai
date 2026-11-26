"""Point-in-time narrative similarity signals using local sentence embeddings.

These signals identify semantic reuse and writing-pattern review signals.  They
are not an assessment of whether a narrative was authored by AI and do not use
claim labels or synthetic-scenario metadata.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from sqlalchemy import select

from src.db.database import DEFAULT_DATABASE_PATH, create_engine, get_session_factory
from src.db.models import Claim


class NarrativeEmbedder(Protocol):
    """Minimal local embedding interface, kept injectable for deterministic tests."""

    def encode(self, narratives: list[str], *, batch_size: int) -> np.ndarray:
        """Return one embedding row per supplied narrative."""


class SentenceTransformerEmbedder:
    """Lazy adapter around sentence-transformers running on the local CPU."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = None

    def encode(self, narratives: list[str], *, batch_size: int) -> np.ndarray:
        """Embed narratives locally; no claim text is sent to an external API."""
        if self._model is None:
            # This project uses the local PyTorch sentence-transformer path only.
            # Prevent optional TensorFlow/Keras imports from breaking that path in
            # environments with an incompatible Keras installation.
            os.environ.setdefault("USE_TF", "0")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on installation state
                raise RuntimeError(
                    "sentence-transformers is required for narrative similarity. "
                    "Install the project dependencies before building features."
                ) from exc
            # The model is part of the local MVP environment.  Loading it in
            # local-files-only mode avoids network retries during live API/UI
            # analysis and makes the persisted-index workflow reproducible.
            self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
        return np.asarray(
            self._model.encode(  # type: ignore[union-attr]
                narratives,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class NarrativeSimilarityConfig:
    """Versionable settings for local narrative similarity features."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    highly_similar_threshold: float = 0.90
    repeated_template_minimum_matches: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not -1.0 <= self.highly_similar_threshold <= 1.0:
            raise ValueError("highly_similar_threshold must be between -1 and 1.")
        if self.repeated_template_minimum_matches < 1:
            raise ValueError("repeated_template_minimum_matches must be positive.")


SIMILARITY_FEATURE_COLUMNS = (
    "maximum_prior_semantic_similarity",
    "average_same_agent_prior_semantic_similarity",
    "maximum_other_agent_prior_semantic_similarity",
    "highly_similar_prior_claim_count",
    "repeated_template_narrative_indicator",
)

DEFAULT_EMBEDDING_INDEX_PATH = Path("data/processed/narrative_embedding_index.npz")


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize rows so a dot product is cosine similarity, including zero rows."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)


@dataclass(frozen=True, slots=True)
class NarrativeEmbeddingIndex:
    """Disk-persistable local embeddings for strictly earlier-claim retrieval.

    The index contains only the four fields needed for semantic retrieval.  It
    deliberately excludes labels, scenario metadata, and outcomes.  At query
    time it filters dates before computing similarity, so cached embeddings do
    not weaken the point-in-time boundary.
    """

    claim_ids: np.ndarray
    agent_ids: np.ndarray
    submitted_dates: np.ndarray
    embeddings: np.ndarray
    highly_similar_threshold: float
    repeated_template_minimum_matches: int

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("Embedding index vectors must be two-dimensional.")
        if not (len(self.claim_ids) == len(self.agent_ids) == len(self.submitted_dates) == len(self.embeddings)):
            raise ValueError("Embedding index metadata and vectors must have matching lengths.")

    @classmethod
    def from_claims(cls, claims: pd.DataFrame, analyzer: "NarrativeSimilarityAnalyzer") -> "NarrativeEmbeddingIndex":
        required = {"claim_id", "agent_id", "submitted_date", "narrative"}
        missing = required.difference(claims.columns)
        if missing:
            raise ValueError(f"Claims missing embedding-index inputs: {sorted(missing)}")
        ordered = claims.loc[:, ["claim_id", "agent_id", "submitted_date", "narrative"]].copy()
        ordered["submitted_date"] = pd.to_datetime(ordered["submitted_date"])
        ordered = ordered.sort_values(["submitted_date", "claim_id"], kind="stable").reset_index(drop=True)
        embeddings = analyzer._encode_narratives(ordered["narrative"].astype(str).tolist())
        return cls(
            claim_ids=ordered["claim_id"].astype(str).to_numpy(dtype=str),
            agent_ids=ordered["agent_id"].astype(str).to_numpy(dtype=str),
            submitted_dates=ordered["submitted_date"].to_numpy(dtype="datetime64[D]"),
            embeddings=embeddings,
            highly_similar_threshold=analyzer.config.highly_similar_threshold,
            repeated_template_minimum_matches=analyzer.config.repeated_template_minimum_matches,
        )

    def write(self, output_path: Path) -> Path:
        """Persist a deterministic local artifact without raw narrative text."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            claim_ids=self.claim_ids,
            agent_ids=self.agent_ids,
            submitted_dates=self.submitted_dates,
            embeddings=self.embeddings,
            highly_similar_threshold=np.asarray(self.highly_similar_threshold, dtype=np.float32),
            repeated_template_minimum_matches=np.asarray(self.repeated_template_minimum_matches, dtype=np.int64),
        )
        return output_path

    @classmethod
    def read(cls, input_path: Path) -> "NarrativeEmbeddingIndex":
        with np.load(input_path, allow_pickle=False) as payload:
            return cls(
                claim_ids=payload["claim_ids"].astype(str),
                agent_ids=payload["agent_ids"].astype(str),
                submitted_dates=payload["submitted_dates"].astype("datetime64[D]"),
                embeddings=_normalize_embeddings(payload["embeddings"].astype(np.float32, copy=False)),
                highly_similar_threshold=float(payload["highly_similar_threshold"]),
                repeated_template_minimum_matches=int(payload["repeated_template_minimum_matches"]),
            )

    def score_claim(self, claim: pd.Series) -> tuple[dict[str, float | int], list[str]]:
        """Score an indexed claim against only strictly earlier index entries."""
        required = {"claim_id", "agent_id", "submitted_date"}
        missing = required.difference(claim.index)
        if missing:
            raise ValueError(f"Claim missing embedding-index query inputs: {sorted(missing)}")
        claim_id = str(claim["claim_id"])
        positions = np.flatnonzero(self.claim_ids == claim_id)
        if len(positions) != 1:
            raise ValueError(f"Claim {claim_id} is not uniquely represented in the embedding index.")
        position = int(positions[0])
        decision_date = np.datetime64(pd.Timestamp(claim["submitted_date"]).date(), "D")
        # The id check is defensive; the strict date check already excludes the
        # current claim and all same-day/future claims in the normal index.
        prior_mask = (self.submitted_dates < decision_date) & (self.claim_ids != claim_id)
        if not prior_mask.any():
            return ({column: 0 for column in SIMILARITY_FEATURE_COLUMNS}, [])
        prior_positions = np.flatnonzero(prior_mask)
        similarities = np.clip(self.embeddings[prior_positions] @ self.embeddings[position], -1.0, 1.0)
        same_agent = self.agent_ids[prior_positions] == str(claim["agent_id"])
        other_agent = ~same_agent
        high_mask = similarities >= self.highly_similar_threshold
        high_positions = np.flatnonzero(high_mask)
        references = (
            self.claim_ids[prior_positions[high_positions[np.argsort(similarities[high_positions])[::-1]]]].astype(str).tolist()[:5]
            if len(high_positions)
            else []
        )
        high_count = int(len(high_positions))
        return (
            {
                "maximum_prior_semantic_similarity": float(similarities.max()),
                "average_same_agent_prior_semantic_similarity": float(similarities[same_agent].mean()) if same_agent.any() else 0.0,
                "maximum_other_agent_prior_semantic_similarity": float(similarities[other_agent].max()) if other_agent.any() else 0.0,
                "highly_similar_prior_claim_count": high_count,
                "repeated_template_narrative_indicator": int(high_count >= self.repeated_template_minimum_matches),
            },
            references,
        )


class NarrativeSimilarityAnalyzer:
    """Calculate claim signals against a strictly earlier local narrative index."""

    def __init__(
        self,
        config: NarrativeSimilarityConfig | None = None,
        embedder: NarrativeEmbedder | None = None,
    ) -> None:
        self.config = config or NarrativeSimilarityConfig()
        self._embedder = embedder or SentenceTransformerEmbedder(self.config.model_name)

    def _encode_narratives(self, narratives: list[str]) -> np.ndarray:
        """Encode and normalize a batch once for reusable local retrieval."""
        embeddings = self._embedder.encode(narratives, batch_size=self.config.batch_size)
        if embeddings.ndim != 2 or len(embeddings) != len(narratives):
            raise ValueError("Narrative embedder must return a two-dimensional row per narrative.")
        return _normalize_embeddings(embeddings.astype(np.float32, copy=False))

    def build_embedding_index(self, claims: pd.DataFrame) -> NarrativeEmbeddingIndex:
        """Generate one local embedding artifact for a deterministic dataset."""
        return NarrativeEmbeddingIndex.from_claims(claims, self)

    def build_features(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Return point-in-time similarity features without labels or scenario data.

        Claims sharing a submission date are evaluated before any one of them is
        added to the comparison index.  Therefore a claim cannot match itself,
        a same-day claim, or a future claim.
        """
        required = {"claim_id", "agent_id", "submitted_date", "narrative"}
        missing = required.difference(claims.columns)
        if missing:
            raise ValueError(f"Claims missing similarity inputs: {sorted(missing)}")
        ordered = claims.loc[:, ["claim_id", "agent_id", "submitted_date", "narrative"]].copy()
        ordered["submitted_date"] = pd.to_datetime(ordered["submitted_date"])
        ordered = ordered.sort_values(["submitted_date", "claim_id"], kind="stable").reset_index(drop=True)
        embeddings = self._encode_narratives(ordered["narrative"].astype(str).tolist())

        # Allocate once rather than repeatedly copying the full history while
        # evaluating each claim.  The logical index still contains only prior
        # claims at every scoring point.
        history_embeddings = np.empty_like(embeddings)
        history_count = 0
        history_agents: list[str] = []
        rows: list[dict[str, object]] = []
        for _, day_claims in ordered.groupby("submitted_date", sort=True):
            for row_index, claim in day_claims.iterrows():
                current_agent = str(claim["agent_id"])
                if history_count:
                    similarities = np.clip(history_embeddings[:history_count] @ embeddings[row_index], -1.0, 1.0)
                    same_agent_mask = np.fromiter((agent == current_agent for agent in history_agents), dtype=bool)
                    other_agent_mask = ~same_agent_mask
                    high_count = int(np.count_nonzero(similarities >= self.config.highly_similar_threshold))
                    maximum = float(similarities.max())
                    same_agent_average = float(similarities[same_agent_mask].mean()) if same_agent_mask.any() else 0.0
                    other_agent_maximum = float(similarities[other_agent_mask].max()) if other_agent_mask.any() else 0.0
                else:
                    high_count = 0
                    maximum = 0.0
                    same_agent_average = 0.0
                    other_agent_maximum = 0.0
                rows.append(
                    {
                        "claim_id": claim["claim_id"],
                        "maximum_prior_semantic_similarity": maximum,
                        "average_same_agent_prior_semantic_similarity": same_agent_average,
                        "maximum_other_agent_prior_semantic_similarity": other_agent_maximum,
                        "highly_similar_prior_claim_count": high_count,
                        "repeated_template_narrative_indicator": int(
                            high_count >= self.config.repeated_template_minimum_matches
                        ),
                    }
                )

            # Make this day's claims visible only after every one has been evaluated.
            for row_index, claim in day_claims.iterrows():
                history_embeddings[history_count] = embeddings[row_index]
                history_count += 1
                history_agents.append(str(claim["agent_id"]))

        return pd.DataFrame(rows).sort_values("claim_id", kind="stable").reset_index(drop=True)

    def score_against_prior_claims(self, claim: pd.Series, prior_claims: pd.DataFrame) -> tuple[dict[str, float | int], list[str]]:
        """Score one claim against explicitly supplied earlier claims only.

        This supports online inference without admitting the current claim into
        its own comparison set.  Callers remain responsible for obtaining only
        records submitted before the current claim's submission time.
        """
        required = {"claim_id", "agent_id", "narrative"}
        missing = required.difference(prior_claims.columns) | required.difference(claim.index)
        if missing:
            raise ValueError(f"Claims missing similarity inputs: {sorted(missing)}")
        if prior_claims.empty:
            return ({column: 0 for column in SIMILARITY_FEATURE_COLUMNS}, [])
        narratives = prior_claims["narrative"].astype(str).tolist() + [str(claim["narrative"])]
        embeddings = self._encode_narratives(narratives)
        similarities = np.clip(embeddings[:-1] @ embeddings[-1], -1.0, 1.0)
        same_agent = prior_claims["agent_id"].astype(str).to_numpy() == str(claim["agent_id"])
        other_agent = ~same_agent
        high_mask = similarities >= self.config.highly_similar_threshold
        high_count = int(np.count_nonzero(high_mask))
        high_positions = np.flatnonzero(high_mask)
        references = (
            prior_claims.iloc[high_positions[np.argsort(similarities[high_positions])[::-1]]]["claim_id"].astype(str).head(5).tolist()
            if high_count
            else []
        )
        return (
            {
                "maximum_prior_semantic_similarity": float(similarities.max()),
                "average_same_agent_prior_semantic_similarity": float(similarities[same_agent].mean()) if same_agent.any() else 0.0,
                "maximum_other_agent_prior_semantic_similarity": float(similarities[other_agent].max()) if other_agent.any() else 0.0,
                "highly_similar_prior_claim_count": high_count,
                "repeated_template_narrative_indicator": int(high_count >= self.config.repeated_template_minimum_matches),
            },
            references,
        )


def build_embedding_index_from_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
    analyzer: NarrativeSimilarityAnalyzer | None = None,
) -> NarrativeEmbeddingIndex:
    """Build a local retrieval artifact using only claim-time narrative fields."""
    engine = create_engine(database_path)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        claims = pd.read_sql(
            select(Claim.claim_id, Claim.agent_id, Claim.submitted_date, Claim.narrative),
            session.connection(),
        )
    return (analyzer or NarrativeSimilarityAnalyzer()).build_embedding_index(claims)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local persisted narrative embedding index.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    return parser.parse_args()


def main() -> None:
    """Create reusable local embeddings without labels or synthetic metadata."""
    args = _parse_args()
    index = build_embedding_index_from_database(args.database)
    index.write(args.output)
    print(f"Wrote {len(index.claim_ids)} narrative embeddings to {args.output}")


if __name__ == "__main__":
    main()
