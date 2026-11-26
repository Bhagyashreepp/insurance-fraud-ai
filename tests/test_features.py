"""Tests for point-in-time, target-safe claim feature construction."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest

from src.features.config import FeatureConfig
from src.features.engine import build_claim_features, write_features
from src.models.narrative_similarity import (
    DEFAULT_EMBEDDING_INDEX_PATH,
    NarrativeEmbeddingIndex,
    NarrativeSimilarityAnalyzer,
    NarrativeSimilarityConfig,
)


class _DeterministicEmbedder:
    """Small local stand-in so feature tests never download a model."""

    def encode(self, narratives: list[str], *, batch_size: int) -> np.ndarray:
        vectors = {
            "Water leaked in the kitchen.": [1.0, 0.0, 0.0],
            "Smoke damaged the garage.": [0.0, 1.0, 0.0],
            "Water damage in the kitchen caused by a burst pipe.": [0.99, 0.1, 0.0],
            "A tree fell on the detached garage during high winds.": [0.0, 0.0, 1.0],
        }
        return np.asarray([vectors.get(narrative, [0.0, 0.0, 1.0]) for narrative in narratives], dtype=np.float32)


class _CountingEmbedder(_DeterministicEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, narratives: list[str], *, batch_size: int) -> np.ndarray:
        self.calls += 1
        return super().encode(narratives, batch_size=batch_size)


def _similarity_analyzer() -> NarrativeSimilarityAnalyzer:
    return NarrativeSimilarityAnalyzer(
        NarrativeSimilarityConfig(highly_similar_threshold=0.95),
        embedder=_DeterministicEmbedder(),
    )


def _source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.DataFrame(
        [
            {"claim_id": "C1", "policy_id": "P1", "customer_id": "U1", "agent_id": "A1", "loss_type": "water", "incident_date": "2024-01-08", "submitted_date": "2024-01-10", "claim_amount": 100.0, "narrative": "Water leaked in the kitchen."},
            {"claim_id": "C2", "policy_id": "P1", "customer_id": "U1", "agent_id": "A1", "loss_type": "water", "incident_date": "2024-01-09", "submitted_date": "2024-01-11", "claim_amount": 300.0, "narrative": "Water leaked in the kitchen."},
            {"claim_id": "C3", "policy_id": "P1", "customer_id": "U1", "agent_id": "A1", "loss_type": "fire", "incident_date": "2024-01-09", "submitted_date": "2024-01-11", "claim_amount": 500.0, "narrative": "Smoke damaged the garage."},
        ]
    )
    policies = pd.DataFrame(
        [{"policy_id": "P1", "policy_status": "active", "effective_date": "2023-01-01", "expiration_date": "2025-01-01"}]
    )
    documents = pd.DataFrame(
        [
            {"claim_id": "C1", "document_date": "2024-01-08", "uploaded_date": "2024-01-10"},
            {"claim_id": "C2", "document_date": "2024-01-14", "uploaded_date": "2024-01-11"},
            {"claim_id": "C3", "document_date": "2024-01-09", "uploaded_date": "2024-01-12"},
        ]
    )
    return claims, policies, documents


def test_historical_features_use_only_strictly_prior_claims() -> None:
    claims, policies, documents = _source_tables()
    features = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer())
    first = features.set_index("claim_id").loc["C1"]
    second = features.set_index("claim_id").loc["C2"]
    same_day = features.set_index("claim_id").loc["C3"]
    assert first.agent_historical_claim_count == 0
    assert second.agent_historical_claim_count == 1
    assert second.agent_historical_average_claim_amount == 100.0
    assert second.duplicate_similarity == 1.0
    assert same_day.agent_historical_claim_count == 1  # C2 cannot be visible to C3.
    assert same_day.claim_frequency == 1
    assert same_day.document_count == 0  # Its only document uploads after decision time.
    assert same_day.missing_document_indicator == 1


def test_target_label_never_influences_or_appears_in_features() -> None:
    claims, policies, documents = _source_tables()
    claims["fraud_label"] = [0, 1, 1]
    claims["injected_patterns"] = ["", "amount_inflation", "multi_signal"]
    before = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer())
    claims["fraud_label"] = [1, 0, 0]
    claims["injected_patterns"] = ["duplicate_claim", "", ""]
    after = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer())
    assert "fraud_label" not in before.columns
    assert "injected_patterns" not in before.columns
    pdt.assert_frame_equal(before, after)


def test_policy_document_and_text_features() -> None:
    claims, policies, documents = _source_tables()
    features = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer())
    second = features.set_index("claim_id").loc["C2"]
    assert second.policy_active_indicator == 1
    assert second.document_count == 1
    assert second.document_incident_date_difference_days == 5.0
    assert second.description_length > 0
    assert second.word_count == 5
    assert second.sentence_count == 1


def test_features_write_to_parquet(tmp_path: Path) -> None:
    claims, policies, documents = _source_tables()
    output = write_features(
        build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer()), tmp_path / "features.parquet"
    )
    assert output.is_file()
    assert len(pd.read_parquet(output)) == len(claims)


def test_near_duplicate_narratives_create_semantic_reuse_signals() -> None:
    claims, policies, documents = _source_tables()
    claims.loc[1, "narrative"] = "Water damage in the kitchen caused by a burst pipe."
    features = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer()).set_index("claim_id")
    second = features.loc["C2"]
    assert second.maximum_prior_semantic_similarity > 0.95
    assert second.average_same_agent_prior_semantic_similarity > 0.95
    assert second.highly_similar_prior_claim_count == 1
    assert second.repeated_template_narrative_indicator == 1


def test_unrelated_narratives_do_not_create_reuse_signals() -> None:
    claims, policies, documents = _source_tables()
    claims.loc[1, "narrative"] = "A tree fell on the detached garage during high winds."
    features = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer()).set_index("claim_id")
    second = features.loc["C2"]
    assert second.maximum_prior_semantic_similarity == 0.0
    assert second.highly_similar_prior_claim_count == 0
    assert second.repeated_template_narrative_indicator == 0


def test_similarity_excludes_self_matches_and_future_claims() -> None:
    claims, policies, documents = _source_tables()
    claims.loc[2, "submitted_date"] = "2024-01-12"
    claims.loc[2, "narrative"] = "Water leaked in the kitchen."
    features = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer()).set_index("claim_id")
    assert features.loc["C1", "maximum_prior_semantic_similarity"] == 0.0  # Cannot match itself or future C3.
    assert features.loc["C1", "highly_similar_prior_claim_count"] == 0
    assert features.loc["C3", "maximum_prior_semantic_similarity"] == 1.0  # Earlier C1/C2 are visible.


def test_similarity_ignores_label_and_injection_metadata() -> None:
    claims, policies, documents = _source_tables()
    claims["fraud_label"] = [0, 1, 1]
    claims["injected_patterns"] = ["", "reused_narrative", "multi_signal"]
    before = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer())
    claims["fraud_label"] = [1, 0, 0]
    claims["injected_patterns"] = ["duplicate_claim", "", ""]
    after = build_claim_features(claims, policies, documents, similarity_analyzer=_similarity_analyzer())
    similarity_columns = [column for column in before.columns if "similar" in column or "template" in column]
    pdt.assert_frame_equal(before.loc[:, similarity_columns], after.loc[:, similarity_columns])


def test_similarity_references_include_only_highly_similar_prior_claims() -> None:
    claims, _, _ = _source_tables()
    analyzer = _similarity_analyzer()
    signals, references = analyzer.score_against_prior_claims(claims.iloc[1], claims.iloc[[0, 2]])
    assert signals["highly_similar_prior_claim_count"] == 1
    assert references == ["C1"]


def test_large_artifact_mode_omits_offline_similarity_without_affecting_other_features() -> None:
    claims, policies, documents = _source_tables()
    features = build_claim_features(claims, policies, documents, include_similarity=False)
    assert "maximum_prior_semantic_similarity" not in features
    assert "claim_amount" in features
    assert "fraud_label" not in features


def test_persisted_embedding_index_avoids_per_request_reembedding(tmp_path: Path) -> None:
    claims, _, _ = _source_tables()
    embedder = _CountingEmbedder()
    analyzer = NarrativeSimilarityAnalyzer(NarrativeSimilarityConfig(highly_similar_threshold=0.95), embedder=embedder)
    index = analyzer.build_embedding_index(claims)
    assert embedder.calls == 1
    path = index.write(tmp_path / "narrative_index.npz")
    persisted = NarrativeEmbeddingIndex.read(path)
    first = persisted.score_claim(claims.iloc[1])
    second = persisted.score_claim(claims.iloc[1])
    assert embedder.calls == 1
    assert first == second


def test_embedding_index_excludes_self_future_and_same_day_claims() -> None:
    claims, _, _ = _source_tables()
    claims.loc[2, "narrative"] = "Water leaked in the kitchen."
    analyzer = _similarity_analyzer()
    index = analyzer.build_embedding_index(claims)
    c1, c2, c3 = (claims.iloc[position] for position in range(3))
    assert index.score_claim(c1)[0]["maximum_prior_semantic_similarity"] == 0.0
    # C3 is same-day with C2 and cannot use C2, while C1 remains earlier.
    c3_signals, c3_references = index.score_claim(c3)
    assert c3_signals["highly_similar_prior_claim_count"] == 1
    assert c3_references == ["C1"]
    c2_signals, c2_references = index.score_claim(c2)
    assert c2_signals["highly_similar_prior_claim_count"] == 1
    assert c2_references == ["C1"]


def test_embedding_index_scores_a_50000_claim_history_without_reembedding() -> None:
    size = 50_000
    claim_ids = np.asarray([f"C{position:05d}" for position in range(size)], dtype=str)
    index = NarrativeEmbeddingIndex(
        claim_ids=claim_ids,
        agent_ids=np.full(size, "A1", dtype=str),
        submitted_dates=np.arange(size, dtype="timedelta64[D]") + np.datetime64("2020-01-01"),
        embeddings=np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (size, 1)),
        highly_similar_threshold=0.95,
        repeated_template_minimum_matches=1,
    )
    claim = pd.Series({"claim_id": claim_ids[-1], "agent_id": "A1", "submitted_date": str(pd.Timestamp(index.submitted_dates[-1]).date())})
    started = perf_counter()
    signals, references = index.score_claim(claim)
    elapsed = perf_counter() - started
    assert elapsed < 2.0
    assert signals["highly_similar_prior_claim_count"] == size - 1
    assert len(references) == 5


@pytest.mark.skipif(not DEFAULT_EMBEDDING_INDEX_PATH.is_file(), reason="The optional local 50,000-claim artifact is not present.")
def test_persisted_50000_claim_artifact_scores_representative_claim_without_timeout() -> None:
    index = NarrativeEmbeddingIndex.read(DEFAULT_EMBEDDING_INDEX_PATH)
    position = int(np.flatnonzero(index.claim_ids == "CLM-018567")[0])
    claim = pd.Series(
        {
            "claim_id": "CLM-018567",
            "agent_id": index.agent_ids[position],
            "submitted_date": str(pd.Timestamp(index.submitted_dates[position]).date()),
        }
    )
    started = perf_counter()
    signals, references = index.score_claim(claim)
    elapsed = perf_counter() - started
    assert elapsed < 5.0
    assert signals["maximum_prior_semantic_similarity"] >= index.highly_similar_threshold
    assert references
