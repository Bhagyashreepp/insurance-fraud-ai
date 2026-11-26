"""Evidence-bound deterministic and optional OpenAI explanation rendering."""

from .service import ClaimExplanation, ExplanationConfig, ExplanationEvidence, generate_explanation

__all__ = ["ClaimExplanation", "ExplanationConfig", "ExplanationEvidence", "generate_explanation"]
