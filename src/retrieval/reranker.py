"""
reranker.py — Cross-Encoder Re-ranking
=========================================
Re-ranks retrieved documents using a cross-encoder model for better precision.

TERMINOLOGY:
    - Re-ranking: A second-stage ranking step that takes the initial retrieval
      results and re-orders them using a more powerful (but slower) model.
      Think of it as a "quality filter" applied to the retrieved candidates.

    - Bi-Encoder vs Cross-Encoder:
        * Bi-Encoder (what we use for initial retrieval):
          - Encodes query and documents SEPARATELY
          - Fast: can pre-compute all document embeddings
          - Used for initial retrieval from large corpus
          - Less accurate at fine-grained relevance

        * Cross-Encoder (what we use for re-ranking):
          - Encodes query and document TOGETHER as a pair
          - Slow: must compute for each (query, document) pair
          - Used for re-ranking a small set of candidates
          - Much more accurate at judging relevance

    - Two-Stage Retrieval:
        Stage 1 (Fast): Bi-encoder retrieves 20-50 candidates from millions
        Stage 2 (Precise): Cross-encoder re-ranks the 20-50 candidates
        This gives us both speed AND accuracy.

    - Why Not Just Use Cross-Encoder?
      Cross-encoder is O(n) for n documents — scoring each requires a
      full model forward pass. For 100K documents, this would take minutes.
      Bi-encoder is O(1) after pre-computing embeddings (just a dot product).

HOW IT WORKS:
    1. Take the top-N retrieved candidates (e.g., top 20)
    2. For each candidate, form a (query, document) pair
    3. Pass each pair through the cross-encoder model
    4. Get a relevance score for each pair
    5. Re-sort by cross-encoder scores
    6. Return top-K (e.g., top 5) highest scoring
"""

from typing import List, Optional, Tuple

from src.config import config
from src.ingestion.loader import Document


class Reranker:
    """
    Re-ranks retrieved documents using a cross-encoder model.

    Improves precision by using a more powerful model to score
    (query, document) pairs directly.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the cross-encoder re-ranker.

        Args:
            model_name: Cross-encoder model to use.
                        Defaults to config.reranker_model
        """
        self.model_name = model_name or config.reranker_model
        self.model = None  # Lazy loading

    def _load_model(self):
        """Lazy-load the cross-encoder model (only when first needed)."""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            print(f"🔧 Loading re-ranker model: {self.model_name}...")
            self.model = CrossEncoder(self.model_name)
            print("✅ Re-ranker model loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank candidate documents using the cross-encoder.

        Args:
            query: The user's search query
            candidates: List of (Document, initial_score) from retrieval
            top_k: Number of results to return after re-ranking

        Returns:
            Re-ranked list of (Document, cross_encoder_score) tuples

        Note:
            Cross-encoder scores are on a different scale than similarity scores.
            They represent logits (can be any real number), where higher = more relevant.
        """
        if not candidates:
            return []

        self._load_model()

        # Form (query, document) pairs for cross-encoder
        pairs = [(query, doc.text) for doc, _ in candidates]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Combine with documents
        scored_results = [
            (doc, float(score))
            for (doc, _), score in zip(candidates, scores)
        ]

        # Sort by cross-encoder score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:top_k]
