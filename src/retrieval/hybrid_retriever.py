"""
hybrid_retriever.py — Hybrid Retrieval (Vector + Keyword)
===========================================================
Combines vector search and keyword search for better retrieval.

TERMINOLOGY:
    - Hybrid Retrieval: Combining two or more retrieval strategies to
      leverage their complementary strengths:
        * Vector search → catches semantic similarity
        * Keyword search → catches exact term matches
      Together, they cover more ground than either alone.

    - Score Fusion: The method of combining scores from different retrievers.
      Common approaches:
        * Weighted Sum: α × vector_score + (1-α) × bm25_score
        * Reciprocal Rank Fusion (RRF): Combines rankings rather than scores
        * Max: Take the higher score from either method

    - Alpha (α): The weight given to vector search vs keyword search.
        * α = 1.0 → pure vector search
        * α = 0.0 → pure keyword search
        * α = 0.7 → 70% vector, 30% keyword (our default)

    - Score Normalization: Scaling scores to [0, 1] range so they're
      comparable. BM25 scores can range from 0 to infinity, while cosine
      similarity ranges from -1 to 1. We need to normalize before combining.

WHY HYBRID?
    Research consistently shows hybrid retrieval outperforms either method alone:
    - Vector search alone misses exact keyword matches
    - Keyword search alone misses semantic relationships
    - Hybrid captures both → higher recall and precision

HOW IT WORKS:
    1. Run vector search → get top-K₁ results with similarity scores
    2. Run keyword search → get top-K₂ results with BM25 scores
    3. Normalize both score sets to [0, 1]
    4. Combine using weighted sum: α × vector + (1-α) × keyword
    5. Deduplicate (same doc from both) → take max combined score
    6. Sort by combined score → return top-K
"""

from typing import Dict, List, Tuple

from src.config import config
from src.ingestion.loader import Document
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    """
    Combines vector and keyword retrieval for superior results.

    Uses weighted score fusion to merge results from both methods.
    The alpha parameter controls the balance between semantic and
    keyword matching.

    Usage:
        hybrid = HybridRetriever(vector_retriever, keyword_retriever, alpha=0.7)
        results = hybrid.retrieve("What is transfer learning?", top_k=5)
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        alpha: float = None,
    ):
        """
        Args:
            vector_retriever: FAISS-based vector retriever
            keyword_retriever: BM25-based keyword retriever
            alpha: Weight for vector scores (0.0 to 1.0).
                   Higher = more weight on semantic similarity.
                   Defaults to config.hybrid_alpha (0.7).
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha if alpha is not None else config.hybrid_alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid retrieval combining vector and keyword search.

        Args:
            query: The user's search query
            top_k: Number of final results to return
            alpha: Optional override for the vector weight

        Returns:
            List of (Document, combined_score) tuples, sorted by relevance
        """
        alpha = alpha if alpha is not None else self.alpha

        # Fetch more candidates from each retriever than needed
        # This gives us a larger pool to merge and re-score
        fetch_k = top_k * 2

        # Step 1: Get vector search results
        vector_results = self.vector_retriever.retrieve(query, top_k=fetch_k)

        # Step 2: Get keyword search results
        keyword_results = self.keyword_retriever.retrieve(query, top_k=fetch_k)

        # Step 3: Normalize scores
        vector_results = self._normalize_scores(vector_results)
        keyword_results = self._normalize_scores(keyword_results)

        # Step 4: Merge results with weighted fusion
        merged = self._merge_results(vector_results, keyword_results, alpha)

        # Step 5: Sort by combined score and return top-K
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def _normalize_scores(
        self,
        results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        This is essential because vector scores (cosine similarity: -1 to 1)
        and BM25 scores (0 to ∞) are on completely different scales.
        Without normalization, one would dominate the other.
        """
        if not results:
            return results

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same, normalize to 1.0
            return [(doc, 1.0) for doc, _ in results]

        return [
            (doc, (score - min_score) / (max_score - min_score))
            for doc, score in results
        ]

    def _merge_results(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        alpha: float,
    ) -> List[Tuple[Document, float]]:
        """
        Merge results from both retrievers using weighted score fusion.

        If a document appears in both result sets, its scores are combined:
            combined = α × vector_score + (1-α) × keyword_score

        If a document appears in only one set, the missing score is 0.

        Args:
            vector_results: Normalized vector search results
            keyword_results: Normalized keyword search results
            alpha: Weight for vector scores

        Returns:
            Merged list of (Document, combined_score) tuples
        """
        # Build a map: document_text → (document, vector_score, keyword_score)
        doc_scores: Dict[str, dict] = {}

        for doc, score in vector_results:
            key = doc.text[:200]  # Use text prefix as key
            if key not in doc_scores:
                doc_scores[key] = {
                    "doc": doc,
                    "vector": 0.0,
                    "keyword": 0.0,
                }
            doc_scores[key]["vector"] = max(doc_scores[key]["vector"], score)

        for doc, score in keyword_results:
            key = doc.text[:200]
            if key not in doc_scores:
                doc_scores[key] = {
                    "doc": doc,
                    "vector": 0.0,
                    "keyword": 0.0,
                }
            doc_scores[key]["keyword"] = max(doc_scores[key]["keyword"], score)

        # Compute combined scores
        merged = []
        for data in doc_scores.values():
            combined_score = (
                alpha * data["vector"] + (1 - alpha) * data["keyword"]
            )
            merged.append((data["doc"], combined_score))

        return merged
