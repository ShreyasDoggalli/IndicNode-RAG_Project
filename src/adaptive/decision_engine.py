"""
decision_engine.py — Runtime Parameter Selection
===================================================
Makes runtime decisions about retrieval parameters based on query
analysis and system state.

TERMINOLOGY:
    - Decision Engine: The component that decides HOW to process each query.
      It's the "brain" of the adaptive system, translating query analysis
      and feedback into concrete parameter choices.

    - Runtime Optimization: Adjusting system behavior during execution
      (at "inference time") rather than requiring offline tuning or training.
      This is what makes our RAG system "adaptive."

    - Decision Parameters:
        * top_k: How many documents to retrieve
        * retrieval_strategy: "vector", "keyword", or "hybrid"
        * use_reranking: Whether to apply cross-encoder re-ranking
        * alpha: Weight for vector vs keyword in hybrid search
        * model_size: "small" or "large" (for model routing)

    - Model Routing (Bonus): Sending simple queries to a smaller/faster model
      and complex queries to a larger/more capable model. This optimizes
      both latency and cost.

    - Latency Budget: The maximum acceptable response time. If the system
      is running slow, the decision engine can reduce retrieval depth
      (lower top_k, skip re-ranking) to stay within budget.

HOW IT WORKS:
    1. Receive query analysis from QueryAnalyzer
    2. Check current system state (recent latency, quality metrics)
    3. Apply decision rules:
        - Short query → small K, vector-only
        - Complex query → large K, hybrid + re-ranking
        - High latency → reduce K, skip re-ranking
    4. Return a Decision object with all parameters
    5. The pipeline uses these parameters for the current query
"""

from dataclasses import dataclass
from typing import Optional

from src.adaptive.query_analyzer import QueryAnalysis, QueryComplexity
from src.config import config


@dataclass
class Decision:
    """
    Runtime parameters decided for a specific query.

    These parameters control how the RAG pipeline processes the current query.
    They are adjusted per-query based on query analysis and system feedback.
    """

    top_k: int
    retrieval_strategy: str  # "vector", "keyword", "hybrid"
    use_reranking: bool
    hybrid_alpha: float
    model_size: str  # "small" or "large" (for model routing)
    temperature: float
    max_context_chars: int

    def __repr__(self) -> str:
        return (
            f"Decision(top_k={self.top_k}, strategy={self.retrieval_strategy}, "
            f"rerank={self.use_reranking}, alpha={self.hybrid_alpha:.2f}, "
            f"model={self.model_size})"
        )


class DecisionEngine:
    """
    Makes runtime decisions about retrieval parameters.

    Combines query analysis with system feedback to optimize
    each query's processing pipeline.

    The engine follows these principles:
        1. Simple queries → fast, lightweight processing
        2. Complex queries → thorough, heavyweight processing
        3. High latency → reduce processing to meet latency budget
        4. Low quality → increase retrieval depth

    Usage:
        engine = DecisionEngine()
        decision = engine.decide(query_analysis, recent_latency_ms=1500)
    """

    def __init__(self):
        self.latency_threshold = config.latency_threshold_ms
        self.min_top_k = config.min_top_k
        self.max_top_k = config.max_top_k

    def decide(
        self,
        analysis: QueryAnalysis,
        recent_avg_latency_ms: Optional[float] = None,
        recent_avg_quality: Optional[float] = None,
    ) -> Decision:
        """
        Make a decision for the current query.

        Args:
            analysis: Query analysis from QueryAnalyzer
            recent_avg_latency_ms: Moving average of recent latencies
            recent_avg_quality: Moving average of recent quality scores

        Returns:
            Decision object with all retrieval parameters
        """
        # Start with suggestions from query analysis
        top_k = analysis.suggested_top_k
        strategy = analysis.suggested_strategy
        use_reranking = analysis.complexity != QueryComplexity.SIMPLE
        alpha = config.hybrid_alpha
        model_size = "small"
        temperature = 0.1
        max_context_chars = 4000

        # ─── Complexity-based adjustments ────────────────────────

        if analysis.complexity == QueryComplexity.SIMPLE:
            # Simple queries: minimal processing for fast response
            top_k = max(self.min_top_k, 3)
            strategy = "vector"
            use_reranking = False
            model_size = "small"
            temperature = 0.1
            max_context_chars = 2000

        elif analysis.complexity == QueryComplexity.MEDIUM:
            # Medium queries: balanced approach
            top_k = 5
            strategy = "hybrid"
            use_reranking = True
            model_size = "small"
            temperature = 0.2
            max_context_chars = 3000

        elif analysis.complexity == QueryComplexity.COMPLEX:
            # Complex queries: thorough processing
            top_k = min(self.max_top_k, 10)
            strategy = "hybrid"
            use_reranking = True
            model_size = "large"
            temperature = 0.3
            max_context_chars = 5000

        # ─── Comparison adjustments ──────────────────────────────

        if analysis.has_comparison:
            # Comparisons need more diverse results
            top_k = min(top_k + 3, self.max_top_k)
            alpha = 0.5  # More balanced between vector and keyword

        # ─── Latency-based adjustments ───────────────────────────

        if recent_avg_latency_ms is not None:
            if recent_avg_latency_ms > self.latency_threshold:
                # System is slow → reduce processing
                top_k = max(self.min_top_k, top_k - 2)
                use_reranking = False
                model_size = "small"
                max_context_chars = min(max_context_chars, 2000)
                print(
                    f"⚡ Latency high ({recent_avg_latency_ms:.0f}ms) "
                    f"→ reduced top_k to {top_k}, disabled re-ranking"
                )

            elif recent_avg_latency_ms < self.latency_threshold * 0.5:
                # System is fast → we have budget for more processing
                if analysis.complexity != QueryComplexity.SIMPLE:
                    top_k = min(top_k + 2, self.max_top_k)
                    use_reranking = True

        # ─── Quality-based adjustments ───────────────────────────

        if recent_avg_quality is not None:
            if recent_avg_quality < config.quality_threshold:
                # Quality is low → try harder
                top_k = min(top_k + 3, self.max_top_k)
                use_reranking = True
                max_context_chars = min(max_context_chars + 1000, 6000)
                print(
                    f"📈 Quality low ({recent_avg_quality:.2f}) "
                    f"→ increased top_k to {top_k}"
                )

        # ─── Build decision ──────────────────────────────────────

        decision = Decision(
            top_k=top_k,
            retrieval_strategy=strategy,
            use_reranking=use_reranking,
            hybrid_alpha=alpha,
            model_size=model_size,
            temperature=temperature,
            max_context_chars=max_context_chars,
        )

        return decision
