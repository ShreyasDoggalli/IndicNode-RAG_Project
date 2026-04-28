"""
feedback_loop.py — Feedback Tracking & Auto-adjustment
========================================================
Tracks per-query metrics and adjusts system parameters over time.

TERMINOLOGY:
    - Feedback Loop: A mechanism where the system's output is fed back as
      input to improve future behavior. In our RAG system:
        Output metrics (latency, quality) → Analysis → Parameter adjustment

    - Moving Average: A statistical method that smooths out fluctuations
      by averaging the last N values. We use this to track trends rather
      than reacting to individual outliers.
        Example: If the last 5 latencies were [100, 200, 150, 300, 250],
        the moving average is 200ms.

    - Exponential Moving Average (EMA): A weighted moving average that gives
      more weight to recent values. Formula:
        EMA = α × new_value + (1-α) × old_EMA
      Where α (smoothing factor) controls how fast the average adapts.
      α = 0.3 means "30% weight on new data, 70% on history"

    - Quality Proxy: Since we can't ask users for feedback on every query,
      we use automated proxies:
        * Response confidence score (from response_parser)
        * Answer length (very short = likely poor)
        * Source references (citing documents = likely grounded)
        * Refusal rate (too many "I don't know" = poor retrieval)

    - Adaptive Parameters: Values that change based on feedback:
        * top_k_adjustment: Offset to apply to the base top_k
        * strategy_preference: Shift toward vector or hybrid
        * rerank_benefit: Whether re-ranking is helping quality

    - No Training Required: Unlike ML model training, our feedback loop
      uses simple heuristic rules. No gradient descent, no loss functions.
      Just tracking metrics and adjusting knobs.

HOW IT WORKS:
    1. After each query, record metrics: latency, quality, decisions made
    2. Compute moving averages of recent metrics
    3. Compare against thresholds
    4. Adjust baseline parameters:
        - If latency trending up → suggest lower top_k
        - If quality trending down → suggest higher top_k, enable re-ranking
        - If re-ranking isn't helping → suggest disabling it
    5. Decision engine reads these adjustments for future queries
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config import config


@dataclass
class QueryRecord:
    """
    Record of a single query's performance.

    Stored in the feedback history for trend analysis.
    """

    timestamp: float
    query: str
    complexity: str
    top_k_used: int
    strategy_used: str
    used_reranking: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    quality_score: float
    response_word_count: int
    was_refusal: bool


@dataclass
class FeedbackState:
    """
    Current state of the feedback system's learned adjustments.

    These values influence future decisions.
    """

    top_k_offset: int = 0  # Add/subtract from suggested top_k
    prefer_hybrid: bool = False  # Shift toward hybrid retrieval
    reranking_helpful: bool = True  # Whether re-ranking improves quality
    avg_latency_ms: float = 0.0
    avg_quality: float = 0.5
    total_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "top_k_offset": self.top_k_offset,
            "prefer_hybrid": self.prefer_hybrid,
            "reranking_helpful": self.reranking_helpful,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_quality": self.avg_quality,
            "total_queries": self.total_queries,
        }


class FeedbackLoop:
    """
    Tracks query metrics and adjusts system behavior over time.

    The feedback loop learns from recent queries (using a sliding window)
    and provides parameter adjustments to the decision engine.

    Key principle: No ML training required. Just simple tracking and rules.

    Usage:
        feedback = FeedbackLoop(window_size=20)

        # After each query:
        feedback.record(query_record)

        # Before each query:
        state = feedback.get_state()
        # Use state.avg_latency_ms and state.avg_quality in decision engine
    """

    def __init__(
        self,
        window_size: int = 20,
        ema_alpha: float = 0.3,
    ):
        """
        Args:
            window_size: Number of recent queries to consider
            ema_alpha: EMA smoothing factor (0-1, higher = more reactive)
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.history: deque = deque(maxlen=window_size)
        self.state = FeedbackState()

        # Track re-ranking impact
        self._rerank_quality_sum = 0.0
        self._no_rerank_quality_sum = 0.0
        self._rerank_count = 0
        self._no_rerank_count = 0

    def record(self, record: QueryRecord) -> None:
        """
        Record a query's performance metrics.

        After recording, automatically updates the feedback state.

        Args:
            record: QueryRecord with all metrics for this query
        """
        self.history.append(record)
        self.state.total_queries += 1

        # Update EMA for latency
        self.state.avg_latency_ms = (
            self.ema_alpha * record.total_latency_ms
            + (1 - self.ema_alpha) * self.state.avg_latency_ms
        )

        # Update EMA for quality
        self.state.avg_quality = (
            self.ema_alpha * record.quality_score
            + (1 - self.ema_alpha) * self.state.avg_quality
        )

        # Track re-ranking effectiveness
        if record.used_reranking:
            self._rerank_quality_sum += record.quality_score
            self._rerank_count += 1
        else:
            self._no_rerank_quality_sum += record.quality_score
            self._no_rerank_count += 1

        # Run adjustment logic
        self._adjust_parameters()

    def _adjust_parameters(self) -> None:
        """
        Analyze recent trends and adjust feedback state.

        Called after every query record. Uses simple threshold-based rules.
        """
        if len(self.history) < 3:
            return  # Need at least 3 queries for meaningful trends

        recent = list(self.history)[-10:]  # Last 10 queries

        # ─── Top-K Adjustment ────────────────────────────────────

        recent_quality = [r.quality_score for r in recent]
        avg_recent_quality = sum(recent_quality) / len(recent_quality)

        if avg_recent_quality < config.quality_threshold:
            # Quality is low → increase retrieval depth
            self.state.top_k_offset = min(self.state.top_k_offset + 1, 5)
        elif avg_recent_quality > 0.7 and self.state.top_k_offset > 0:
            # Quality is good → can reduce back
            self.state.top_k_offset = max(self.state.top_k_offset - 1, -2)

        # ─── Strategy Adjustment ─────────────────────────────────

        # If recent refusal rate is high, prefer hybrid (broader search)
        recent_refusals = sum(1 for r in recent if r.was_refusal)
        refusal_rate = recent_refusals / len(recent)

        if refusal_rate > 0.3:
            self.state.prefer_hybrid = True
        elif refusal_rate < 0.1:
            self.state.prefer_hybrid = False

        # ─── Re-ranking Assessment ───────────────────────────────

        if self._rerank_count >= 3 and self._no_rerank_count >= 3:
            rerank_avg = self._rerank_quality_sum / self._rerank_count
            no_rerank_avg = self._no_rerank_quality_sum / self._no_rerank_count

            # Re-ranking is helpful if it improves quality by at least 5%
            self.state.reranking_helpful = rerank_avg > no_rerank_avg * 1.05

    def get_state(self) -> FeedbackState:
        """
        Get the current feedback state for the decision engine.

        Returns:
            FeedbackState with current parameter adjustments
        """
        return self.state

    def get_metrics_summary(self) -> dict:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dictionary with aggregated metrics useful for reporting.
        """
        if not self.history:
            return {"message": "No queries recorded yet"}

        records = list(self.history)

        latencies = [r.total_latency_ms for r in records]
        retrieval_latencies = [r.retrieval_latency_ms for r in records]
        generation_latencies = [r.generation_latency_ms for r in records]
        qualities = [r.quality_score for r in records]

        return {
            "total_queries": self.state.total_queries,
            "window_queries": len(records),
            "latency": {
                "avg_total_ms": sum(latencies) / len(latencies),
                "avg_retrieval_ms": sum(retrieval_latencies) / len(retrieval_latencies),
                "avg_generation_ms": sum(generation_latencies) / len(generation_latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            },
            "quality": {
                "avg_score": sum(qualities) / len(qualities),
                "min_score": min(qualities),
                "max_score": max(qualities),
            },
            "refusal_rate": sum(1 for r in records if r.was_refusal) / len(records),
            "adjustments": self.state.to_dict(),
        }

    def save_history(self, filepath: str = None) -> None:
        """Save query history to a JSON file."""
        path = Path(filepath or config.metrics_output_dir / "feedback_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for r in self.history:
            records.append({
                "timestamp": r.timestamp,
                "query": r.query,
                "complexity": r.complexity,
                "top_k_used": r.top_k_used,
                "strategy_used": r.strategy_used,
                "used_reranking": r.used_reranking,
                "retrieval_latency_ms": r.retrieval_latency_ms,
                "generation_latency_ms": r.generation_latency_ms,
                "total_latency_ms": r.total_latency_ms,
                "quality_score": r.quality_score,
                "response_word_count": r.response_word_count,
                "was_refusal": r.was_refusal,
            })

        with open(path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"💾 Saved feedback history ({len(records)} records) to {path}")
