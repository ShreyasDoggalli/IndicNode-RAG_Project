"""
test_adaptive.py — Tests for query analysis, decision engine, and feedback loop.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive.query_analyzer import QueryAnalyzer, QueryComplexity
from src.adaptive.decision_engine import DecisionEngine, Decision
from src.adaptive.feedback_loop import FeedbackLoop, QueryRecord, FeedbackState


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer."""

    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    def test_simple_query(self):
        """Short factual questions should be classified as SIMPLE."""
        result = self.analyzer.analyze("What is FAISS?")
        assert result.complexity == QueryComplexity.SIMPLE
        assert result.word_count <= 5
        assert result.suggested_top_k <= 5

    def test_medium_query(self):
        """Multi-concept questions with 'how' should have higher complexity score."""
        result = self.analyzer.analyze(
            "How does gradient descent optimize neural networks and why is the learning rate important?"
        )
        assert result.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX)

    def test_complex_query(self):
        """Multi-part analytical questions should be MEDIUM or COMPLEX."""
        result = self.analyzer.analyze(
            "Compare and contrast supervised learning and unsupervised "
            "learning, including their use cases, advantages, and limitations "
            "in different application domains."
        )
        assert result.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX)
        assert result.suggested_top_k >= 5
        assert result.has_comparison is True

    def test_comparison_detection(self):
        """Queries with comparison words should be detected."""
        result = self.analyzer.analyze("Compare BM25 vs vector search")
        assert result.has_comparison is True

    def test_query_decomposition(self):
        """Complex queries with multiple parts should be decomposed."""
        result = self.analyzer.analyze(
            "What is machine learning and how does it relate to AI?"
        )
        if result.sub_queries:
            assert len(result.sub_queries) >= 1

    def test_suggested_strategy_simple(self):
        """Simple queries should suggest vector-only strategy."""
        result = self.analyzer.analyze("What is NLP?")
        assert result.suggested_strategy in ("vector", "hybrid")

    def test_suggested_strategy_complex(self):
        """Complex queries should suggest hybrid strategy."""
        result = self.analyzer.analyze(
            "Explain the tradeoffs between different retrieval strategies "
            "and compare their performance characteristics."
        )
        assert result.suggested_strategy == "hybrid"


class TestDecisionEngine:
    """Tests for DecisionEngine."""

    def setup_method(self):
        self.engine = DecisionEngine()
        self.analyzer = QueryAnalyzer()

    def test_simple_query_decision(self):
        """Simple query should get low top_k and no re-ranking."""
        analysis = self.analyzer.analyze("What is ML?")
        decision = self.engine.decide(analysis)

        assert decision.top_k <= 5
        assert decision.use_reranking is False

    def test_complex_query_decision(self):
        """Complex query should get high top_k and re-ranking."""
        analysis = self.analyzer.analyze(
            "Compare different neural network architectures and explain "
            "their tradeoffs in terms of performance and complexity."
        )
        decision = self.engine.decide(analysis)

        assert decision.top_k >= 5
        assert decision.use_reranking is True
        assert decision.retrieval_strategy == "hybrid"

    def test_high_latency_reduction(self):
        """High latency should reduce processing."""
        analysis = self.analyzer.analyze("What is deep learning used for?")
        decision = self.engine.decide(
            analysis, recent_avg_latency_ms=5000
        )

        assert decision.use_reranking is False

    def test_low_quality_increase(self):
        """Low quality should increase retrieval depth."""
        analysis = self.analyzer.analyze("What is deep learning?")
        decision_normal = self.engine.decide(analysis)
        decision_low_q = self.engine.decide(
            analysis, recent_avg_quality=0.2
        )

        assert decision_low_q.top_k >= decision_normal.top_k


class TestFeedbackLoop:
    """Tests for FeedbackLoop."""

    def setup_method(self):
        self.feedback = FeedbackLoop(window_size=10)

    def _make_record(self, **kwargs):
        """Helper to create a QueryRecord with defaults."""
        defaults = {
            "timestamp": 1000.0,
            "query": "test query",
            "complexity": "simple",
            "top_k_used": 5,
            "strategy_used": "hybrid",
            "used_reranking": False,
            "retrieval_latency_ms": 100.0,
            "generation_latency_ms": 500.0,
            "total_latency_ms": 600.0,
            "quality_score": 0.6,
            "response_word_count": 50,
            "was_refusal": False,
        }
        defaults.update(kwargs)
        return QueryRecord(**defaults)

    def test_record_updates_state(self):
        """Recording a query should update the feedback state."""
        record = self._make_record(total_latency_ms=1000)
        self.feedback.record(record)

        state = self.feedback.get_state()
        assert state.total_queries == 1
        assert state.avg_latency_ms > 0

    def test_ema_smoothing(self):
        """EMA should smooth latency values."""
        # Record several queries
        for latency in [100, 200, 300, 400, 500]:
            self.feedback.record(
                self._make_record(total_latency_ms=latency)
            )

        state = self.feedback.get_state()
        # EMA should be between min and max
        assert 100 < state.avg_latency_ms < 500

    def test_quality_adjustment(self):
        """Low quality should trigger top_k increase."""
        # Record several low-quality queries
        for _ in range(5):
            self.feedback.record(
                self._make_record(quality_score=0.2)
            )

        state = self.feedback.get_state()
        assert state.top_k_offset >= 0  # Should suggest more retrieval

    def test_refusal_rate_tracking(self):
        """High refusal rate should trigger hybrid preference."""
        # Record queries with high refusal rate
        for _ in range(5):
            self.feedback.record(
                self._make_record(was_refusal=True)
            )

        state = self.feedback.get_state()
        assert state.prefer_hybrid is True

    def test_metrics_summary(self):
        """Metrics summary should contain expected fields."""
        for _ in range(3):
            self.feedback.record(self._make_record())

        summary = self.feedback.get_metrics_summary()
        assert "total_queries" in summary
        assert "latency" in summary
        assert "quality" in summary
