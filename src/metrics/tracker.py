"""
tracker.py — Performance Timing Tracker
==========================================
Tracks timing for each stage of the RAG pipeline.

TERMINOLOGY:
    - Instrumentation: Adding timing measurements to code to understand
      where time is spent. Essential for optimization.

    - Pipeline Stages: The RAG pipeline has distinct stages, each measured:
        1. Query Analysis (adaptive layer)
        2. Cache Lookup
        3. Retrieval (vector/keyword/hybrid search)
        4. Re-ranking
        5. Prompt Building
        6. LLM Generation
        7. Response Parsing

    - P50 / P95 Latency:
        * P50 (50th percentile / median): Half of queries are faster than this
        * P95 (95th percentile): 95% of queries are faster than this
        * P99 (99th percentile): 99% of queries are faster than this
        P95 is the most commonly reported latency metric because it shows
        how the system behaves for the vast majority of users while ignoring
        extreme outliers.

    - Latency Breakdown: Showing how much time each stage contributes
      to the total. Example:
        Total: 2000ms
        ├── Retrieval: 300ms (15%)
        ├── Re-ranking: 500ms (25%)
        └── Generation: 1200ms (60%)
      This tells us generation is the bottleneck.

HOW IT WORKS:
    1. Create a timer context manager for each pipeline stage
    2. Record time elapsed in each stage
    3. Aggregate across queries for P50/P95 calculations
    4. Provide breakdown reports
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StageTimings:
    """Timing data for a single query."""

    query_analysis_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    prompt_building_ms: float = 0.0
    generation_ms: float = 0.0
    response_parsing_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query_analysis_ms": round(self.query_analysis_ms, 2),
            "cache_lookup_ms": round(self.cache_lookup_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "reranking_ms": round(self.reranking_ms, 2),
            "prompt_building_ms": round(self.prompt_building_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "response_parsing_ms": round(self.response_parsing_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


class PerformanceTracker:
    """
    Tracks timing for each RAG pipeline stage.

    Provides context managers for timing, and aggregation methods
    for computing P50/P95 statistics.

    Usage:
        tracker = PerformanceTracker()

        timings = StageTimings()
        with tracker.time("retrieval") as t:
            results = retriever.retrieve(query)
        timings.retrieval_ms = t.elapsed_ms

        tracker.record(timings)
        report = tracker.get_report()
    """

    def __init__(self):
        self.all_timings: List[StageTimings] = []

    @contextmanager
    def time(self, stage_name: str):
        """
        Context manager that measures elapsed time.

        Usage:
            with tracker.time("retrieval") as t:
                do_retrieval()
            print(f"Took {t.elapsed_ms}ms")
        """
        timer = _Timer()
        timer.start()
        try:
            yield timer
        finally:
            timer.stop()

    def record(self, timings: StageTimings) -> None:
        """Record timing data for a single query."""
        self.all_timings.append(timings)

    def get_report(self) -> dict:
        """
        Generate a performance report with P50/P95 statistics.

        Returns:
            Dictionary with per-stage and overall statistics
        """
        if not self.all_timings:
            return {"message": "No timings recorded yet"}

        stages = [
            "query_analysis_ms",
            "cache_lookup_ms",
            "retrieval_ms",
            "reranking_ms",
            "prompt_building_ms",
            "generation_ms",
            "response_parsing_ms",
            "total_ms",
        ]

        report = {"query_count": len(self.all_timings), "stages": {}}

        for stage in stages:
            values = [getattr(t, stage) for t in self.all_timings]
            values = [v for v in values if v > 0]  # Skip zeros

            if values:
                arr = np.array(values)
                report["stages"][stage] = {
                    "p50": round(float(np.percentile(arr, 50)), 2),
                    "p95": round(float(np.percentile(arr, 95)), 2),
                    "p99": round(float(np.percentile(arr, 99)), 2),
                    "mean": round(float(np.mean(arr)), 2),
                    "min": round(float(np.min(arr)), 2),
                    "max": round(float(np.max(arr)), 2),
                }

        # Add retrieval vs generation breakdown
        retrieval_times = [t.retrieval_ms + t.reranking_ms for t in self.all_timings]
        generation_times = [t.generation_ms for t in self.all_timings]
        total_times = [t.total_ms for t in self.all_timings]

        avg_total = np.mean(total_times) if total_times else 1
        report["breakdown"] = {
            "avg_retrieval_pct": round(
                np.mean(retrieval_times) / max(avg_total, 1) * 100, 1
            ),
            "avg_generation_pct": round(
                np.mean(generation_times) / max(avg_total, 1) * 100, 1
            ),
            "avg_overhead_pct": round(
                (avg_total - np.mean(retrieval_times) - np.mean(generation_times))
                / max(avg_total, 1)
                * 100,
                1,
            ),
        }

        return report

    def get_timings_data(self) -> List[dict]:
        """Get all timings as a list of dicts for visualization."""
        return [t.to_dict() for t in self.all_timings]


class _Timer:
    """Simple timer helper for the context manager."""

    def __init__(self):
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
