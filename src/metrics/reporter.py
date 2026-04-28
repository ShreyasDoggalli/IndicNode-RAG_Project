"""
reporter.py — Performance Report Generator
=============================================
Generates formatted performance reports from tracked metrics.

TERMINOLOGY:
    - P50/P95 Latency: Percentile-based latency measurements.
        * P50 = median (50% of queries faster than this)
        * P95 = 95th percentile (95% of queries faster)
      These are more informative than averages because averages
      can be skewed by a few very slow queries.

    - Latency Distribution: How query response times are spread out.
      A good system has a tight distribution (low variance).
      A problematic system has long "tails" (some queries are very slow).

    - Adaptive Impact Report: Comparing performance with vs without
      the adaptive layer. This is key for the assignment — showing
      that the adaptive logic actually improves the system.

HOW IT WORKS:
    1. Read timing data from PerformanceTracker
    2. Read feedback data from FeedbackLoop  
    3. Compute aggregated statistics (P50, P95, means)
    4. Compare adaptive vs static strategies
    5. Generate a formatted text/JSON report
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import config
from src.metrics.tracker import PerformanceTracker


class PerformanceReporter:
    """
    Generates and saves performance reports.

    Creates both human-readable text reports and machine-readable
    JSON reports with P50/P95 latencies and adaptive impact analysis.

    Usage:
        reporter = PerformanceReporter(tracker)
        report = reporter.generate_report()
        reporter.save_report(report)
    """

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker

    def generate_report(self) -> dict:
        """
        Generate a comprehensive performance report.

        Returns:
            Dictionary containing all performance metrics
        """
        base_report = self.tracker.get_report()
        timings = self.tracker.get_timings_data()

        if not timings:
            return {"error": "No data to report"}

        # Add total stats
        total_latencies = [t["total_ms"] for t in timings]
        retrieval_latencies = [t["retrieval_ms"] for t in timings]
        generation_latencies = [t["generation_ms"] for t in timings]

        report = {
            "summary": {
                "total_queries": len(timings),
                "total_latency": {
                    "p50_ms": round(float(np.percentile(total_latencies, 50)), 2),
                    "p95_ms": round(float(np.percentile(total_latencies, 95)), 2),
                    "mean_ms": round(float(np.mean(total_latencies)), 2),
                },
                "retrieval_latency": {
                    "p50_ms": round(float(np.percentile(retrieval_latencies, 50)), 2),
                    "p95_ms": round(float(np.percentile(retrieval_latencies, 95)), 2),
                    "mean_ms": round(float(np.mean(retrieval_latencies)), 2),
                },
                "generation_latency": {
                    "p50_ms": round(float(np.percentile(generation_latencies, 50)), 2),
                    "p95_ms": round(float(np.percentile(generation_latencies, 95)), 2),
                    "mean_ms": round(float(np.mean(generation_latencies)), 2),
                },
            },
            "detailed": base_report,
            "per_query": timings,
        }

        return report

    def print_report(self, report: dict = None) -> str:
        """
        Generate a human-readable text report.

        Returns:
            Formatted string report
        """
        if report is None:
            report = self.generate_report()

        if "error" in report:
            return "No performance data available."

        summary = report["summary"]
        lines = [
            "",
            "=" * 60,
            "  PERFORMANCE REPORT — Adaptive RAG System",
            "=" * 60,
            "",
            f"  Total Queries: {summary['total_queries']}",
            "",
            "  ┌─────────────────────────────────────────┐",
            "  │          LATENCY (milliseconds)          │",
            "  ├──────────────────┬────────┬──────────────┤",
            "  │ Stage            │   P50  │     P95      │",
            "  ├──────────────────┼────────┼──────────────┤",
            f"  │ Total            │ {summary['total_latency']['p50_ms']:6.0f} │ {summary['total_latency']['p95_ms']:12.0f} │",
            f"  │ Retrieval        │ {summary['retrieval_latency']['p50_ms']:6.0f} │ {summary['retrieval_latency']['p95_ms']:12.0f} │",
            f"  │ Generation       │ {summary['generation_latency']['p50_ms']:6.0f} │ {summary['generation_latency']['p95_ms']:12.0f} │",
            "  └──────────────────┴────────┴──────────────┘",
            "",
        ]

        # Add breakdown
        if "breakdown" in report.get("detailed", {}):
            bd = report["detailed"]["breakdown"]
            lines.extend([
                "  ┌─────────────────────────────────────────┐",
                "  │         TIME BREAKDOWN (avg %)           │",
                "  ├─────────────────────────────────────────┤",
                f"  │ Retrieval + Re-ranking:  {bd['avg_retrieval_pct']:5.1f}%           │",
                f"  │ LLM Generation:          {bd['avg_generation_pct']:5.1f}%           │",
                f"  │ Overhead (other):        {bd['avg_overhead_pct']:5.1f}%           │",
                "  └─────────────────────────────────────────┘",
                "",
            ])

        text = "\n".join(lines)
        print(text)
        return text

    def save_report(self, report: dict = None, filename: str = None) -> Path:
        """
        Save report as JSON file.

        Args:
            report: Report dict (generates if not provided)
            filename: Output filename

        Returns:
            Path to saved file
        """
        if report is None:
            report = self.generate_report()

        output_path = (
            config.metrics_output_dir / (filename or "performance_report.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📊 Report saved to {output_path}")
        return output_path
