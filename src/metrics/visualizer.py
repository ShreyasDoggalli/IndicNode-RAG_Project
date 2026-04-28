"""
visualizer.py — Performance Visualization
============================================
Generates charts and plots from performance data.

TERMINOLOGY:
    - Latency Distribution Plot: A histogram showing how response times
      are spread. Helps identify if there's a "long tail" of slow queries.

    - Time Series Plot: Shows how metrics change over time (query number).
      Useful for seeing if the adaptive layer improves over time.

    - Pipeline Breakdown Chart: A stacked bar or pie chart showing how
      much time each pipeline stage takes relative to the total.

    - Adaptive vs Static Comparison: Shows side-by-side performance
      with and without the adaptive decision layer.

HOW IT WORKS:
    1. Read timing data from PerformanceTracker
    2. Generate matplotlib charts  
    3. Save as PNG files to metrics_output/
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np

from src.config import config
from src.metrics.tracker import PerformanceTracker


class PerformanceVisualizer:
    """
    Generates performance charts from tracking data.

    All charts are saved as PNG files in the metrics output directory.

    Usage:
        visualizer = PerformanceVisualizer(tracker)
        visualizer.plot_all()
    """

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.output_dir = config.metrics_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "ggplot")

    def plot_all(self) -> List[Path]:
        """Generate all charts. Returns list of saved file paths."""
        paths = []
        timings = self.tracker.get_timings_data()

        if not timings:
            print("⚠️  No data to visualize")
            return paths

        paths.append(self.plot_latency_distribution(timings))
        paths.append(self.plot_latency_over_time(timings))
        paths.append(self.plot_pipeline_breakdown(timings))
        paths.append(self.plot_retrieval_vs_generation(timings))

        print(f"📈 Generated {len(paths)} charts in {self.output_dir}")
        return paths

    def plot_latency_distribution(self, timings: List[dict]) -> Path:
        """
        Plot histogram of total latency distribution.

        Shows P50 and P95 lines for reference.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        total_latencies = [t["total_ms"] for t in timings]

        ax.hist(
            total_latencies,
            bins=min(20, len(total_latencies)),
            color="#4C72B0",
            edgecolor="white",
            alpha=0.8,
        )

        # Add P50 and P95 lines
        p50 = np.percentile(total_latencies, 50)
        p95 = np.percentile(total_latencies, 95)

        ax.axvline(p50, color="#E74C3C", linestyle="--", linewidth=2, label=f"P50: {p50:.0f}ms")
        ax.axvline(p95, color="#F39C12", linestyle="--", linewidth=2, label=f"P95: {p95:.0f}ms")

        ax.set_xlabel("Total Latency (ms)", fontsize=12)
        ax.set_ylabel("Query Count", fontsize=12)
        ax.set_title("Latency Distribution", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)

        path = self.output_dir / "latency_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_latency_over_time(self, timings: List[dict]) -> Path:
        """
        Plot latency over successive queries.

        Shows if the adaptive system improves over time.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        query_nums = range(1, len(timings) + 1)
        total = [t["total_ms"] for t in timings]
        retrieval = [t["retrieval_ms"] for t in timings]
        generation = [t["generation_ms"] for t in timings]

        ax.plot(query_nums, total, "o-", color="#4C72B0", label="Total", linewidth=2, markersize=4)
        ax.plot(query_nums, retrieval, "s-", color="#55A868", label="Retrieval", linewidth=1.5, markersize=3)
        ax.plot(query_nums, generation, "^-", color="#C44E52", label="Generation", linewidth=1.5, markersize=3)

        ax.set_xlabel("Query Number", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title("Latency Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)

        path = self.output_dir / "latency_over_time.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_pipeline_breakdown(self, timings: List[dict]) -> Path:
        """
        Plot average time breakdown by pipeline stage.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        stages = {
            "Query Analysis": np.mean([t["query_analysis_ms"] for t in timings]),
            "Retrieval": np.mean([t["retrieval_ms"] for t in timings]),
            "Re-ranking": np.mean([t["reranking_ms"] for t in timings]),
            "Prompt Building": np.mean([t["prompt_building_ms"] for t in timings]),
            "LLM Generation": np.mean([t["generation_ms"] for t in timings]),
            "Response Parsing": np.mean([t["response_parsing_ms"] for t in timings]),
        }

        # Filter out zero stages
        stages = {k: v for k, v in stages.items() if v > 0}

        if not stages:
            plt.close(fig)
            return self.output_dir / "pipeline_breakdown.png"

        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
        
        wedges, texts, autotexts = ax.pie(
            stages.values(),
            labels=stages.keys(),
            autopct="%1.1f%%",
            colors=colors[:len(stages)],
            startangle=90,
            pctdistance=0.85,
        )

        # Make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        ax.add_artist(centre_circle)

        ax.set_title(
            "Pipeline Time Breakdown (avg)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        path = self.output_dir / "pipeline_breakdown.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_retrieval_vs_generation(self, timings: List[dict]) -> Path:
        """
        Plot retrieval time vs generation time as a scatter plot.

        Helps identify which stage is the bottleneck.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        retrieval = [t["retrieval_ms"] + t.get("reranking_ms", 0) for t in timings]
        generation = [t["generation_ms"] for t in timings]

        ax.scatter(
            retrieval,
            generation,
            c="#4C72B0",
            alpha=0.7,
            edgecolors="white",
            s=80,
        )

        # Add diagonal line
        max_val = max(max(retrieval, default=0), max(generation, default=0))
        if max_val > 0:
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal time")

        ax.set_xlabel("Retrieval Time (ms)", fontsize=12)
        ax.set_ylabel("Generation Time (ms)", fontsize=12)
        ax.set_title(
            "Retrieval vs Generation Time",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)

        path = self.output_dir / "retrieval_vs_generation.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
