#!/usr/bin/env python3
"""
benchmark.py — Performance Benchmarking Script
=================================================
Runs a set of queries with and without adaptive logic to measure impact.

Usage:
    python scripts/benchmark.py                  # Run default benchmarks
    python scripts/benchmark.py --queries 50     # Custom query count
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline

# Benchmark queries covering different complexity levels
BENCHMARK_QUERIES = [
    # SIMPLE queries (short, factual)
    "What is machine learning?",
    "Define artificial intelligence.",
    "What is FAISS?",
    "What is a neural network?",
    "Define deep learning.",
    "What is NLP?",
    "What is supervised learning?",

    # MEDIUM queries (multi-concept)
    "How does gradient descent optimize neural networks?",
    "What is the difference between classification and regression?",
    "How do transformers use attention mechanisms?",
    "Explain the concept of transfer learning in deep learning.",
    "What role does regularization play in preventing overfitting?",

    # COMPLEX queries (multi-part, analytical)
    "Compare and contrast supervised learning and unsupervised learning, including their use cases and limitations.",
    "Explain how convolutional neural networks process images and why they are better than fully connected networks for image tasks.",
    "What are the tradeoffs between using a larger language model versus a smaller one, and how does this affect latency and accuracy?",
    "How do embedding models convert text to vectors, and why is cosine similarity used to measure their similarity?",
    "Explain the complete RAG pipeline from document ingestion to answer generation, including the role of each component.",

    # EDGE CASES
    "Why?",  # Very short
    "Tell me everything about AI, ML, deep learning, transformers, and how they relate to each other in modern applications.",  # Very long
    "FAISS BM25 vector search",  # Keyword-heavy
]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Adaptive RAG system"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=len(BENCHMARK_QUERIES),
        help=f"Number of queries to run (max: {len(BENCHMARK_QUERIES)})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare adaptive vs static mode",
    )

    args = parser.parse_args()
    queries = BENCHMARK_QUERIES[:args.queries]

    print("\n" + "=" * 60)
    print("  🏋️ Performance Benchmark")
    print("=" * 60)
    print(f"  Queries: {len(queries)}")
    print(f"  Compare mode: {args.compare}")
    print("=" * 60 + "\n")

    # Initialize pipeline
    pipeline = RAGPipeline()

    try:
        pipeline.load_indices()
    except FileNotFoundError:
        print("❌ No indices found. Run ingestion first:")
        print("   python scripts/ingest.py")
        sys.exit(1)

    # ─── Run Adaptive Mode ───────────────────────────────────
    print("\n📊 Running queries with ADAPTIVE mode...")
    adaptive_results = []
    for i, query in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] ", end="")
        result = pipeline.query(query, use_adaptive=True, verbose=False)
        adaptive_results.append(result)
        status = "CACHED" if result["cached"] else f"{result['timings']['total_ms']:.0f}ms"
        print(f"  {status} | {query[:50]}...")

    # Generate report
    print("\n\n" + "=" * 60)
    print("  📈 ADAPTIVE MODE RESULTS")
    print("=" * 60)
    adaptive_report = pipeline.report(save=True)

    # ─── Compare with Static Mode ────────────────────────────
    if args.compare:
        print("\n\n📊 Running queries with STATIC mode...")
        # Reset tracker for static mode
        from src.metrics.tracker import PerformanceTracker
        from src.metrics.reporter import PerformanceReporter
        pipeline.tracker = PerformanceTracker()
        pipeline.reporter = PerformanceReporter(pipeline.tracker)
        pipeline.cache.clear()

        static_results = []
        for i, query in enumerate(queries, 1):
            print(f"\n  [{i}/{len(queries)}] ", end="")
            result = pipeline.query(query, use_adaptive=False, verbose=False)
            static_results.append(result)
            status = "CACHED" if result["cached"] else f"{result['timings']['total_ms']:.0f}ms"
            print(f"  {status} | {query[:50]}...")

        print("\n\n" + "=" * 60)
        print("  📈 STATIC MODE RESULTS")
        print("=" * 60)
        static_report = pipeline.report(save=False)

        # ─── Comparison ──────────────────────────────────────
        print("\n\n" + "=" * 60)
        print("  📊 ADAPTIVE vs STATIC COMPARISON")
        print("=" * 60)

        if "summary" in adaptive_report and "summary" in static_report:
            a = adaptive_report["summary"]
            s = static_report["summary"]

            print(f"\n  {'Metric':<25} {'Adaptive':>12} {'Static':>12} {'Delta':>12}")
            print("  " + "─" * 61)

            for metric in ["total_latency", "retrieval_latency", "generation_latency"]:
                a_val = a[metric]["p50_ms"]
                s_val = s[metric]["p50_ms"]
                delta = a_val - s_val
                delta_str = f"{delta:+.0f}ms"
                print(f"  {metric + ' (P50)':<25} {a_val:>10.0f}ms {s_val:>10.0f}ms {delta_str:>12}")

                a_val = a[metric]["p95_ms"]
                s_val = s[metric]["p95_ms"]
                delta = a_val - s_val
                delta_str = f"{delta:+.0f}ms"
                print(f"  {metric + ' (P95)':<25} {a_val:>10.0f}ms {s_val:>10.0f}ms {delta_str:>12}")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
