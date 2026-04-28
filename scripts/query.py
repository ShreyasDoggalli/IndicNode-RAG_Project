#!/usr/bin/env python3
"""
query.py — Interactive Query Script
======================================
CLI tool to query the RAG system interactively.

Usage:
    python scripts/query.py                                  # Interactive mode
    python scripts/query.py -q "What is machine learning?"   # Single query
    python scripts/query.py --no-adaptive                    # Disable adaptive layer
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Query the Adaptive RAG system"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Single query to process (omit for interactive mode)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive decision layer (use static parameters)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Load saved indices
    try:
        pipeline.load_indices()
    except FileNotFoundError:
        print("❌ No indices found. Run ingestion first:")
        print("   python scripts/ingest.py")
        sys.exit(1)

    use_adaptive = not args.no_adaptive

    if args.query:
        # Single query mode
        result = pipeline.query(
            args.query,
            use_adaptive=use_adaptive,
            verbose=not args.json,
        )

        if args.json:
            print(json.dumps(result, indent=2, default=str))
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("  🤖 Adaptive RAG — Interactive Mode")
        print("=" * 60)
        print("  Type your queries below.")
        print("  Commands: 'quit' | 'report' | 'cache' | 'feedback'")
        print("=" * 60 + "\n")

        while True:
            try:
                query = input("❓ Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! 👋")
                break

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋")
                break

            if query.lower() == "report":
                pipeline.report()
                continue

            if query.lower() == "cache":
                print(f"\n📦 Cache stats: {pipeline.cache.stats()}\n")
                continue

            if query.lower() == "feedback":
                summary = pipeline.feedback_loop.get_metrics_summary()
                print(f"\n📊 Feedback: {json.dumps(summary, indent=2)}\n")
                continue

            # Process query
            result = pipeline.query(
                query,
                use_adaptive=use_adaptive,
                verbose=True,
            )

        # Final report
        pipeline.report()


if __name__ == "__main__":
    main()
