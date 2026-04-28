#!/usr/bin/env python3
"""
ingest.py — Document Ingestion Script
========================================
CLI tool to ingest documents into the RAG system.

Usage:
    python scripts/ingest.py                          # Ingest from default data/sample_docs/
    python scripts/ingest.py --dir /path/to/docs/     # Ingest from custom directory
    python scripts/ingest.py --chunk-size 256          # Custom chunk size
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.ingestion.loader import DocumentLoader
from src.ingestion.preprocessor import TextPreprocessor
from src.ingestion.chunker import TextChunker
from src.indexing.embedder import Embedder
from src.indexing.faiss_store import FAISSStore
from src.indexing.bm25_store import BM25Store


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Adaptive RAG system"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(config.data_dir / "sample_docs"),
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.chunk_size,
        help=f"Chunk size in characters (default: {config.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.chunk_overlap,
        help=f"Chunk overlap in characters (default: {config.chunk_overlap})",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  📥 Document Ingestion")
    print("=" * 60)
    print(f"  Source: {args.dir}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Chunk overlap: {args.chunk_overlap}")
    print("=" * 60 + "\n")

    # Step 1: Load
    loader = DocumentLoader()
    documents = loader.load_directory(args.dir)

    if not documents:
        print("❌ No documents found. Check the directory path.")
        sys.exit(1)

    # Step 2: Preprocess
    preprocessor = TextPreprocessor()
    documents = preprocessor.preprocess_documents(documents)

    # Step 3: Chunk
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)

    # Step 4: Embed
    embedder = Embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    # Step 5: Build FAISS index
    faiss_store = FAISSStore(dimension=embedder.dimension)
    faiss_store.add_documents(chunks, embeddings)
    faiss_store.save()

    # Step 6: Build BM25 index
    bm25_store = BM25Store()
    bm25_store.build_index(chunks)
    bm25_store.save()

    print("\n" + "=" * 60)
    print("  ✅ Ingestion Complete!")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Vectors indexed: {faiss_store.size}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
