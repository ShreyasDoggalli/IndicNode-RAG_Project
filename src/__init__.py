"""
Adaptive RAG Inference System
==============================
A mini RAG pipeline that optimizes itself at inference time.

Components:
    - ingestion: Document loading and chunking
    - indexing: Vector (FAISS) and keyword (BM25) indexing
    - retrieval: Hybrid retrieval with re-ranking
    - generation: LLM-based answer generation
    - adaptive: Query analysis, decision engine, feedback loop
    - cache: Query caching layer
    - metrics: Performance tracking and reporting
"""

__version__ = "1.0.0"
__author__ = "Shreyas Doggalli"
