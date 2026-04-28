"""
config.py — Central Configuration
===================================
This module manages all configuration for the RAG system.

TERMINOLOGY:
    - Environment Variables: System-level settings loaded from a .env file,
      allowing configuration without code changes.
    - Hyperparameters: Tunable values that control model/system behavior
      (e.g., chunk_size, top_k, embedding dimensions).

HOW IT WORKS:
    1. Loads .env file using python-dotenv
    2. Reads environment variables with sensible defaults
    3. Exposes a single Config dataclass used everywhere in the system
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    """
    Central configuration for the Adaptive RAG system.

    All settings are loaded from environment variables with defaults,
    so the system works out-of-the-box without any .env file.
    """

    # ─── LLM Settings ───────────────────────────────────────────
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # ─── Embedding Settings ─────────────────────────────────────
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = 384  # Dimension for all-MiniLM-L6-v2

    # ─── Re-ranker Settings ─────────────────────────────────────
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ─── Chunking Settings ──────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ─── Retrieval Settings ─────────────────────────────────────
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "15"))
    min_top_k: int = int(os.getenv("MIN_TOP_K", "2"))

    # ─── Hybrid Retrieval Weights ───────────────────────────────
    # alpha = weight for vector search; (1 - alpha) = weight for keyword search
    hybrid_alpha: float = 0.7

    # ─── Adaptive Settings ──────────────────────────────────────
    latency_threshold_ms: float = float(
        os.getenv("LATENCY_THRESHOLD_MS", "2000")
    )
    quality_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.5"))

    # ─── Cache Settings ─────────────────────────────────────────
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "100"))
    cache_similarity_threshold: float = float(
        os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")
    )

    # ─── Paths ──────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    vector_store_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "vector_store"
    )
    bm25_store_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "bm25_store"
    )
    metrics_output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "metrics_output"
    )

    def __post_init__(self):
        """Create necessary directories."""
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_store_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
