"""
query_cache.py — Smart Query Caching (Bonus)
===============================================
Caches query results to avoid redundant computation.

TERMINOLOGY:
    - LRU Cache (Least Recently Used): A cache that evicts the oldest
      unused entries when full. If the cache holds 100 entries and a 101st
      arrives, the least recently accessed entry is removed.

    - Exact Cache: Matches queries by exact string equality.
      "What is ML?" hits, but "What is machine learning?" misses.
      Fast but limited — users rarely ask the exact same question twice.

    - Semantic Cache: Matches queries by meaning similarity.
      "What is ML?" and "Define machine learning" would match because
      their embeddings are similar (cosine similarity > threshold).
      More useful but more expensive (requires embedding computation).

    - Cache Hit/Miss:
        * Hit: The query (or a similar one) was found in cache → return cached result
        * Miss: No match found → run the full pipeline

    - Cache Invalidation: Deciding when cached results are stale.
      If you add new documents, old cached answers might be wrong.
      We handle this by clearing the cache on re-ingestion.

    - TTL (Time To Live): How long a cached entry remains valid.
      After TTL expires, the entry is removed even if the cache isn't full.

WHY CACHE?
    - LLM generation is slow (1-5 seconds per query)
    - Many users ask similar questions
    - Cache hit → instant response (~1ms vs ~2000ms)
    - Reduces LLM API costs (if using OpenAI)
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import config


@dataclass
class CacheEntry:
    """
    A cached query result.

    Attributes:
        query: The original query string
        query_embedding: The query's embedding vector (for semantic matching)
        response: The cached response text
        confidence: Quality score of the cached response
        timestamp: When this entry was created
        hit_count: How many times this entry was returned
    """

    query: str
    query_embedding: np.ndarray
    response: str
    confidence: float
    timestamp: float
    hit_count: int = 0


class QueryCache:
    """
    Smart query cache with exact and semantic matching.

    Two-tier lookup:
        1. Exact match: O(1) hash lookup by query string
        2. Semantic match: O(n) scan comparing query embeddings

    The semantic layer catches paraphrases and similar questions,
    dramatically improving cache hit rates.

    Usage:
        cache = QueryCache(max_size=100, similarity_threshold=0.95)
        cache.put(query, embedding, response, confidence)
        result = cache.get(new_query, new_embedding)
    """

    def __init__(
        self,
        max_size: int = None,
        similarity_threshold: float = None,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        """
        Args:
            max_size: Maximum cache entries. Defaults to config.cache_max_size
            similarity_threshold: Minimum cosine similarity for semantic match.
                                  Defaults to config.cache_similarity_threshold
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size or config.cache_max_size
        self.similarity_threshold = (
            similarity_threshold or config.cache_similarity_threshold
        )
        self.ttl_seconds = ttl_seconds

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stats
        self.hits = 0
        self.misses = 0

    def get(
        self,
        query: str,
        query_embedding: np.ndarray = None,
    ) -> Optional[CacheEntry]:
        """
        Look up a query in the cache.

        First tries exact match, then falls back to semantic match.

        Args:
            query: The query string
            query_embedding: Embedding vector for semantic matching

        Returns:
            CacheEntry if found, None if cache miss
        """
        if not config.cache_enabled:
            return None

        # Step 1: Exact match (fast)
        normalized_query = query.strip().lower()
        if normalized_query in self._cache:
            entry = self._cache[normalized_query]

            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[normalized_query]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(normalized_query)
            entry.hit_count += 1
            self.hits += 1
            return entry

        # Step 2: Semantic match (slower, but catches paraphrases)
        if query_embedding is not None:
            best_match = self._semantic_search(query_embedding)
            if best_match is not None:
                best_match.hit_count += 1
                self.hits += 1
                return best_match

        self.misses += 1
        return None

    def put(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: str,
        confidence: float,
    ) -> None:
        """
        Add a query result to the cache.

        Args:
            query: The query string
            query_embedding: The query's embedding vector
            response: The response to cache
            confidence: Quality score of the response
        """
        if not config.cache_enabled:
            return

        normalized_query = query.strip().lower()

        # Evict oldest entry if cache is full
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest

        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            response=response,
            confidence=confidence,
            timestamp=time.time(),
        )

        self._cache[normalized_query] = entry

    def _semantic_search(
        self,
        query_embedding: np.ndarray,
    ) -> Optional[CacheEntry]:
        """
        Find semantically similar cached queries.

        Computes cosine similarity between the query embedding and
        all cached query embeddings. Returns the best match if
        above the similarity threshold.
        """
        if not self._cache:
            return None

        query_embedding = query_embedding.flatten()
        best_score = -1
        best_entry = None

        for entry in self._cache.values():
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                continue

            # Cosine similarity (embeddings are normalized)
            cached_emb = entry.query_embedding.flatten()
            similarity = np.dot(query_embedding, cached_emb)

            if similarity > best_score:
                best_score = similarity
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry is not None:
            return best_entry

        return None

    def clear(self) -> None:
        """Clear the entire cache. Call after re-ingesting documents."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        print("🗑️  Cache cleared")

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1f}%",
        }
